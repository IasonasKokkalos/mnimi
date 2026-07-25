"""Drive a MemorySystem through LongMemEval.

Per question: reset the system, feed every haystack session via ``add``, ask for
assembled context via ``get_context``, hand context+question to a reader model,
then score the answer with the LLM judge. The reader is a local Ollama model
(Phase A: ``qwen2.5:1.5b-instruct-q4_0``); the judge runs on the OpenAI API
(``gpt-4o-2024-08-06``). No Anthropic client is built or used on this path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .artifacts import fingerprint
from .base import MemorySystem
from .dataset import Question, Session, load
from .judge import Judge

# The reader is intentionally plain — no extended thinking — so the variable
# under test is the quality of the memory context, not the reader's reasoning.
READER_SYSTEM = (
    "You are a helpful assistant with access to memory from your prior "
    "conversations with the user. Use the memory context to answer the user's "
    "question as accurately as possible. If the memory does not contain enough "
    "information to answer, say you don't know rather than guessing."
)

# Bumped whenever the reader prompt changes shape. Phase D replaces this with a
# JSON + Chain-of-Note prompt carrying the question date; the version string and
# the hash below both move then, which is what makes that swap loud in the header.
READER_PROMPT_VERSION = "plain-prose-v1"

# Decode config, pinned. temperature=0 alone does NOT give greedy decoding — it
# leaves the sampler free to break ties differently between runs, which was
# observed in practice: identical inputs produced "I don't have access" on one
# run and "I do not have access" on the next. top_k=1 forces greedy; the seed
# pins whatever sampling survives. Both go in the header, because they
# determine the number.
READER_SEED = 0
READER_TOP_K = 1
# CPU-only. Measured on this machine: with the model offloaded to GPU, three
# identical calls returned two distinct answers even at temperature=0/top_k=1 —
# GPU kernels reorder floating-point reductions, the logits shift, and greedy
# decoding flips at a near-tie and diverges mid-sentence. The same test on CPU
# returned one distinct answer. Reproducibility is the whole point of the
# header, so the default trades speed for a number that replays; --num-gpu
# overrides it for exploratory runs.
READER_NUM_GPU = 0
# Pinned for the same reason. Ollama chooses thread count at model-load time
# from whatever the machine looks like then, and the number of threads changes
# the order floating-point reductions happen in — so two loads of the same model
# can produce different logits. Fixing both removes the last load-time variable.
READER_NUM_THREAD = 8
READER_NUM_BATCH = 512


def reader_prompt_hash() -> str:
    """Digest of the exact reader prompt text that produced a run."""
    return fingerprint(READER_SYSTEM)

# LongMemEval-S is ~115k tokens/question and this reader's window is ~32k, so
# full_history overflows. Rather than let Ollama silently drop tokens, the reader
# truncates explicitly (keep-most-recent, drop-oldest) and reports what it cut.
# The gate uses a char/token estimate; the exact fed count comes back from Ollama.
_CHARS_PER_TOKEN = 4
_SCAFFOLD_TOKENS = 256  # headroom for the system prompt + question framing


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN if text else 0


@dataclass
class ReaderOutput:
    """A reader answer plus the token accounting for that call."""

    text: str
    prompt_tokens: int | None  # exact, from Ollama's prompt_eval_count
    truncated: bool
    tokens_dropped: int  # estimated (see _estimate_tokens)


@dataclass
class Prediction:
    """One reader answer, plus everything the judge stage needs to grade it.

    Carries ``question`` and the gold ``answer`` deliberately: that is what lets
    ``predictions.jsonl`` be judged on its own, with no dataset load and no
    re-ingestion, however long after the predict stage ran.
    """

    question_id: str
    category: str
    is_abstention: bool
    question: str
    answer: str
    predicted: str
    reader_prompt_tokens: int | None = None
    truncated: bool = False
    tokens_dropped: int = 0


@dataclass
class Result:
    question_id: str
    category: str
    is_abstention: bool
    correct: bool
    answer: str
    predicted: str
    reader_prompt_tokens: int | None = None
    truncated: bool = False
    tokens_dropped: int = 0


class Reader:
    """Ollama-backed question answerer with explicit context truncation."""

    def __init__(
        self,
        model: str,
        *,
        num_ctx: int,
        answer_reserve: int = 1024,
        seed: int = READER_SEED,
        top_k: int = READER_TOP_K,
        num_gpu: int = READER_NUM_GPU,
        num_thread: int = READER_NUM_THREAD,
        num_batch: int = READER_NUM_BATCH,
        client=None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.answer_reserve = answer_reserve
        self.seed = seed
        self.top_k = top_k
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.num_batch = num_batch
        if client is None:
            import ollama

            client = ollama
        self.client = client

    def _fit(self, context: str) -> tuple[str, bool, int]:
        """Trim context to the reader's budget, keeping the most recent text."""
        budget = self.num_ctx - self.answer_reserve - _SCAFFOLD_TOKENS
        if _estimate_tokens(context) <= budget:
            return context, False, 0
        keep_chars = max(0, budget * _CHARS_PER_TOKEN)
        kept = context[-keep_chars:]  # tail = most recent sessions (oldest first)
        dropped = _estimate_tokens(context) - _estimate_tokens(kept)
        return kept, True, dropped

    def answer(self, context: str, question: str) -> ReaderOutput:
        context, truncated, dropped = self._fit(context)
        user = (
            f"# Memory context\n{context}\n\n# Question\n{question}"
            if context.strip()
            else question
        )
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": user},
            ],
            # num_ctx pinned so Ollama does not silently fall back to its
            # 2048/4096 default and truncate for us; temperature/top_k/seed
            # pinned so the same input yields the same answer across runs.
            options={
                "temperature": 0,
                "top_k": self.top_k,
                "seed": self.seed,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_batch": self.num_batch,
                "num_ctx": self.num_ctx,
                "num_predict": self.answer_reserve,
            },
        )
        # ollama-python returns a ChatResponse (attr access) on >=0.4 and a plain
        # dict on older builds — support both without pinning to one shape.
        if isinstance(response, dict):
            text = response["message"]["content"]
            prompt_tokens = response.get("prompt_eval_count")
        else:
            text = response.message.content
            prompt_tokens = getattr(response, "prompt_eval_count", None)
        return ReaderOutput((text or "").strip(), prompt_tokens, truncated, dropped)


PredictProgressFn = Callable[[int, int, Question, bool], None]
JudgeProgressFn = Callable[[int, int, "Prediction", bool], None]


def predict(
    system: MemorySystem,
    *,
    reader_model: str,
    limit: int | None,
    num_ctx: int,
    num_gpu: int = READER_NUM_GPU,
    dataset_file: str | None = None,
    progress: PredictProgressFn | None = None,
    reader_client=None,
) -> list[Prediction]:
    """Stage 1: ingest and read. No judge, no API key, no grading.

    This is the expensive half — it re-ingests every session and runs the local
    reader over every question — which is exactly why it is separable from the
    half that gets re-run.
    """
    questions = load(limit=limit, filename=dataset_file)
    reader = Reader(reader_model, num_ctx=num_ctx, num_gpu=num_gpu, client=reader_client)

    predictions: list[Prediction] = []
    for i, q in enumerate(questions):
        system.reset()
        for session in q.sessions:
            system.add(_session_to_messages(session))
        context = system.get_context(q.question)
        out = reader.answer(context, q.question)
        predictions.append(
            Prediction(
                question_id=q.question_id,
                category=q.category,
                is_abstention=q.is_abstention,
                question=q.question,
                answer=q.answer,
                predicted=out.text,
                reader_prompt_tokens=out.prompt_tokens,
                truncated=out.truncated,
                tokens_dropped=out.tokens_dropped,
            )
        )
        if progress is not None:
            progress(i + 1, len(questions), q, out.truncated)
    return predictions


def judge_predictions(
    predictions: list[Prediction],
    *,
    judge_model: str,
    judge_cache=None,
    judge_client=None,
    progress: JudgeProgressFn | None = None,
) -> list[Result]:
    """Stage 2: grade predictions. Needs no dataset, no system, no reader.

    Everything required to grade travels in the prediction rows, so this stage
    replays from ``predictions.jsonl`` alone — and with the verdict cache warm,
    replays for free.
    """
    judge = Judge(judge_model, client=judge_client, cache=judge_cache)

    results: list[Result] = []
    for i, p in enumerate(predictions):
        correct = judge.is_correct(
            question_id=p.question_id,
            question=p.question,
            answer=p.answer,
            response=p.predicted,
            question_type=p.category,
            abstention=p.is_abstention,
        )
        results.append(
            Result(
                question_id=p.question_id,
                category=p.category,
                is_abstention=p.is_abstention,
                correct=correct,
                answer=p.answer,
                predicted=p.predicted,
                reader_prompt_tokens=p.reader_prompt_tokens,
                truncated=p.truncated,
                tokens_dropped=p.tokens_dropped,
            )
        )
        if progress is not None:
            progress(i + 1, len(predictions), p, correct)
    return results


def _session_to_messages(session: Session) -> list[dict]:
    """Turns for one session, each carrying the session timestamp as ``ts``.

    The timestamp travels as structured data on every turn rather than as an
    injected ``{"role": "system"}`` text marker. The marker was a leak: every
    system under test received a pseudo-turn the harness invented and had to
    know to handle. ``ts`` is now part of the ABC contract instead — systems
    that need logical time read the field, and the rendering choice belongs to
    whoever assembles context.
    """
    return [
        {
            "role": turn.get("role", ""),
            "content": turn.get("content", ""),
            "ts": session.date,
        }
        for turn in session.turns
    ]
