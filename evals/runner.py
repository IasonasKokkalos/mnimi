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
        client=None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.answer_reserve = answer_reserve
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
            # temperature=0 for determinism; num_ctx pinned so Ollama does not
            # silently fall back to its 2048/4096 default and truncate for us.
            options={
                "temperature": 0,
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


ProgressFn = Callable[[int, int, Question, bool], None]


def run(
    system: MemorySystem,
    *,
    reader_model: str,
    judge_model: str,
    limit: int | None,
    num_ctx: int,
    dataset_file: str | None = None,
    progress: ProgressFn | None = None,
    judge_cache=None,
    reader_client=None,
    judge_client=None,
) -> list[Result]:
    """Evaluate ``system`` over the first ``limit`` questions; return per-question results.

    The reader (Ollama) and judge (OpenAI) use separate transports — no shared
    client. ``reader_client`` / ``judge_client`` inject test doubles; ``judge_cache``
    is an optional :class:`~evals.judge_cache.JudgeCache` for call avoidance.
    """
    questions = load(limit=limit, filename=dataset_file)

    reader = Reader(reader_model, num_ctx=num_ctx, client=reader_client)
    judge = Judge(judge_model, client=judge_client, cache=judge_cache)

    results: list[Result] = []
    for i, q in enumerate(questions):
        system.reset()
        for session in q.sessions:
            system.add(_session_to_messages(session))
        context = system.get_context(q.question)
        out = reader.answer(context, q.question)
        correct = judge.is_correct(
            question_id=q.question_id,
            question=q.question,
            answer=q.answer,
            response=out.text,
            question_type=q.category,
            abstention=q.is_abstention,
        )
        results.append(
            Result(
                question_id=q.question_id,
                category=q.category,
                is_abstention=q.is_abstention,
                correct=correct,
                answer=q.answer,
                predicted=out.text,
                reader_prompt_tokens=out.prompt_tokens,
                truncated=out.truncated,
                tokens_dropped=out.tokens_dropped,
            )
        )
        if progress is not None:
            progress(i + 1, len(questions), q, correct)
    return results


def _session_to_messages(session: Session) -> list[dict]:
    """Turns for one session, prefixed with a date marker.

    The date marker keeps the session timestamp in the assembled context (it
    matters for temporal-reasoning) without widening the ABC's ``add(messages)``
    signature with a separate date argument.
    """
    messages: list[dict] = []
    if session.date:
        messages.append({"role": "system", "content": f"[Session date: {session.date}]"})
    messages.extend(session.turns)
    return messages
