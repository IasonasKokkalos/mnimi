"""Drive a MemorySystem through LongMemEval.

Per question: reset the system, feed every haystack session via ``add``, ask for
assembled context via ``get_context``, hand context+question to a reader model,
then score the answer with the LLM judge. The reader is a local Ollama model
(Phase A: ``qwen2.5:1.5b-instruct-q4_0``); the judge runs on the OpenAI API
(``gpt-4o-2024-08-06``). No Anthropic client is built or used on this path.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from string import Template

from .artifacts import canonical, fingerprint
from .base import MemorySystem
from .dataset import DEFAULT_SAMPLE_SEED, SAMPLE_STRATIFIED, Question, Session, load
from .judge import Judge

# mnimi-con-v1: LongMemEval's Chain-of-Note reading prompt, byte-grounded in
# the reference implementation (src/generation/run_generation.py line 55 in
# github.com/xiaowu0162/LongMemEval — Figure 13 typesets this template; the
# code is the byte authority, same precedent as the judge templates), with
# exactly two deviations:
#
#   1. The abstention sentence ("If the chat history does not contain
#      enough..."), inserted immediately after the step-by-step instruction.
#      30 dataset questions are abstention questions and Figure 13 has no
#      abstention route.
#   2. The ${cache_bust} prefix — determinism plumbing, not instruction: it
#      forces the shared prefill prefix against any other question to zero
#      length so every prefill starts at n_past=0. See CACHE STATE below.
#
# Slot mapping against the reference: {chat history} -> ${context},
# {question date} -> ${question_date}, {question} -> ${question}. Explicit
# substitution (string.Template), never str.format: the reference's rendered
# values are chat text that can contain any brace or dollar character, and
# Template.substitute never rescans substituted values.
#
# The reference sends this as a SINGLE USER MESSAGE
# (run_generation.py: messages=[{"role": "user", "content": prompt}]) and so
# does this harness — the old harness-invented system message is gone, and its
# absence is a recorded value in reader_request_shape() below.
READER_TEMPLATE = (
    "${cache_bust}\n\n"
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history. Answer the question step by "
    "step: first extract all the relevant information, and then reason over the "
    "information to get the answer. If the chat history does not contain enough "
    "information to answer the question, say that you do not know rather than "
    "guessing."
    "\n\n\nHistory Chats:\n\n${context}\n\nCurrent Date: ${question_date}\n"
    "Question: ${question}\nAnswer (step by step):"
)
_READER_T = Template(READER_TEMPLATE)

# Bumped whenever the reader prompt changes shape.
# v2 (plain-prose-v2): added the per-question cache-bust prefix.
# mnimi-con-v1: Figure 13 CoN body, single user message, question_date
# consumed ("Current Date:"), abstention sentence added. json-con-paper-v1
# stays RESERVED for a byte-exact reproduction of the paper's prompt with no
# deviations at all; this is not it and is not named as it.
READER_PROMPT_VERSION = "mnimi-con-v1"

# Decode config, pinned. temperature=0 alone does NOT give greedy decoding — it
# leaves the sampler free to break ties differently between runs, which was
# observed in practice: identical inputs produced "I don't have access" on one
# run and "I do not have access" on the next. top_k=1 forces greedy; the seed
# pins whatever sampling survives. Both go in the header, because they
# determine the number.
READER_SEED = 0
READER_TOP_K = 1

# Full GPU offload. The history here is load-bearing, so both the wrong finding
# and its correction are recorded rather than the correction alone — see
# docs/DECISIONS.md for the full bisection.
#
#   First conclusion (WRONG): "GPU is non-deterministic; CPU-only is required."
#   Three identical GPU calls returned two distinct answers at temperature=0,
#   which was attributed to CUDA atomics reordering float reductions.
#
#   Why it was wrong: that test never pinned num_batch. Ollama was free to pick
#   a batch size per load, and llama.cpp logits are not bit-identical across
#   batch sizes — so the run-to-run drift came from an unpinned harness setting,
#   not from the GPU. An unpinned batch size mimics the signature of CUDA
#   atomics exactly, which is what made the misattribution easy.
#
#   Re-tested with num_batch pinned: byte-identical output across (1) repeat
#   calls, (2) a cold model reload, (3) a full machine reboot, and (4) 91% GPU
#   utilisation under contention. GPU is reproducible AND ~23x faster on a 32k
#   prefill (10.4s vs 237.2s), so it is the pinned default.
READER_NUM_GPU = 99

# LOAD-BEARING. This is the pin whose absence produced the wrong conclusion
# above; it must be sent explicitly on every call and never left to Ollama.
READER_NUM_BATCH = 512

# Measured to have NO effect on output at full offload: num_thread 4, 8 and 16
# produce byte-identical text when num_gpu=99, because the compute that decides
# the logits is on the GPU. Kept and pinned anyway because --num-gpu can drop
# the reader back to CPU, where thread count IS load-bearing (it changes the
# float reduction order across loads). It is a real pin for the CPU path, not a
# dependency of the GPU path.
READER_NUM_THREAD = 8

# Flash attention, pinned. FA changes attention tiling, which changes the order
# of float reductions, which flips argmax at near-ties: with cache state held
# constant, FA=0 vs FA=1 changed 2/2 probe predictions (divergence at byte 0 of
# a 1,924-char answer, and at byte 255 of a 383-char one).
#
# Left unset, the daemon resolves `flash_attn = auto`, and what `auto` picks is
# a property of the host GPU, not of this configuration — so an unpinned run is
# only reproducible on one machine. Pinned to 1 (enabled) because that is what
# `auto` already resolved to on the development GPU, which keeps continuity with
# every run recorded so far, and because it is the faster kernel. The value
# matters less than the fact that it is stated: 0 is equally reproducible.
#
# This is a DAEMON-level setting. It cannot be sent per request — it is read
# from OLLAMA_FLASH_ATTENTION when the daemon starts. Hence the run precondition
# in `preflight_reader_env`.
READER_FLASH_ATTENTION = 1

# CACHE STATE. Load-bearing, and the pin that closed the drift that survived
# `num_batch`. Pinning num_batch is NOT sufficient: llama.cpp keeps a
# content-addressed prompt cache, and a cache hit recomputes the final logits in
# a batch of ONE token ("need to evaluate at least 1 token", n_past=27622)
# instead of inside the 512-token prefill batch a cold pass uses. Different
# batch size at the logits position, different reduction order, different
# argmax — the exact mechanism num_batch was pinned to prevent, re-entering
# through the cache. Both states are individually deterministic and both
# survive a daemon restart, so this presents as a bistable output, not as noise.
#
# `cache_prompt: false` in the request options is IGNORED by this Ollama build
# (measured). The only switch that works is llama.cpp's `--cache-ram 0`, reached
# through the LLAMA_ARG_CACHE_RAM passthrough on the daemon environment — so,
# like flash attention, it is a daemon-level precondition rather than a
# per-request option.
READER_CACHE_RAM = 0

# The daemon environment this harness requires. Both are resolved at daemon
# start, so a run served by a daemon launched without them is not the
# configuration these pins describe, whatever pins.json says.
REQUIRED_OLLAMA_ENV = {
    "OLLAMA_FLASH_ATTENTION": str(READER_FLASH_ATTENTION),
    "LLAMA_ARG_CACHE_RAM": str(READER_CACHE_RAM),
}


def reader_request_shape() -> dict:
    """The reader request as sent, with per-question values left unrendered.

    Mirrors ``judge_request_shape``: roles and the (absent) system slot are
    recorded values, so restructuring the message stack moves the hash the
    same way editing the template does. ``${cache_bust}`` and
    ``${question_date}`` appear literally in the template text — per-question
    values stay out of the hash, the presence of their slots does not.
    """
    return {
        "system_message": None,  # None = no system message is sent
        "roles": ["user"],
        "template": READER_TEMPLATE,
    }


def reader_prompt_hash() -> str:
    """Digest of the exact reader request shape that produced a run.

    Hashes the unsubstituted template, not a rendered instance: the cache-bust
    prefix varies per question by design, so hashing a rendered prompt would
    make every question look like a different prompt configuration.
    """
    return fingerprint(canonical(reader_request_shape()))

# LongMemEval-S is ~115k tokens/question and this reader's window is ~32k, so
# full_history overflows. Rather than let Ollama silently drop tokens, the reader
# truncates explicitly (keep-most-recent, drop-oldest) and reports what it cut.
# The gate uses a char/token estimate; the exact fed count comes back from Ollama.
#
# Why a measured run feeds ~27k tokens and not ~32k. Two separate effects:
#   1. Deliberate reserve, 1,056 tokens: `answer_reserve` (800) keeps room for
#      the generation, `_SCAFFOLD_TOKENS` (256) for the system prompt and the
#      question framing. Without these the prompt could fill the window and
#      leave nothing to answer with.
#   2. A conservative estimate, ~4k tokens. The trim gate assumes 4 chars per
#      token, but this dataset actually runs ~4.7 (26,662 tokens measured from
#      125,952 chars). Assuming a denser encoding than reality means the gate
#      cuts more than it needs to, so the window is under-filled.
# Effect (1) is intentional; effect (2) is a known conservatism that costs
# roughly 4k tokens of usable evidence. Raising _CHARS_PER_TOKEN toward the
# measured ratio would recover it — but it changes every truncated number, so
# it is left alone here and flagged rather than tuned mid-phase.
_CHARS_PER_TOKEN = 4
_SCAFFOLD_TOKENS = 256  # headroom for the prompt template + question framing

# Generation budget, 800 to match the paper's max generation length (greedy).
#
# NOT a pure generation-length knob — read this before changing it. The value is
# used TWICE: as Ollama's `num_predict`, and as the reserve subtracted from the
# context window in `_fit`. Lowering it from 1024 to 800 therefore widened the
# trim budget by 224 tokens, which changes how much history reaches the reader
# on every truncated question, which changes full_history's predictions and its
# score. That coupling was accepted deliberately when the paper's 800 was
# adopted: full_history is a truncated-context baseline, not the ceiling, so
# moving its truncation point moves a number that was never a ceiling claim.
# All three trim inputs are pinned in `pins.json` (schema /2).
READER_ANSWER_RESERVE = 800


def reader_trim_pins() -> dict:
    """The trim gate's inputs, for `build_pins`.

    `num_ctx` alone does not determine what the reader sees; these three do the
    rest of the arithmetic, and full_history truncates on every question.
    """
    return {
        "reader_answer_reserve": READER_ANSWER_RESERVE,
        "reader_scaffold_tokens": _SCAFFOLD_TOKENS,
        "reader_chars_per_token": _CHARS_PER_TOKEN,
    }


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
        answer_reserve: int = READER_ANSWER_RESERVE,
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

    def answer(
        self,
        context: str,
        question: str,
        cache_bust: str = "",
        question_date: str = "",
    ) -> ReaderOutput:
        """Answer one question.

        ``question_date`` is the dataset's "asked on" date, consumed by
        ``mnimi-con-v1``'s ``Current Date:`` line — temporal questions are
        unanswerable without knowing when "now" is.
        """
        context, truncated, dropped = self._fit(context)
        # cache_bust leads the (single) user message so the shared prefix
        # against the previous request is zero-length. Measured: with the
        # prompt cache off, this takes prefill from "27,191 of 27,255 tokens"
        # (64 reused from a shared prefix, i.e. the boundary depends on the
        # PREVIOUS question) to the full token count every time.
        prompt = _READER_T.substitute(
            cache_bust=cache_bust,
            context=context,
            question_date=question_date,
            question=question,
        )
        # One user message, matching the reference implementation's request
        # (run_generation.py: messages=[{"role": "user", "content": prompt}]).
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
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


class ReaderEnvError(RuntimeError):
    """The serving daemon is not the configuration the pins claim."""


def preflight_reader_env(model_load_log: str, daemon_env: dict | None = None) -> None:
    """Fail fast when the daemon's RESOLVED settings contradict the pins.

    Checks resolved, not requested. `flash_attn = auto` in a load log is a
    request whose answer depends on the host GPU; `enabled`/`disabled` is what
    actually happened. The tray app is the specific hazard this guards against:
    it starts a daemon on 11434 with none of REQUIRED_OLLAMA_ENV set, and a run
    served by it silently carries `auto` and a live prompt cache.

    Raises ReaderEnvError rather than warning: a run under the wrong daemon is
    not a degraded number, it is a different configuration.
    """
    want_fa = "enabled" if READER_FLASH_ATTENTION else "disabled"
    m = re.search(r"flash_attn\s*=\s*(\w+)", model_load_log or "")
    resolved = m.group(1) if m else None
    if resolved is None:
        raise ReaderEnvError(
            "no resolved flash_attn line in the model load log — cannot confirm "
            "the daemon matches the pins. Launch the daemon manually with "
            + ", ".join(f"{k}={v}" for k, v in REQUIRED_OLLAMA_ENV.items())
        )
    if resolved != want_fa:
        hint = (
            " (`auto` means OLLAMA_FLASH_ATTENTION was unset — this is the tray "
            "app's daemon, not a manually launched one)"
            if resolved == "auto"
            else ""
        )
        raise ReaderEnvError(
            f"daemon resolved flash_attn={resolved}, pins require {want_fa}"
            f"{hint}. Kill `ollama`, `ollama app` AND `llama-server` (the child "
            "runners outlive the daemon and keep holding VRAM), then relaunch "
            f"with OLLAMA_FLASH_ATTENTION={READER_FLASH_ATTENTION}."
        )
    # The prompt cache must be confirmed OFF, not merely un-observed. The
    # disabled daemon says so in a dedicated line and emits no `cache state`
    # line at all, so keying off the limit alone would treat "no evidence" as a
    # pass — which is the one outcome a preflight must never return.
    log = model_load_log or ""
    if re.search(r"prompt cache is disabled", log):
        return
    active = re.search(r"cache state:[^(]*\(limits:\s*([0-9.]+)\s*MiB", log)
    if active:
        raise ReaderEnvError(
            f"daemon prompt cache is ACTIVE (limit {active.group(1)} MiB), pins "
            f"require it disabled. Relaunch with "
            f"LLAMA_ARG_CACHE_RAM={READER_CACHE_RAM} — a live prompt cache "
            "recomputes the final logits in a batch of one on a hit, which "
            "changes the answer."
        )
    raise ReaderEnvError(
        "no prompt-cache state in the model load log — cannot confirm the cache "
        f"is disabled. Relaunch with LLAMA_ARG_CACHE_RAM={READER_CACHE_RAM} and "
        "point OLLAMA_SERVE_LOG at that daemon's log."
    )


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
    strategy: str = SAMPLE_STRATIFIED,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    progress: PredictProgressFn | None = None,
    reader_client=None,
) -> list[Prediction]:
    """Stage 1: ingest and read. No judge, no API key, no grading.

    This is the expensive half — it re-ingests every session and runs the local
    reader over every question — which is exactly why it is separable from the
    half that gets re-run.
    """
    questions = load(
        limit=limit, filename=dataset_file, strategy=strategy, seed=sample_seed
    )
    reader = Reader(reader_model, num_ctx=num_ctx, num_gpu=num_gpu, client=reader_client)

    predictions: list[Prediction] = []
    for i, q in enumerate(questions):
        system.reset()
        for session in _sessions_for(system, q):
            system.add(_session_to_messages(session))
        context = system.get_context(q.question)
        out = reader.answer(
            context, q.question, cache_bust=q.question_id, question_date=q.question_date
        )
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


def _sessions_for(system: MemorySystem, q: Question) -> list[Session]:
    """The sessions one system is fed for one question.

    Every system sees the full haystack except one that declares
    ``evidence_only`` — the oracle evidence-availability bound, which reads only the annotated
    evidence sessions. The branch is on a declared capability, never on a
    system's name, so the runner stays system-agnostic.

    The alternative — attaching ``session_id`` or an evidence flag to every
    message — was rejected: ``answer_session_ids`` marks the sessions that
    contain the answer, and a signal that identifies the answer has no business
    in the inputs of the systems being compared against each other. Selecting
    here keeps it out of every system except the one whose definition needs it.
    """
    if not system.evidence_only:
        return q.sessions
    evidence = set(q.answer_session_ids)
    return [s for s in q.sessions if s.session_id in evidence]


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
