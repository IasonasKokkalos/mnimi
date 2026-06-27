"""Drive a MemorySystem through LongMemEval.

Per question: reset the system, feed every haystack session via ``add``, ask for
assembled context via ``get_context``, hand context+question to a reader model,
then score the answer with the LLM judge. Reader and judge both run on the
Anthropic API (key from ``ANTHROPIC_API_KEY``).
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


@dataclass
class Result:
    question_id: str
    category: str
    is_abstention: bool
    correct: bool
    answer: str
    predicted: str


class Reader:
    """Anthropic-backed question answerer."""

    def __init__(self, model: str, client=None) -> None:
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client

    def answer(self, context: str, question: str) -> str:
        user = (
            f"# Memory context\n{context}\n\n# Question\n{question}"
            if context.strip()
            else question
        )
        # No temperature/thinking: Opus 4.8 rejects sampling params; a small
        # max_tokens keeps the (single, short) answer cheap.
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=READER_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()


ProgressFn = Callable[[int, int, Question, bool], None]


def run(
    system: MemorySystem,
    *,
    reader_model: str,
    judge_model: str,
    limit: int | None,
    dataset_file: str | None = None,
    progress: ProgressFn | None = None,
    client=None,
) -> list[Result]:
    """Evaluate ``system`` over the first ``limit`` questions; return per-question results.

    ``client`` lets a caller inject an Anthropic-compatible client (shared by the
    reader and judge); when ``None`` a real one is constructed.
    """
    questions = load(limit=limit, filename=dataset_file)

    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=4)
    reader = Reader(reader_model, client=client)
    judge = Judge(judge_model, client=client)

    results: list[Result] = []
    for i, q in enumerate(questions):
        system.reset()
        for session in q.sessions:
            system.add(_session_to_messages(session))
        context = system.get_context(q.question)
        predicted = reader.answer(context, q.question)
        correct = judge.is_correct(
            question=q.question,
            answer=q.answer,
            response=predicted,
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
                predicted=predicted,
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
