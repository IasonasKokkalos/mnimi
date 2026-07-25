"""Ceiling baseline: stuff the entire history into context."""

from __future__ import annotations

from ..base import MemorySystem


class FullHistorySystem(MemorySystem):
    """Concatenates every turn of every session into the context string.

    The accuracy ceiling — the reader sees everything — at the token cost a real
    memory system has to beat. mnimi's whole bet is approaching this number
    on a fraction of the tokens.
    """

    name = "full_history"

    def __init__(self) -> None:
        self._turns: list[dict] = []

    def reset(self) -> None:
        self._turns = []

    def add(self, messages: list[dict]) -> None:
        self._turns.extend(messages)

    def get_context(self, query: str) -> str:
        # A dated header is emitted whenever ``ts`` changes, so the session
        # boundary and its date stay reader-visible (temporal questions are
        # unanswerable without them) at one line per session rather than one
        # timestamp per turn.
        lines: list[str] = []
        current_ts = None
        for turn in self._turns:
            ts = turn.get("ts")
            if ts and ts != current_ts:
                lines.append(f"[Session date: {ts}]")
                current_ts = ts
            lines.append(f"{turn.get('role', '')}: {turn.get('content', '')}")
        return "\n".join(lines)
