"""Floor baseline: no memory at all."""

from __future__ import annotations

from ..base import MemorySystem


class NoMemorySystem(MemorySystem):
    """Ignores all history; the reader answers from the question alone.

    Whatever this scores is the share of LongMemEval answerable from world
    knowledge or guessing — the floor every real system must clear.
    """

    name = "no_memory"

    def reset(self) -> None:
        pass

    def add(self, messages: list[dict]) -> None:
        pass

    def get_context(self, query: str) -> str:
        return ""
