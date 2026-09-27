"""A deterministic fake system that worker processes can build by import (the --workers tests).

Module-level so a spawned worker process can import it by name: pickling sends the
factory's qualified name, never an object.
"""

from __future__ import annotations

from evals.base import MemorySystem


class CountingSystem(MemorySystem):
    """Context = the question's own id and how many sessions it was fed; one fake LLM call."""

    name = "counting"

    def __init__(self) -> None:
        self._sessions = 0

    def reset(self) -> None:
        self._sessions = 0

    def add(self, messages: list[dict]) -> None:
        self._sessions += 1

    def get_context(self, query: str) -> str:
        return f"{query}|{self._sessions}"

    def retrieved_ids(self) -> list[str] | None:
        return [f"n{self._sessions}"]

    def take_llm_usage(self) -> dict:
        return {"fake-model": {"calls": 1, "prompt": 5, "completion": 1}}


def make_counting_system() -> CountingSystem:
    return CountingSystem()
