"""Truncated-context baseline: stuff the entire history into context."""

from __future__ import annotations

# The canonical context renderer. Imported, not reimplemented: this format is
# shared by every context-bearing arm (full_history, oracle via inheritance,
# mnimi and naive_rag through their record render path), and the only way
# format parity survives edits is for all of them to run the same code.
from mnimi.memory import RENDER_FORMAT_TEXT, render_turns

from ..base import MemorySystem


class FullHistorySystem(MemorySystem):
    """Concatenates every turn of every session into the context string.

    The truncated-context baseline — the reader sees the most recent window's
    worth of everything — at the token cost a real memory system has to beat.
    """

    name = "full_history"

    def __init__(self, render_format: str = RENDER_FORMAT_TEXT) -> None:
        self._turns: list[dict] = []
        # The same knob mnimi and naive_rag read from MemoryConfig; the harness
        # passes one value to every arm so the pinned render hash is the truth
        # for all of them.
        self._render_format = render_format

    def reset(self) -> None:
        self._turns = []

    def add(self, messages: list[dict]) -> None:
        self._turns.extend(messages)

    def get_context(self, query: str) -> str:
        return render_turns(self._turns, fmt=self._render_format)
