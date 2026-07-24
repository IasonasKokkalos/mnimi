"""The contract every system under test implements."""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemorySystem(ABC):
    """Adapter interface for the eval harness.

    Deliberately separate from ``mnimi.Memory``: this is the rig's plug, so
    every system — no-memory floor, full-history ceiling, naive-RAG, mnimi —
    is driven identically and the numbers are comparable. Per question the runner
    calls :meth:`reset` once, feeds each session via :meth:`add`, then asks for
    the assembled context with :meth:`get_context`.
    """

    name: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """Drop all per-question state so the next question starts clean."""

    @abstractmethod
    def add(self, messages: list[dict]) -> None:
        """Ingest one session's worth of turns (each ``{"role", "content"}``)."""

    @abstractmethod
    def get_context(self, query: str) -> str:
        """Return the context string to hand the reader for ``query``."""
