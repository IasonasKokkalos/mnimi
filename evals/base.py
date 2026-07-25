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
        """Ingest one session's worth of turns.

        Each message is ``{"role", "content", "ts"}``. ``ts`` is the session
        timestamp supplied by the dataset (ISO date string) and carried as
        structured data on every turn — not as an injected pseudo-turn — so a
        third-party adapter never has to know to strip a harness invention.

        **``ts`` is the only clock.** A system that needs "now" for recency,
        decay, or ordering derives it from the ``ts`` values it has been fed
        and never reads wall-clock: wall-clock makes the same inputs score
        differently on different days, which is a silent reproducibility
        failure no guard can catch.
        """

    @abstractmethod
    def get_context(self, query: str) -> str:
        """Return the context string to hand the reader for ``query``."""
