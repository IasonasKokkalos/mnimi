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

    #: Declares that this system reads ONLY the question's annotated evidence
    #: sessions (``answer_session_ids``) rather than the whole haystack. Exactly
    #: one system sets it — the oracle ceiling, whose definition *is* "what the
    #: reader scores when retrieval is perfect".
    #:
    #: It is a declared capability rather than an annotation attached to every
    #: message, and that is the whole point: ``answer_session_ids`` identifies
    #: which sessions contain the answer, so putting it in the message stream
    #: would hand that signal to every system being measured. The runner reads
    #: this flag instead, and therefore never branches on a system's name.
    evidence_only: bool = False

    def retrieval_pins(self) -> dict:
        """Number-determining configuration this system contributes to `pins.json`.

        Empty for systems that do not retrieve. A retrieving system returns the
        subset of ``build_pins``' retrieval keys it actually uses —
        ``embedder_name``, ``embedder_dim``, ``k``, ``dedup_cosine_threshold``.

        This exists because leaving them unset was not a cosmetic gap: two mnimi
        runs whose only difference was the dedup threshold produced identical
        ``pins_hash`` while differing by 19/20 predictions and 20 accuracy
        points. A configuration knob that moves the score and not the hash makes
        the hash a false claim.
        """
        return {}

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
