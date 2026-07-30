"""mnimi itself, under test."""

from __future__ import annotations

from pathlib import Path

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import Embedder

from ..base import MemorySystem
from ._scratch import ScratchDb

#: All eval stores hold exactly one user; the id only has to be stable.
EVAL_USER_ID = "eval"


class MnimiSystem(MemorySystem):
    """Thin adapter: the harness contract on one side, ``Memory`` on the other.

    Deliberately thin. Every decision that could move the number — ingestion
    granularity, the dedup screens, ``top_k``, the embedded string — lives in
    the library, where it is spec'd, tested and versioned. An adapter that
    reshaped messages or re-ranked results would be scoring the adapter.

    ``consolidate()`` is **not** called. It is a no-op stub at v1 so calling it
    could not change this number, but wiring it in now would mean that the day
    it grows merge/conflict/decay behaviour, the benchmark silently changes
    without an edit in ``evals/``. When it does something, adding the call is
    the explicit decision it should be.
    """

    name = "mnimi"

    def __init__(
        self, embedder: Embedder | None = None, config: MemoryConfig | None = None
    ) -> None:
        # Built once and reused across every question: constructing the real
        # embedder loads a ~130MB ONNX graph, and a full run resets 500 times.
        # The argument is a test seam (CI injects HashingEmbedder); the run path
        # goes through build_system(), which hardcodes the pinned model per SPEC.
        if embedder is None:
            from mnimi.embeddings import BgeSmallEmbedder

            embedder = BgeSmallEmbedder()
        self._embedder = embedder
        self._config = config if config is not None else MemoryConfig()
        self._scratch = ScratchDb("mnimi-eval-")
        self._memory: Memory | None = None
        self.reset()

    def retrieval_pins(self) -> dict:
        # `revision` is a static constant on the embedder class (the pinned HF
        # commit for BGE) — read, never resolved: a network lookup here could
        # pin whatever the hub currently serves instead of what ran.
        return {
            "embedder_name": self._embedder.name,
            "embedder_dim": self._embedder.dim,
            "embedder_revision": self._embedder.revision,
            "k": self._config.top_k,
            "dedup_cosine_threshold": self._config.dedup_cosine_threshold,
        }

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.store.close()  # close before the file is unlinked
        self._memory = Memory(str(self._scratch.next()), self._embedder, self._config)

    def add(self, messages: list[dict]) -> None:
        self._memory.add(messages, user_id=EVAL_USER_ID)

    def get_context(self, query: str) -> str:
        return self._memory.get_context(query, EVAL_USER_ID)

    # -- diagnostics, not part of the MemorySystem contract --------------------

    def contents(self) -> list[str]:
        """Stored record text, for granularity checks against naive_rag."""
        return self._memory.store.contents(EVAL_USER_ID)

    def db_path(self) -> Path:
        return self._scratch.path
