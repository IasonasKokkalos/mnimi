"""The bar: round-level RAG with no write-side policy at all."""

from __future__ import annotations

from pathlib import Path

from mnimi import MemoryConfig
from mnimi.embeddings import Embedder

# Imported deliberately, private and all: ingestion granularity is the variable
# that must not float between this system and mnimi, and the only way to
# guarantee that is to run the SAME CODE rather than a faithful-looking copy.
# A copy is exactly how the confound gets in — it stays faithful right up until
# one side is edited.
from mnimi.memory import _messages_to_rounds, _time_ordered
from mnimi.models import MemoryRecord
from mnimi.store import Store

from ..base import MemorySystem
from ._scratch import ScratchDb
from .mnimi import EVAL_USER_ID


class NaiveRagSystem(MemorySystem):
    """Every round embedded and stored verbatim; top-k cosine at read time.

    The bar mnimi has to clear, and it is deliberately built to be mnimi minus
    exactly one thing. Held identical: the embedder, the round construction
    (literally the same function), ``top_k``, the vec0 store, the L2/cosine
    retrieval, and the context assembly. Different: **nothing is deduplicated**
    — no exact-normalize collapse, no cosine screen. Every ingested round
    becomes a record.

    That single delta is what makes the pair interpretable. If naive_rag also
    deduped it would be byte-identical to mnimi-v1 and the comparison would be
    vacuous rather than flat; if it differed in granularity too, a score gap
    could not be attributed to anything.

    Expected at v1: **≈ equal**. LongMemEval's haystack is constructed
    non-conflicting outside the planted evidence (Appendix A.2), so there is
    little for a cosine screen to earn. mnimi coming out *worse* is the
    interesting outcome — that would mean dedup is discarding evidence.
    """

    name = "naive_rag"

    def __init__(
        self, embedder: Embedder | None = None, config: MemoryConfig | None = None
    ) -> None:
        if embedder is None:
            from mnimi.embeddings import BgeSmallEmbedder

            embedder = BgeSmallEmbedder()
        self._embedder = embedder
        self._config = config if config is not None else MemoryConfig()
        self._scratch = ScratchDb("naive-rag-eval-")
        self._store: Store | None = None
        self.reset()

    def retrieval_pins(self) -> dict:
        # No dedup_cosine_threshold: this system does not dedup, and reporting a
        # threshold it never applies would misdescribe the run.
        return {
            "embedder_name": self._embedder.name,
            "embedder_dim": self._embedder.dim,
            "k": self._config.top_k,
        }

    def reset(self) -> None:
        if self._store is not None:
            self._store.close()  # close before the file is unlinked
        self._store = Store(
            str(self._scratch.next()),
            dim=self._embedder.dim,
            embedder_name=self._embedder.name,
            embedder_revision=self._embedder.revision,
        )

    def add(self, messages: list[dict]) -> None:
        """Store every round. This is ``Memory.add`` with the screens removed."""
        rounds = _messages_to_rounds(messages)
        if not rounds:
            return
        embeddings = self._embedder.embed([content for content, _, _ in rounds])
        for (content, roles, ts), embedding in zip(rounds, embeddings, strict=True):
            self._store.insert(
                MemoryRecord(
                    user_id=EVAL_USER_ID,
                    content=content,
                    embedding=embedding,
                    created_at=ts,
                    source=roles,
                )
            )

    def get_context(self, query: str) -> str:
        """Top-k by cosine, time-ordered, joined — the assembly ``Memory`` performs.

        ``_time_ordered`` is imported rather than reimplemented for the same
        reason as ``_messages_to_rounds``: ordering is held identical across the
        pair, so it cannot become a second difference between them.
        """
        (query_embedding,) = self._embedder.embed([query])
        hits = self._store.search(query_embedding, user_id=EVAL_USER_ID, k=self._config.top_k)
        records = _time_ordered([record for record, _cosine in hits])
        return "\n".join(record.content for record in records)

    # -- diagnostics, not part of the MemorySystem contract --------------------

    def contents(self) -> list[str]:
        """Stored record text, for granularity checks against mnimi."""
        return self._store.contents(EVAL_USER_ID)

    def db_path(self) -> Path:
        return self._scratch.path
