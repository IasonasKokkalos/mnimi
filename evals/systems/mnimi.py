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
    granularity, the dedup screens, ``top_k``, the embedded string, the
    extractor and what it makes of a round — lives in the library, where it is
    spec'd, tested and versioned. An adapter that reshaped messages or
    re-ranked results would be scoring the adapter.

    ``consolidate()`` is called only when the arm is built with
    ``consolidate=True`` (PHASE4 D8): once per store, after every session is
    ingested and before the question is ranked (``consolidate_if_wired``). Its
    conflict pass is a no-op after ``add()``; its decay pass is what reaches
    the ranking under ``ranking="score"``. The flag is a pin, so the day the
    wiring moves the number is an explicit edit in ``evals/``, never a silent one.

    ``extractor`` (PHASE2 D6) is the one LLM, injected exactly as the embedder
    is; ``None`` is the v1 arm (rounds only). It is wrapped in the on-disk
    ``CachedExtractor`` (D10) so a corpus is extracted once per configuration.
    """

    name = "mnimi"

    def __init__(
        self,
        embedder: Embedder | None = None,
        config: MemoryConfig | None = None,
        extractor=None,
        extraction_cache: str | Path | None = None,
        consolidate: bool = False,
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
        self._extractor = None
        if extractor is not None:
            from mnimi.extract.cache import CachedExtractor, default_cache_path

            path = Path(extraction_cache) if extraction_cache else default_cache_path(
                extractor.pins
            )
            self._extractor = CachedExtractor(extractor, path)
        self._consolidate = bool(consolidate)
        self._pending = False
        self._scratch = ScratchDb("mnimi-eval-")
        self._memory: Memory | None = None
        self.reset()

    def retrieval_pins(self) -> dict:
        # `revision` is a static constant on the embedder class (the pinned HF
        # commit for BGE) — read, never resolved: a network lookup here could
        # pin whatever the hub currently serves instead of what ran.
        from mnimi.conflict.lexicon import negation_lexicon_hash
        from mnimi.conflict.normalize import conflict_rules_hash
        from mnimi.decay import decay_rules_hash
        from mnimi.extract import prefilter
        from mnimi.extract.protocol import PIN_KEYS
        from mnimi.extract.resolver import RESOLVER_VERSION
        from mnimi.memory import (
            embed_template_hash,
            fact_embed_template_hash,
            render_unit_template_hash,
        )
        from mnimi.store import GUARD_NONE

        pins = {
            "embedder_name": self._embedder.name,
            "embedder_dim": self._embedder.dim,
            "embedder_revision": self._embedder.revision,
            "embed_template_hash": embed_template_hash(),
            "k": self._config.top_k,
            "dedup_cosine_threshold": self._config.dedup_cosine_threshold,
            "dedup_scope": self._config.dedup_scope,
            "query_instruction": self._config.query_instruction,
            "chunk_tokens": self._config.chunk_tokens,
            "chunk_overlap": self._config.chunk_overlap,
        }
        # The extraction era (schema /8): what made the fact records — or
        # GUARD_NONE, the same value the store's guard rows carry for a v1 arm,
        # so "this arm does not extract" is declared, not implied by absence.
        if self._extractor is not None:
            pins.update({key: self._extractor.pins[key] for key in PIN_KEYS})
        else:
            pins.update({key: GUARD_NONE for key in PIN_KEYS})
        pins.update(
            {
                "prefilter_lexicon_hash": prefilter.prefilter_lexicon_hash(),
                "fact_embed_template_hash": fact_embed_template_hash(),
                "resolver_version": RESOLVER_VERSION,
                "render_unit": self._config.render_unit,
                "render_unit_template_hash": render_unit_template_hash(self._config.render_unit),
                # Phase 3 (schema /9): the screens and supersession, and the two
                # frozen artifacts they read. mnimi only — naive_rag has no facts.
                "dedup_entropy_gate": self._config.dedup_entropy_gate,
                "conflict_resolution": self._config.conflict_resolution,
                "negation_lexicon_hash": negation_lexicon_hash(),
                "conflict_rules_hash": conflict_rules_hash(),
                # Phase 4 (schema /10): the read side, decay and the wiring.
                "active_only": self._config.active_only,
                "ranking": self._config.ranking,
                "salience_weights": dict(self._config.salience_weights),
                "recall_min_relevance": self._config.recall_min_relevance,
                "decay_half_life_days": self._config.decay_half_life_days,
                "decay_floor": self._config.decay_floor,
                "consolidate": self._consolidate,
                "decay_rules_hash": decay_rules_hash(),
            }
        )
        return pins

    def reset(self) -> None:
        if self._memory is not None:
            self._memory.store.close()  # close before the file is unlinked
        self._memory = Memory(
            str(self._scratch.next()), self._embedder, self._config, extractor=self._extractor
        )
        self._pending = False

    def add(self, messages: list[dict]) -> None:
        self._memory.add(messages, user_id=EVAL_USER_ID)
        self._pending = True

    def get_context(self, query: str) -> str:
        self.consolidate_if_wired()
        return self._memory.get_context(query, EVAL_USER_ID)

    def consolidate_if_wired(self) -> None:
        """One ``Memory.consolidate`` per store, before the question (PHASE4 D8).

        Only when the arm was built with ``consolidate=True`` and something was
        added since the last call. ``get_context`` calls it; so does the
        retrieval probe, before it ranks, so the probe sees the arm's store.
        """
        if self._consolidate and self._pending:
            self._memory.consolidate(EVAL_USER_ID)
        self._pending = False

    # -- diagnostics, not part of the MemorySystem contract --------------------

    def contents(self) -> list[str]:
        """Stored record text, for granularity checks against naive_rag."""
        return self._memory.store.contents(EVAL_USER_ID)

    def db_path(self) -> Path:
        return self._scratch.path

    @property
    def extraction_stats(self) -> dict:
        """The current store's extraction counters (``Memory.extraction_stats``)."""
        return self._memory.extraction_stats

    @property
    def cache_stats(self) -> dict | None:
        """Hits and misses of the extraction cache, or ``None`` without an extractor."""
        return self._extractor.stats if self._extractor is not None else None
