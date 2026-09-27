"""agentmemory v4 through the harness (PHASE8 Task 13).

The system is JordanMcCann/agentmemory, installed from GitHub at a pinned commit,
not the unrelated 2023 package of the same name on PyPI. The adapter runs it as
shipped. The configuration comes from ``configs/agentmemory_competitor.json``:
- the store settings of the author's own ``run_longmemeval_full.py`` at that commit,
  every setting that branches on question type excluded;
- the dense embedder and the cross-encoder, each loaded from a revision-pinned
  Hugging Face snapshot;
- the author's ``PYTHONHASHSEED``.

Per question the adapter does four things:
- **reset:** a fresh ``MemoryStore(":memory:")`` with the pinned embedder, the pinned
  reranker injected the way the author's script injects it
  (``store._retrieval._reranker``);
- **add:** one ``ingest_conversation`` per session, with a unique ``session_id``, the
  session's own date as ``reference_date``, and on every message as ``timestamp``.
  The session date never becomes a pseudo-message;
- **context:** ``recall(question, limit=candidates)``, the library's own call, then the
  longest recall-order prefix that fits the context budget, rendered through the one
  renderer (``competitor.CompetitorSystem.context_from``);
- **ids:** the nodes' own ids, in recall order.

Disclosed, not patched:
- the library reads wall-clock time (``created_at`` / ``accessed_at``) in its
  recency and activation signals. All of a store's nodes are created within seconds
  of each other, so those signals are near-constant, but a rerun on another day can
  still reorder near-ties;
- node ids are random (``uuid4``), so ``retrieved_ids`` differ between runs of one
  configuration.
"""

from __future__ import annotations

import os

from .competitor import CompetitorSystem, Hit, epoch


def _snapshot(spec: dict) -> str:
    """A local path to ``spec["repo"]`` at ``spec["revision"]``: loaded by path, never ``main``."""
    from huggingface_hub import snapshot_download

    return snapshot_download(spec["repo"], revision=spec["revision"])


class AgentMemorySystem(CompetitorSystem):
    name = "agentmemory"

    def __init__(self, competitor_config: dict, render_format: str = "text") -> None:
        super().__init__(competitor_config, render_format)
        seed = competitor_config.get("pythonhashseed")
        if seed is not None and os.environ.get("PYTHONHASHSEED") != str(seed):
            raise RuntimeError(
                f"agentmemory's configuration pins PYTHONHASHSEED={seed} (the author's "
                f"determinism setting); this process has {os.environ.get('PYTHONHASHSEED')!r}. "
                "Set it in the environment before Python starts."
            )
        import agentmemory

        self._am = agentmemory
        self._embedder = None
        if competitor_config.get("embedder"):
            from agentmemory.embeddings import DenseEmbedder

            spec = competitor_config["embedder"]
            self._embedder = DenseEmbedder(model_name=_snapshot(spec),
                                           device=spec.get("device", "cpu"))
        self._reranker = None
        if competitor_config.get("reranker"):
            from agentmemory.reranking import CrossEncoderReranker

            spec = competitor_config["reranker"]
            self._reranker = CrossEncoderReranker(model_name=_snapshot(spec))
            self._reranker._load_model()
        self._store_kwargs = dict(competitor_config.get("store") or {})
        self._store = None
        self._session_ts: dict[str, str | None] = {}

    def competitor_pins(self) -> dict:
        emb = self._config_block.get("embedder") or {}
        embedder = f"{emb['repo']}@{emb['revision']}" if emb else "library default"
        return {
            "competitor_name": self.name,
            "competitor_version": self._config_block.get("package")
            or getattr(self._am, "__version__", None),
            "competitor_embedder": embedder,
            "competitor_llm": "none",  # no LLM at write or read time
        }

    def reset(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = self._am.MemoryStore(path=":memory:", embedder=self._embedder,
                                           **self._store_kwargs)
        if self._reranker is not None:
            # The author's own injection (run_longmemeval_full.py): one pinned model,
            # not a per-store download of whatever the hub serves.
            self._store._retrieval._reranker = self._reranker
        self._session_ts = {}
        self._last_ids = []

    def add(self, messages: list[dict]) -> None:
        if not messages:
            return
        ts = messages[0].get("ts")
        session_id = f"s{len(self._session_ts):03d}"
        self._session_ts[session_id] = ts
        when = epoch(ts)
        converted = []
        for m in messages:
            item = {"role": m.get("role", ""), "content": m.get("content", "")}
            if when is not None:
                item["timestamp"] = when
            converted.append(item)
        self._store.ingest_conversation(converted, session_id=session_id, reference_date=when)

    def get_context(self, query: str) -> str:
        results = self._store.recall(query, limit=self.candidates)
        hits = []
        for r in results:
            node = r.node
            sid = getattr(getattr(node, "provenance", None), "session_id", None)
            hits.append(Hit(str(node.id), node.content, self._session_ts.get(sid)))
        return self.context_from(hits)
