"""Mem0 OSS through the harness (PHASE8 Task 14).

The system is ``mem0ai==2.2.1`` from PyPI, run as shipped. Its configuration comes
from ``configs/mem0_competitor.json``:
- the write LLM: gpt-4o-mini at temperature 0 over the API (the maintainer's R2a);
- the embedder: ``BAAI/bge-small-en-v1.5`` at mnimi's pinned revision, through
  sentence-transformers and loaded from the revision-pinned snapshot;
- a local Qdrant store and a history database, fresh per question;
- the context budget of 2026-09-27.

Per question the adapter does four things:
- **reset:** closes the previous store (Mem0's ``close`` releases only its history
  database, so the Qdrant client, which holds file locks, is closed too), deletes it,
  and opens a fresh one;
- **add:** one ``Memory.add`` per session, with the session's turns as
  ``{"role", "content"}``, the session date as ``metadata={"session_date": ts}`` and
  ``infer=True``. The open-source build refuses ``timestamp=`` ("Platform-only
  temporal parameter"), so metadata is the system's own field for the date;
- **context:** ``search(question, top_k=candidates, filters={"user_id": …})`` with the
  library's default threshold, then the longest search-order prefix that fits the
  context budget, rendered through the one renderer;
- **ids:** the memories' own ids, in search order.

Disclosed, not patched:
- Mem0 sends telemetry by default; ``MEM0_TELEMETRY=False`` is set before it is
  imported. That changes no behaviour;
- Mem0 stamps ``created_at`` from the wall clock. Its search ranking does not read it;
- the write LLM's token usage is observed by wrapping the OpenAI client's
  ``chat.completions.create`` in this process. The wrapper records ``usage`` for the
  configured model and never alters a request or a response. It feeds the manifest's
  ``cost.competitor_llm_usd``, which is outside the harness ledger (R2a).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .competitor import CompetitorSystem, Hit
from .mnimi import EVAL_USER_ID

_USAGE_SINK = None


def _snapshot(repo: str, revision: str) -> str:
    """A local path to ``repo`` at ``revision``: loaded by path, never ``main``."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


def _install_usage_recorder(sink) -> None:
    """Record the usage of every chat completion this process makes; alter nothing."""
    global _USAGE_SINK
    _USAGE_SINK = sink
    from openai.resources.chat.completions import Completions

    if getattr(Completions.create, "_mnimi_usage_recorder", False):
        return
    original = Completions.create

    def create(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None and _USAGE_SINK is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            _USAGE_SINK(kwargs.get("model"), getattr(response, "model", None),
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0,
                        getattr(details, "cached_tokens", 0) or 0)
        return response

    create._mnimi_usage_recorder = True
    Completions.create = create


class Mem0System(CompetitorSystem):
    name = "mem0"

    def __init__(self, competitor_config: dict, render_format: str = "text") -> None:
        super().__init__(competitor_config, render_format)
        # Mem0 reads the telemetry switch when it is imported.
        os.environ["MEM0_TELEMETRY"] = "False"
        from mem0 import Memory

        self._memory_cls = Memory
        self._llm = dict(competitor_config["llm"])
        self._llm_model = (self._llm.get("config") or {}).get("model")
        emb = competitor_config["embedder"]
        model = emb["repo"]
        extra = {}
        if emb["provider"] == "huggingface":
            model = _snapshot(emb["repo"], emb["revision"])
            extra = {"model_kwargs": {"device": emb.get("device", "cpu")}}
        self._embedder = {"provider": emb["provider"],
                          "config": {"model": model, "embedding_dims": emb["embedding_dims"],
                                     **extra}}
        self._vector_store = dict(competitor_config["vector_store"])
        self._version = competitor_config.get("version", "v1.1")
        self._infer = bool((competitor_config.get("add") or {}).get("infer", True))
        self._threshold = (competitor_config.get("search") or {}).get("threshold", 0.1)
        self._usage: dict[str, dict] = {}
        if self._llm.get("provider") == "openai":
            _install_usage_recorder(self._observe)
        self._scratch = tempfile.TemporaryDirectory(prefix="mem0-eval-",
                                                    ignore_cleanup_errors=True)
        self._n = 0
        self._root: Path | None = None
        self._memory = None

    def competitor_pins(self) -> dict:
        emb = self._config_block["embedder"]
        return {
            "competitor_name": self.name,
            "competitor_version": self._config_block.get("package"),
            "competitor_embedder": f"{emb['repo']}@{emb['revision']}",
            "competitor_llm": f"{self._llm.get('provider')}:{self._llm_model}",
        }

    # -- the write LLM's usage (observed, never altered) ---------------------------------------

    def _observe(self, requested: str | None, served: str | None, prompt: int,
                 completion: int, cached: int = 0) -> None:
        if requested == self._llm_model:
            self._record_usage(served or requested, prompt, completion, cached=cached)

    def _record_usage(self, model: str, prompt: int, completion: int, cached: int = 0) -> None:
        row = self._usage.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0,
                                             "cached": 0})
        row["calls"] += 1
        row["prompt"] += int(prompt)
        row["completion"] += int(completion)
        row["cached"] += int(cached)

    def take_llm_usage(self) -> dict:
        """The write LLM's usage since the last take, per served model; then cleared."""
        taken, self._usage = self._usage, {}
        return taken

    # -- the harness contract ----------------------------------------------------------------

    def _close(self) -> None:
        if self._memory is None:
            return
        client = getattr(getattr(self._memory, "vector_store", None), "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()
        self._memory.close()
        self._memory = None
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)

    def reset(self) -> None:
        self._close()
        self._n += 1
        self._root = Path(self._scratch.name) / f"q{self._n}"
        self._root.mkdir(parents=True, exist_ok=True)
        config = {
            "vector_store": {"provider": self._vector_store["provider"],
                             "config": {**(self._vector_store.get("config") or {}),
                                        "path": str(self._root / "qdrant")}},
            "llm": self._llm,
            "embedder": self._embedder,
            "history_db_path": str(self._root / "history.db"),
            "version": self._version,
        }
        self._memory = self._memory_cls.from_config(config)
        self._last_ids = []

    def add(self, messages: list[dict]) -> None:
        if not messages:
            return
        ts = messages[0].get("ts")
        converted = [{"role": m.get("role", ""), "content": m.get("content", "")}
                     for m in messages]
        self._memory.add(converted, user_id=EVAL_USER_ID, metadata={"session_date": ts},
                         infer=self._infer)

    def get_context(self, query: str) -> str:
        found = self._memory.search(query, top_k=self.candidates,
                                    filters={"user_id": EVAL_USER_ID}, threshold=self._threshold)
        rows = found.get("results", []) if isinstance(found, dict) else list(found)
        hits = [Hit(str(r["id"]), r.get("memory", ""),
                    (r.get("metadata") or {}).get("session_date")) for r in rows]
        return self.context_from(hits)
