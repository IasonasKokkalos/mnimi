"""``mnimi-mcp``: the five locked methods of ``mnimi.Memory`` as MCP tools over stdio.

Tools ``remember``, ``recall``, ``context``, ``export``, ``consolidate`` and the resource
``memory://export`` (the same Markdown the ``mnimi export`` CLI writes). Everything an
application decides stays here and out of the library:

* **The clock.** Every message ``remember`` stores is stamped ``ts`` = the server's calendar
  day at the call (``YYYY-MM-DDT00:00:00``; a day is a session, the benchmark's shape). A
  ``ts`` the model supplies is ignored. The library reads only the ``ts`` it is handed.
* **One ``Memory`` per process**, built lazily on the first tool call from the environment
  (``MNIMI_DB`` required; ``MNIMI_USER_ID`` default ``me``; ``MNIMI_EXTRACTOR`` ``cpu|gpu|none``
  default ``cpu``; ``MNIMI_EMBEDDER`` ``bge|hashing`` default ``bge``), so listing the tools
  loads no model. One lock around every call: the store is one connection.
* **stdout belongs to the stdio transport.** All logging goes to stderr, including the
  library's ``mnimi.memory`` INFO lines (``superseded ...``, ``routed to conflict ...``,
  ``decayed ...``), which are the point of watching it work.

The embedder and extractor construction mirrors ``examples/chat.py``'s ``build_memory`` (the
example is not an importable package, so the twenty lines are copied here with this note; the
CPU recipe is the same dict, ``{**DECODE, "n_gpu_layers": 0}``, MERGED-PLAN T5).
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import sys
import threading
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from mnimi import Memory, MemoryConfig
from mnimi.decay import now_logical
from mnimi.extract.llama import DECODE  # the pinned decode; importing it loads no model
from mnimi_cli.main import export_markdown

log = logging.getLogger("mnimi_mcp")

EXTRACTORS = ("cpu", "gpu", "none")
EMBEDDERS = ("bge", "hashing")

#: The CPU extractor recipe (MERGED-PLAN T5): the paper's decode pins with GPU offload off.
CPU_DECODE = {**DECODE, "n_gpu_layers": 0}

mcp = MCPServer(
    "mnimi",
    instructions=(
        "Long-term memory for this user in one local SQLite file. Call `remember` with the "
        "turns worth keeping (both roles); `context` before answering a question that may "
        "depend on earlier sessions; `recall` to inspect what would be retrieved; `export` to "
        "read everything as Markdown; `consolidate` to run the conflict and decay passes."
    ),
    version="0.1.0",
    log_level="WARNING",
)


# ---------------------------------------------------------------------- settings and clock


@dataclasses.dataclass(frozen=True)
class Settings:
    db: str
    user_id: str
    extractor: str
    embedder: str


def settings() -> Settings:
    """Read the four environment variables; a missing DB or an unknown value exits."""
    db = os.environ.get("MNIMI_DB", "").strip()
    if not db:
        sys.exit("mnimi-mcp: MNIMI_DB is required (the path of the SQLite store)")
    extractor = os.environ.get("MNIMI_EXTRACTOR", "cpu").strip().lower()
    if extractor not in EXTRACTORS:
        sys.exit(f"mnimi-mcp: MNIMI_EXTRACTOR must be one of {EXTRACTORS}, got {extractor!r}")
    embedder = os.environ.get("MNIMI_EMBEDDER", "bge").strip().lower()
    if embedder not in EMBEDDERS:
        sys.exit(f"mnimi-mcp: MNIMI_EMBEDDER must be one of {EMBEDDERS}, got {embedder!r}")
    return Settings(
        db, os.environ.get("MNIMI_USER_ID", "me").strip() or "me", extractor, embedder
    )


def today() -> datetime.date:
    """The application's clock. Tests replace this; the library never sees it."""
    return datetime.date.today()


def session_ts() -> str:
    """One timestamp per calendar day: a day is a session."""
    return today().isoformat() + "T00:00:00"


# ------------------------------------------------------------- the one Memory, built lazily


def _build_embedder(name: str):
    if name == "hashing":
        from mnimi.embeddings import HashingEmbedder

        return HashingEmbedder()
    from mnimi.embeddings import BgeSmallEmbedder

    return BgeSmallEmbedder()


def _build_extractor(name: str, db: str):
    """``none`` (rounds only), ``cpu`` (the recipe), ``gpu`` (the paper's pin)."""
    if name == "none":
        return None
    from mnimi.extract.cache import CachedExtractor
    from mnimi.extract.llama import QwenLlamaExtractor

    decode = dict(DECODE) if name == "gpu" else dict(CPU_DECODE)
    cache = os.path.join(os.path.dirname(os.path.abspath(db)), f"extract-cache-{name}.sqlite")
    return CachedExtractor(QwenLlamaExtractor(decode=decode), cache)


def build_memory(s: Settings) -> Memory:
    os.makedirs(os.path.dirname(os.path.abspath(s.db)) or ".", exist_ok=True)
    return Memory(
        s.db, _build_embedder(s.embedder), MemoryConfig(),
        extractor=_build_extractor(s.extractor, s.db),
    )


_lock = threading.RLock()
_memory: Memory | None = None
_settings: Settings | None = None


def memory() -> Memory:
    """The process's one ``Memory``; built on the first call, under the lock."""
    global _memory, _settings
    with _lock:
        if _memory is None:
            _settings = settings()
            log.info(
                "opening %s for user %r (embedder %s, extractor %s)",
                _settings.db, _settings.user_id, _settings.embedder, _settings.extractor,
            )
            _memory = build_memory(_settings)
        return _memory


def user_id() -> str:
    memory()
    assert _settings is not None
    return _settings.user_id


def reset() -> None:
    """Drop the lazily built ``Memory`` (tests; a new process otherwise)."""
    global _memory, _settings
    with _lock:
        if _memory is not None:
            _memory.store.close()
        _memory, _settings = None, None


# --------------------------------------------------------------------------------- tools


class Message(BaseModel):
    """One chat turn. A ``ts`` the caller adds is dropped: the server's day is the clock."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str


def _hit_text(record) -> str:
    """What a hit reads as: a fact record's fact, else the round's labelled turns."""
    if record.kind == "fact" and record.fact:
        return record.fact
    turns = record.turns or [{"role": record.source or "", "content": record.content}]
    return "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)


@mcp.tool()
def remember(
    messages: Annotated[
        list[Message],
        Field(
            description=(
                "The turns to store, in order, role user or assistant. They are stamped with "
                "today's date by the server; do not try to date them."
            ),
        ),
    ],
) -> dict[str, Any]:
    """Store chat turns in long-term memory. Returns the records stored by kind and the
    session timestamp they were stamped with."""
    if not messages:
        raise ValueError("remember needs at least one message")
    ts = session_ts()
    with _lock:
        mem, uid = memory(), user_id()
        before = {k: mem.store.count(uid, kind=k) for k in ("round", "fact")}
        mem.add([{"role": m.role, "content": m.content, "ts": ts} for m in messages], uid)
        after = {k: mem.store.count(uid, kind=k) for k in ("round", "fact")}
    return {
        "round": after["round"] - before["round"],
        "fact": after["fact"] - before["fact"],
        "ts": ts,
    }


@mcp.tool()
def recall(
    query: str,
    k: Annotated[int, Field(ge=1, le=100, description="how many rounds to return")] = 10,
) -> list[dict[str, Any]]:
    """Retrieve the k memories most relevant to a query, best first, with their scores."""
    with _lock:
        mem, uid = memory(), user_id()
        config = mem.config
        mem.config = dataclasses.replace(config, top_k=k)
        try:
            hits = mem.recall(query, uid)
        finally:
            mem.config = config
    return [
        {
            "score": hit.score,
            "salience": hit.salience,
            "kind": hit.record.kind,
            "text": _hit_text(hit.record),
            "session_date": hit.record.created_at,
        }
        for hit in hits
    ]


@mcp.tool()
def context(query: str) -> str:
    """The retrieved memories for a query, rendered oldest-first as the context block the
    library hands a reader (session-date headers, user:/assistant: lines, facts)."""
    with _lock:
        return memory().get_context(query, user_id())


def _export_markdown() -> str:
    with _lock:
        mem, uid = memory(), user_id()
        records = mem.store.all_records(uid)
        return export_markdown(uid, records, now_logical(mem.store.created_ats(uid)))


@mcp.tool()
def export() -> str:
    """Every memory as human-readable Markdown: one heading per session, each round quoted,
    its facts as bullets, a superseded fact struck through. The same text as `mnimi export`."""
    return _export_markdown()


@mcp.resource(
    "memory://export",
    mime_type="text/markdown",
    description="The whole memory as Markdown (the `export` tool's output).",
)
def export_resource() -> str:
    return _export_markdown()


@mcp.tool()
def consolidate() -> dict[str, Any]:
    """Run the conflict pass (supersede contradicted facts) and the decay pass over the
    user's memory. Idempotent. Returns both passes' counters and the logical now."""
    with _lock:
        mem, uid = memory(), user_id()
        mem.consolidate(uid)
        return {
            "conflict": dict(mem.conflict_stats),
            "decay": dict(mem.decay_stats),
            "now_logical": now_logical(mem.store.created_ats(uid)),
        }


# ---------------------------------------------------------------------------- entry point


def configure_logging() -> None:
    """Everything to stderr: the stdio transport owns stdout."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("mnimi.memory").setLevel(logging.INFO)
    logging.getLogger("mnimi_mcp").setLevel(logging.INFO)


def main() -> None:
    configure_logging()
    settings()  # fail fast on a missing MNIMI_DB, before the transport starts
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
