"""``mnimi-mcp``: the five locked methods of ``mnimi.Memory`` as MCP tools over stdio.

Tools ``remember``, ``recall``, ``context``, ``export``, ``consolidate`` and the resource
``memory://export`` (the same Markdown the ``mnimi export`` CLI writes). Everything an
application decides stays here and out of the library:

* **The clock.** Every message ``remember`` stores is stamped ``ts`` = the server's calendar
  day at the call (``YYYY-MM-DDT00:00:00``; a day is a session, the benchmark's shape). A
  ``ts`` the model supplies is ignored. The same day is handed to ``recall`` and ``context`` as
  the documented query prefix ``[Current date: <ts>]``, which the library parses for the
  question's own time window and strips before embedding: that is what makes the shipped
  configuration's time-aware term fire (``mnimi.temporal``). The library reads only the ``ts``
  it is handed.
* **One ``Memory`` per process**, built lazily on the first tool call from the environment
  (``MNIMI_DB`` required; ``MNIMI_USER_ID`` default ``me``; ``MNIMI_EXTRACTOR`` ``cpu|gpu|none``
  default ``cpu``; ``MNIMI_EMBEDDER`` ``bge|hashing`` default ``bge``), so listing the tools
  loads no model. One lock around every call: the store is one connection.
* **stdout belongs to the stdio transport.** All logging goes to stderr, including the
  library's ``mnimi.memory`` INFO lines (``superseded ...``, ``routed to conflict ...``,
  ``decayed ...``), which are the point of watching it work.
* **Failures reach the client as text.** A store built under other pins, a model that cannot
  load, a bad setting: the message is raised as the SDK's ``ToolError`` / ``ResourceError`` so
  the model (and the person behind it) reads it, instead of a bare "Error executing tool".

The embedder and extractor construction mirrors ``examples/chat.py``'s ``build_memory`` (the
example is not an importable package, so the twenty lines are copied here with this note; the
CPU profile is the same dict, ``{**DECODE, "n_gpu_layers": 0, "max_tokens": 2048}``,
MERGED-PLAN T5 and TESTFIXESPT1).
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
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from pydantic import BaseModel, ConfigDict, Field

from mnimi import Memory, MemoryConfig
from mnimi.decay import now_logical
from mnimi.extract.llama import DECODE  # the pinned decode; importing it loads no model
from mnimi.extract.protocol import ExtractedFact
from mnimi.extract.wrappers import (
    DEFAULT_SPLIT_CHARS,
    BackslashEscapingExtractor,
    ClientFactsExtractor,
    SplittingExtractor,
)
from mnimi.temporal import dated_query
from mnimi_cli.main import export_markdown

from . import __version__

log = logging.getLogger("mnimi_mcp")

EXTRACTORS = ("cpu", "gpu", "none")
EMBEDDERS = ("bge", "hashing")

#: The CPU extractor recipe (MERGED-PLAN T5): the paper's decode pins with GPU offload off.
# max_tokens doubled 2026-10-14 (TESTFIXESPT1 Task 6): a 600-char sub-round of numeric-dense text
# still filled the paper's 1,024-token budget (2,083 chars of JSON, cut inside a slot). The CPU
# profile has its own decode hash and its own cache; the paper's DECODE never moves.
CPU_DECODE = {**DECODE, "n_gpu_layers": 0, "max_tokens": 2048}

mcp = MCPServer(
    "mnimi",
    instructions=(
        "Long-term memory for this user in one local SQLite file. Call `remember` with the "
        "turns worth keeping (both roles); `context` before answering a question that may "
        "depend on earlier sessions; `recall` to inspect what would be retrieved (it also "
        "marks those memories as accessed); `export` to read everything as Markdown; "
        "`consolidate` to run the conflict and decay passes."
    ),
    version=__version__,
    log_level="WARNING",
)


# ---------------------------------------------------------------------- settings and clock


@dataclasses.dataclass(frozen=True)
class Settings:
    db: str
    user_id: str
    extractor: str
    embedder: str
    split_chars: int


def settings() -> Settings:
    """Read the five environment variables.

    Raises ``ValueError`` on a missing DB or an unknown value: ``main()`` turns that into an
    exit before the transport starts, ``memory()`` into a ``ToolError`` the client can read.
    ``MNIMI_DB`` gets ``~`` and ``$VAR`` expanded and is made absolute, because a shell other
    than bash hands ``-e MNIMI_DB=~/x.db`` through literally and the host's working directory is
    not the user's.
    """
    raw = os.environ.get("MNIMI_DB", "").strip()
    if not raw:
        raise ValueError("mnimi-mcp: MNIMI_DB is required (the path of the SQLite store)")
    db = os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
    extractor = os.environ.get("MNIMI_EXTRACTOR", "cpu").strip().lower()
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"mnimi-mcp: MNIMI_EXTRACTOR must be one of {EXTRACTORS}, got {extractor!r}"
        )
    embedder = os.environ.get("MNIMI_EMBEDDER", "bge").strip().lower()
    if embedder not in EMBEDDERS:
        raise ValueError(f"mnimi-mcp: MNIMI_EMBEDDER must be one of {EMBEDDERS}, got {embedder!r}")
    raw_split = os.environ.get("MNIMI_SPLIT_CHARS", str(DEFAULT_SPLIT_CHARS)).strip()
    try:
        split_chars = int(raw_split)
        if split_chars < 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            "mnimi-mcp: MNIMI_SPLIT_CHARS must be a non-negative integer number of characters "
            "(0 = off), "
            f"got {raw_split!r}"
        ) from None
    return Settings(
        db, os.environ.get("MNIMI_USER_ID", "me").strip() or "me", extractor, embedder,
        split_chars,
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


def _build_extractor(name: str, db: str, split_chars: int):
    """``none`` (client facts only), ``cpu`` (the recipe), ``gpu`` (the paper's pin) — each
    behind the client-facts wrapper, the model ones behind the split and the pre-escape too
    (TESTFIXESPT1 Task 5)."""
    if name == "none":
        return ClientFactsExtractor(None)
    from mnimi.extract.cache import CachedExtractor
    from mnimi.extract.llama import QwenLlamaExtractor

    decode = dict(DECODE) if name == "gpu" else dict(CPU_DECODE)
    cache = os.path.join(os.path.dirname(db), f"extract-cache-{name}.sqlite")
    model = BackslashEscapingExtractor(CachedExtractor(QwenLlamaExtractor(decode=decode), cache))
    return ClientFactsExtractor(SplittingExtractor(model, max_chars=split_chars))


def build_memory(s: Settings) -> Memory:
    os.makedirs(os.path.dirname(s.db), exist_ok=True)
    return Memory(
        s.db, _build_embedder(s.embedder), MemoryConfig(),
        extractor=_build_extractor(s.extractor, s.db, s.split_chars),
    )


_lock = threading.RLock()
_memory: Memory | None = None
_settings: Settings | None = None


def memory() -> Memory:
    """The process's one ``Memory``; built on the first call, under the lock.

    A failed build is reported as a ``ToolError`` carrying the cause's text (the
    ``MemoryMetaError`` of a store built under other pins, the ``RuntimeError`` of a model
    that cannot load, the ``ValueError`` of a bad setting) and leaves nothing half-open.
    """
    global _memory, _settings
    with _lock:
        if _memory is None:
            try:
                s = settings()
                log.info(
                    "opening %s for user %r (embedder %s, extractor %s)",
                    s.db, s.user_id, s.embedder, s.extractor,
                )
                _memory = build_memory(s)
                _settings = s
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                log.error("cannot open the memory store: %s", exc)
                raise ToolError(f"mnimi-mcp cannot open the memory store: {exc}") from exc
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


class ClientFact(BaseModel):
    """One fact the caller extracted itself. ``raw`` and ``when`` must be text of the round:
    a missing ``raw`` is filled with the turn sentence closest to ``content``; a ``when`` the
    round does not contain is dropped (the fact is kept undated)."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description=(
        "one self-contained sentence; begin with 'The user' or 'The assistant'; keep names, "
        "numbers and dates exactly as written in the turn"))
    raw: str | None = Field(None, description=(
        "a contiguous verbatim excerpt of the turn, prefixed 'user: ' or 'assistant: '; "
        "omit it and the server picks the turn sentence closest to content"))
    when: str | None = Field(None, description=(
        "the time expression exactly as the turn writes it ('yesterday', 'on May 14th', "
        "'2026-10-14'); resolved against the session day; omit when the turn states none"))
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    salience: Literal[0.25, 0.5, 1.0] = Field(1.0, description=(
        "1.0 durable facts about the user; 0.5 what the assistant decided for them; 0.25 the rest"))


def _hit_text(record) -> str:
    """What a hit reads as: a fact record's fact, else the round's labelled turns."""
    if record.kind == "fact" and record.fact:
        return record.fact
    turns = record.turns or [{"role": record.source or "", "content": record.content}]
    return "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)


def _delta(after: dict, before: dict) -> dict:
    return {key: after[key] - before.get(key, 0) for key in after}


@mcp.tool()
def remember(
    messages: Annotated[
        list[Message],
        Field(
            min_length=1,
            description=(
                "The turns to store, in order, role user or assistant. They are stamped with "
                "today's date by the server; do not try to date them."
            ),
        ),
    ],
    facts: Annotated[
        list[ClientFact] | None,
        Field(
            description=(
                "Optional: the facts you extracted from these turns yourself. When given, the "
                "local model is not run for this round and these are stored instead — use it "
                "for dense turns (results, decisions, many numbers) the small model would skip "
                "or cut. Needs exactly one round: one user turn, optionally followed by one "
                "assistant turn."
            ),
        ),
    ] = None,
) -> dict[str, Any]:
    """Store chat turns in long-term memory. Returns the records this call added by kind
    (`round`, `fact`; a duplicate round adds 0), how many earlier facts it superseded, the
    session timestamp the turns were stamped with, and how many facts you supplied."""
    ts = session_ts()
    if facts:
        if any(not f.content.strip() for f in facts):
            raise ToolError("a fact's content must not be blank")
        roles = [m.role for m in messages]
        if roles not in (["user"], ["user", "assistant"]):
            raise ToolError(
                "facts need exactly one round: one user turn, optionally followed by one "
                f"assistant turn; got roles {roles}"
            )
    unused = 0
    with _lock:
        mem, uid = memory(), user_id()
        extractor = mem.extractor
        if facts:
            extractor.provide([
                ExtractedFact(f.content.strip(), (f.raw or "").strip(), f.when, f.subject,
                              f.predicate, f.object, float(f.salience))
                for f in facts
            ])
        before = {k: mem.store.count(uid, kind=k) for k in ("round", "fact")}
        superseded_before = mem.conflict_stats["superseded"]
        try:
            mem.add([{"role": m.role, "content": m.content, "ts": ts} for m in messages], uid)
        finally:
            if facts:
                unused = len(extractor.pending or [])
                extractor.pending = None
        after = {k: mem.store.count(uid, kind=k) for k in ("round", "fact")}
        superseded = mem.conflict_stats["superseded"] - superseded_before
    out = {
        "round": after["round"] - before["round"],
        "fact": after["fact"] - before["fact"],
        "superseded": superseded,
        "ts": ts,
        "facts_supplied": len(facts or []),
    }
    if unused:
        out["facts_unused"] = unused  # the pre-filter skipped the round (a greeting, a thanks)
    return out


@mcp.tool()
def recall(
    query: str,
    k: Annotated[int, Field(ge=1, le=100, description="how many rounds to return")] = 10,
) -> list[dict[str, Any]]:
    """Retrieve the k memories most relevant to a query, best first, with their scores. Not
    a pure read: the returned memories count as accessed today, which protects them from
    the next decay pass."""
    dated = dated_query(session_ts(), query)
    with _lock:
        mem, uid = memory(), user_id()
        config = mem.config
        mem.config = dataclasses.replace(config, top_k=k)
        try:
            hits = mem.recall(dated, uid)
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
    dated = dated_query(session_ts(), query)
    with _lock:
        return memory().get_context(dated, user_id())


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
    try:
        return _export_markdown()
    except ToolError as exc:
        raise ResourceError(str(exc)) from exc


@mcp.tool()
def consolidate() -> dict[str, Any]:
    """Run the conflict pass (supersede contradicted facts) and the decay pass over the
    user's memory. Idempotent. Returns what THIS call changed: the conflict pass's counters,
    the decay pass's (`passes`, `decayed`, `at_floor`), and the logical now."""
    with _lock:
        mem, uid = memory(), user_id()
        conflict_before, decay_before = dict(mem.conflict_stats), dict(mem.decay_stats)
        mem.consolidate(uid)
        return {
            "conflict": _delta(mem.conflict_stats, conflict_before),
            "decay": _delta(mem.decay_stats, decay_before),
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
    try:
        settings()  # fail fast on a missing MNIMI_DB, before the transport starts
    except ValueError as exc:
        sys.exit(str(exc))
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
