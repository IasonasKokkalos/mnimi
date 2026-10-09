"""``mnimi export <db> <user_id> [-o FILE] [--format md|text]`` (MERGED-PLAN T2, LAUNCH §5).

A read over a store, with no model loaded. ``Memory`` validates a store's identity at
open through ``memory_meta`` (the embedder's name, revision and dimension, the chunking
pair, the five extractor pins, the frozen hashes), so the CLI reads those rows first and
hands ``Memory`` two stubs that carry them: an embedder whose ``embed`` refuses to run,
and an extractor whose ``extract`` refuses to run. ``export()`` never calls either.

Two formats:

- ``text``: ``Memory.export()``'s bytes, unchanged.
- ``md``: a transform of the records, not of the text. One ``## <session date>`` heading
  per timestamp change (oldest first, as ``export()`` orders them), each round's turns as
  a ``>`` quoted block whose lines are the one renderer's bytes, the round's facts as a
  bullet list, a superseded fact struck through with the winner's id, the winner marked
  ``→ supersedes #<loser id>`` (``export.py``'s direction: ``supersedes`` is set on the
  record that won), ``valid_time`` in bold.

No clock: the header's ``now`` is ``now_logical``, the latest dated session timestamp in
the user's own store, exactly as ``export()`` prints it.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Sequence

from mnimi import Memory, MemoryConfig
from mnimi.decay import now_logical
from mnimi.export import HEADER, NO_NOW
from mnimi.extract.protocol import PIN_KEYS
from mnimi.memory import render_records
from mnimi.models import KIND_FACT, KIND_ROUND, MemoryRecord
from mnimi.store import GUARD_NONE, MemoryMetaError

FORMATS = ("md", "text")

#: The ``memory_meta`` rows the stubs are built from. Everything else the guard compares
#: comes from the installed library (its frozen hashes), which is the point: a store from
#: a different era is refused, not misread.
_EMBEDDER_ROWS = ("embedder_name", "embedder_revision", "embedder_dim")
_CHUNK_ROWS = ("chunk_tokens", "chunk_overlap")


class MetaEmbedder:
    """An embedder that carries a store's pins and refuses to embed."""

    def __init__(self, name: str, revision: str, dim: int) -> None:
        self.name = name
        self.revision = revision
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("export never embeds")


class MetaExtractor:
    """An extractor that carries a store's five pins and refuses to extract."""

    def __init__(self, pins: dict) -> None:
        self._pins = dict(pins)

    @property
    def pins(self) -> dict:
        return dict(self._pins)

    def extract(self, turns: list[dict]):
        raise RuntimeError("export never extracts")


def read_meta(db_path: str) -> dict[str, str]:
    """The ``memory_meta`` rows of an existing store, read with plain sqlite3."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "memory_meta" not in tables:
            raise MemoryMetaError(
                f"{db_path} has no memory_meta table: not a mnimi store, or one that "
                "predates the reproducibility guard"
            )
        return dict(db.execute("SELECT key, value FROM memory_meta").fetchall())
    finally:
        db.close()


def open_store(db_path: str) -> Memory:
    """Open an existing store with no model, under its own pins."""
    meta = read_meta(db_path)
    missing = [key for key in (*_EMBEDDER_ROWS, *_CHUNK_ROWS, *PIN_KEYS) if key not in meta]
    if missing:
        raise MemoryMetaError(f"{db_path}: memory_meta lacks {missing}")
    embedder = MetaEmbedder(
        meta["embedder_name"], meta["embedder_revision"], int(meta["embedder_dim"])
    )
    pins = {key: meta[key] for key in PIN_KEYS}
    extractor = None if all(v == GUARD_NONE for v in pins.values()) else MetaExtractor(pins)
    config = MemoryConfig(
        chunk_tokens=int(meta["chunk_tokens"]), chunk_overlap=int(meta["chunk_overlap"])
    )
    return Memory(db_path, embedder, config, extractor=extractor)


def _ordered(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """``export.py``'s order: oldest first, ties by insertion order."""
    return sorted(records, key=lambda r: (r.created_at or "", r.id or 0))


def _quoted(block: str) -> str:
    return "\n".join("> " + line for line in block.split("\n"))


def _fact_bullet(record: MemoryRecord, superseded_by: dict[int, int]) -> str:
    text = record.fact or record.content
    if record.valid_time:
        text = f"{text} (**{record.valid_time}**)"
    if record.salience == 0:
        winner = superseded_by.get(record.id)
        mark = f" (superseded by #{winner})" if winner is not None else " (superseded)"
        return f"- ~~{text}~~{mark}"
    if record.supersedes is not None:
        text = f"{text} → supersedes #{record.supersedes}"
    return f"- {text}"


def export_markdown(user_id: str, records: list[MemoryRecord], now: str | None) -> str:
    """The ``md`` format: the same records ``export()`` dumps, as Markdown."""
    rounds = [r for r in records if r.kind == KIND_ROUND]
    facts = [r for r in records if r.kind == KIND_FACT]
    by_round: dict[str | None, list[MemoryRecord]] = {}
    for fact in facts:
        by_round.setdefault(fact.round_key, []).append(fact)
    # The reverse of ``supersedes`` (set on the winner): loser id -> winner id.
    superseded_by = {
        r.supersedes: r.id for r in facts if r.supersedes is not None and r.id is not None
    }

    lines = [
        HEADER,
        "",
        f"- user_id: {user_id}",
        f"- now_logical: {now or NO_NOW}",
        f"- records: {len(records)} (rounds {len(rounds)}, facts {len(facts)})",
    ]
    current_ts = None
    for record in _ordered(rounds):
        if record.created_at != current_ts:
            current_ts = record.created_at
            lines += ["", f"## {current_ts or NO_NOW}"]
        lines += ["", _quoted(render_records([record]))]
        round_facts = by_round.pop(record.round_key, []) if record.round_key else []
        if round_facts:
            lines.append("")
            lines += [_fact_bullet(fact, superseded_by) for fact in round_facts]

    orphans = [f for group in by_round.values() for f in group]
    if orphans:
        lines += ["", "## facts with no round record", ""]
        lines += [_fact_bullet(fact, superseded_by) for fact in _ordered(orphans)]
    lines.append("")
    return "\n".join(lines)


def export_command(db: str, user_id: str, fmt: str) -> str:
    memory = open_store(db)
    if fmt == "text":
        return memory.export(user_id)
    records = memory.store.all_records(user_id)
    return export_markdown(user_id, records, now_logical(memory.store.created_ats(user_id)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mnimi",
        description="The mnimi command line. The library itself is `import mnimi`.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser(
        "export",
        help="dump a user's store as human-readable Markdown or text (no model is loaded)",
        description=(
            "Dump one user's memory store. `md` (the default) writes one heading per "
            "session, each round as a quoted block, its facts as bullets and a superseded "
            "fact struck through; `text` writes Memory.export()'s bytes unchanged."
        ),
    )
    export.add_argument("db", help="path to the store (a mnimi SQLite file)")
    export.add_argument("user_id", help="the user whose records to dump")
    export.add_argument(
        "-o", "--output", metavar="FILE", help="write here instead of stdout (UTF-8, LF)"
    )
    export.add_argument(
        "--format", choices=FORMATS, default="md", help="md (the default) or text"
    )
    return parser


def _write_stdout(text: str) -> None:
    """Write the dump as UTF-8 whatever the console's codec.

    A Windows pipe hands Python a cp1252 ``stdout``, which cannot encode the dump's
    ``→`` and would crash the command; a redirected ``mnimi export`` must still land
    the same bytes ``-o`` would write.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError, TypeError):  # pragma: no cover - a stream that refuses
            pass
    stream.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        if not os.path.isfile(args.db):
            print(f"mnimi export: no such store: {args.db}", file=sys.stderr)
            return 2
        try:
            text = export_command(args.db, args.user_id, args.format)
        except MemoryMetaError as exc:
            print(f"mnimi export: cannot open {args.db}: {exc}", file=sys.stderr)
            return 3
        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        else:
            _write_stdout(text)
        return 0
    return 2  # unreachable: argparse requires a command


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
