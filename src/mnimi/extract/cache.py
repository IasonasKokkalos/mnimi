"""The extraction cache: one SQLite file per extractor configuration.

A corpus is extracted once. ``CachedExtractor`` wraps any ``Extractor`` and
memoises its raw output by the round's turns — roles and content, never the
session timestamp, because the model never sees the date (the resolver reads
``when`` against ``ts`` outside the model), so identical rounds in different
sessions share one entry. The file is named by the extractor's pins hash and
records that hash inside itself, so a prompt, decode or model change is a
different file by construction and a mislabelled file is refused.

A hit re-parses the stored output through ``schema.parse_output``: a parser
fix re-reads the cache without a model call, and the cache never holds an
interpretation, only bytes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .protocol import ExtractionResult, canonical, extractor_pins_hash, sha256_text
from .schema import parse_output

CACHE_DIR = Path(".cache") / "extract"


def default_cache_path(pins: dict, base: Path | None = None) -> Path:
    """``<cwd>/.cache/extract/<extractor_pins_hash>.sqlite`` (gitignored scratch)."""
    root = Path(base) if base is not None else Path.cwd() / CACHE_DIR
    return root / f"{extractor_pins_hash(pins)}.sqlite"


class CachedExtractor:
    """Transparent cache in front of an extractor; same pins, same results."""

    def __init__(self, inner, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"hits": 0, "misses": 0}
        self.db = sqlite3.connect(str(self.path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS extractions (key TEXT PRIMARY KEY, "
            "raw_output TEXT NOT NULL, truncated INTEGER NOT NULL, "
            "truncated_input INTEGER NOT NULL DEFAULT 0)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        expected = extractor_pins_hash(inner.pins)
        row = self.db.execute(
            "SELECT value FROM cache_meta WHERE key = 'extractor_pins_hash'"
        ).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO cache_meta (key, value) VALUES ('extractor_pins_hash', ?)",
                (expected,),
            )
        elif row[0] != expected:
            self.db.close()
            raise ValueError(
                f"{self.path} was filled under extractor pins {row[0][:12]}…, not "
                f"{expected[:12]}…; a cache is one extractor configuration — use another path"
            )
        self.db.commit()

    @staticmethod
    def key(turns: list[dict]) -> str:
        """Roles and content only: what the model is shown."""
        return sha256_text(
            canonical(
                [
                    {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
                    for turn in turns
                ]
            )
        )

    @property
    def pins(self) -> dict:
        return self.inner.pins

    def extract(self, turns: list[dict]) -> ExtractionResult:
        key = self.key(turns)
        row = self.db.execute(
            "SELECT raw_output, truncated, truncated_input FROM extractions WHERE key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            self.stats["hits"] += 1
            raw_output, truncated, truncated_input = row[0], bool(row[1]), bool(row[2])
            facts, bad_parse = parse_output(raw_output)
            if truncated or bad_parse:
                facts = []
            return ExtractionResult(facts=facts, truncated=truncated or bad_parse,
                                    raw_output=raw_output, truncated_input=truncated_input)
        self.stats["misses"] += 1
        result = self.inner.extract(turns)
        self.db.execute(
            "INSERT OR REPLACE INTO extractions (key, raw_output, truncated, truncated_input) "
            "VALUES (?, ?, ?, ?)",
            (key, result.raw_output, int(result.truncated), int(result.truncated_input)),
        )
        self.db.commit()
        return result

    def close(self) -> None:
        self.db.close()
