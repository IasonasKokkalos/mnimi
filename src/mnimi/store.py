"""sqlite-vec persistence: schema, insert, and vector KNN search.

The only module that talks to SQLite. Metadata lives in a normal ``memories``
table; embeddings live in ``vec0`` virtual tables keyed by the same rowid —
``vec_memories`` for round records (the v1 table, byte for byte) and
``vec_facts`` for fact records (the extraction era), so a search scoped to one
kind never has to over-fetch past the other. A KNN query is joined back to
metadata and scoped to a user. Everything is one file on disk — that is the
whole zero-infra pitch.
"""

from __future__ import annotations

import json
import sqlite3

import sqlite_vec

from .extract.protocol import PIN_KEYS
from .models import KIND_FACT, KIND_ROUND, KINDS, MemoryRecord

#: The value a guard row takes when the stage it describes is not in use — a
#: store built without an extractor writes it for the five extractor rows.
#: (``negation_lexicon_hash`` carried it until Phase 3 built the lexicon.)
GUARD_NONE = "none"

_VEC_TABLE = {KIND_ROUND: "vec_memories", KIND_FACT: "vec_facts"}
_COLUMNS = (
    "m.id, m.user_id, m.content, m.created_at, m.salience, m.source, m.supersedes, "
    "m.turns, m.round_key, m.kind, m.fact, m.raw, m.subject, m.predicate, m.object, "
    "m.valid_time, m.time_mention, m.pair_key"
)


class MemoryMetaError(RuntimeError):
    """The store's pinned identity does not match what the caller provided.

    Raised at open time so a mismatched embedder fails loudly as a
    reproducibility error instead of surfacing later as an opaque sqlite-vec
    dimension error at insert time.
    """


class Store:
    """Owns the SQLite connection and the mnimi schema."""

    def __init__(
        self,
        db_path: str,
        dim: int,
        *,
        embedder_name: str,
        embedder_revision: str,
        embed_template_hash: str,
        fact_embed_template_hash: str,
        prefilter_lexicon_hash: str,
        resolver_version: str,
        negation_lexicon_hash: str,
        conflict_rules_hash: str,
        extractor_pins: dict | None = None,
        chunk_tokens: int = 0,
        chunk_overlap: int = 64,
    ) -> None:
        self.dim = dim
        self.embedder_name = embedder_name
        self.embedder_revision = embedder_revision
        self.embed_template_hash = embed_template_hash
        # Which strings were embedded: a round whole, or windows of it. Two
        # stores built under different chunking hold different vectors for
        # the same input, so the pair is guarded like the template hash.
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_overlap = int(chunk_overlap)
        # The extraction era's rows (SPEC §Reproducibility guard, PHASE2 D11):
        # what produced the fact records, or GUARD_NONE when nothing did.
        pins = dict(extractor_pins or {})
        if extractor_pins is not None:
            missing = [key for key in PIN_KEYS if key not in pins]
            if missing:
                raise ValueError(f"extractor pins are missing {missing}")
        self.extractor_pins = {key: str(pins.get(key, GUARD_NONE)) for key in PIN_KEYS}
        self.fact_embed_template_hash = fact_embed_template_hash
        self.prefilter_lexicon_hash = prefilter_lexicon_hash
        # Phase 3 (D12): the negation lexicon and the normalization / conflict
        # rules decide which facts stay active; both are frozen and hashed.
        self.negation_lexicon_hash = negation_lexicon_hash
        self.conflict_rules_hash = conflict_rules_hash
        self.resolver_version = resolver_version
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._load_extension()
        self._init_schema()

    def _load_extension(self) -> None:
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

    def _expected_meta(self) -> dict[str, str]:
        return {
            "embedder_name": self.embedder_name,
            "embedder_revision": self.embedder_revision,
            "embedder_dim": str(self.dim),
            # The embed-text template form. An edit to it moves every vector
            # while leaving the embedder pins untouched, which is exactly the
            # silent mismatch this guard exists to refuse.
            "embed_template_hash": self.embed_template_hash,
            # The chunking under which the vectors were built (R4). A v1
            # store has no such keys and is refused on open, by design.
            "chunk_tokens": str(self.chunk_tokens),
            "chunk_overlap": str(self.chunk_overlap),
            # The extraction era (PHASE2 D11). A store built before it lacks
            # every row below and is refused; a store built without an
            # extractor carries GUARD_NONE and is refused by an open that
            # brings one, and vice versa.
            **self.extractor_pins,
            "negation_lexicon_hash": self.negation_lexicon_hash,
            "prefilter_lexicon_hash": self.prefilter_lexicon_hash,
            "fact_embed_template_hash": self.fact_embed_template_hash,
            "resolver_version": self.resolver_version,
            # Phase 3 (D3/D12): the sixteenth row. A v1.9 store lacks it and
            # is refused at open.
            "conflict_rules_hash": self.conflict_rules_hash,
        }

    def _init_schema(self) -> None:
        tables = {
            row["name"]
            for row in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "memory_meta" in tables:
            self._validate_meta()
            return
        if "memories" in tables:
            raise MemoryMetaError(
                "existing database has no memory_meta table; it predates the "
                "reproducibility guard and cannot be validated - re-ingest into a fresh store"
            )
        self._create_schema()
        self._write_meta()

    def _write_meta(self) -> None:
        # Written once at DB creation, validated on every open. A mismatch means
        # the store and the caller disagree about what the vectors are.
        self.db.execute(
            "CREATE TABLE memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.db.executemany(
            "INSERT INTO memory_meta (key, value) VALUES (?, ?)",
            list(self._expected_meta().items()),
        )
        self.db.commit()

    def _validate_meta(self) -> None:
        stored = {
            row["key"]: row["value"]
            for row in self.db.execute("SELECT key, value FROM memory_meta")
        }
        mismatches = [
            f"{key}: store has {stored.get(key)!r}, caller provided {value!r}"
            for key, value in self._expected_meta().items()
            if stored.get(key) != value
        ]
        if mismatches:
            raise MemoryMetaError(
                "memory_meta mismatch - this store was created under a different "
                "pin: " + "; ".join(mismatches) + ". Re-ingest into a "
                "fresh store; never mix pins under a run."
            )

    def _create_schema(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                salience     REAL    NOT NULL DEFAULT 1.0,
                source       TEXT    NOT NULL DEFAULT 'message',
                supersedes   INTEGER,
                turns        TEXT,
                round_key    TEXT,
                kind         TEXT    NOT NULL DEFAULT 'round',
                fact         TEXT,
                raw          TEXT,
                subject      TEXT,
                predicate    TEXT,
                object       TEXT,
                valid_time   TEXT,
                time_mention TEXT,
                pair_key     TEXT
            )
            """
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_round ON memories(user_id, round_key)"
        )
        # Conflict candidates are found through the normalized pair, store-wide,
        # never through a vector (PHASE3 D2).
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_pair ON memories(user_id, pair_key)"
        )
        # vec0 virtual tables; rowid is shared with the memories table so the
        # two stay joined without an extra key column. distance_metric is
        # spelled out even though L2 is the default: the ranking contract
        # depends on it. One table per record kind (see the module docstring).
        for table in _VEC_TABLE.values():
            self.db.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
                    embedding float[{self.dim}] distance_metric=L2
                )
                """
            )
        self.db.commit()

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record, assigning it an id.

        The record must carry its timestamp already: ``ts`` from the ingested
        message is the only clock. The store never reads wall-clock — doing so
        would make the same DB score differently on different days.
        """
        if record.created_at is None:
            raise ValueError(
                "record has no timestamp; pass the message ts as created_at - "
                "ts is the only clock, the store never stamps wall-clock time"
            )
        if record.kind not in KINDS:
            raise ValueError(f"unknown record kind {record.kind!r}; expected one of {KINDS}")
        cur = self.db.execute(
            """
            INSERT INTO memories
                (user_id, content, created_at, salience, source, supersedes, turns, round_key,
                 kind, fact, raw, subject, predicate, object, valid_time, time_mention,
                 pair_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.user_id,
                record.content,
                record.created_at,
                record.salience,
                record.source,
                record.supersedes,
                json.dumps(record.turns) if record.turns is not None else None,
                record.round_key,
                record.kind,
                record.fact,
                record.raw,
                record.subject,
                record.predicate,
                record.object,
                record.valid_time,
                record.time_mention,
                record.pair_key,
            ),
        )
        record.id = int(cur.lastrowid)
        if record.embedding is not None:
            self.db.execute(
                f"INSERT INTO {_VEC_TABLE[record.kind]}(rowid, embedding) VALUES (?, ?)",
                (record.id, sqlite_vec.serialize_float32(record.embedding)),
            )
        self.db.commit()
        return record

    def _knn(self, kind: str, embedding: list[float], user_id: str, k: int) -> list[sqlite3.Row]:
        # KNN over the vector index is global, so we over-fetch and then filter
        # by user — otherwise a busy neighbour could crowd out the queried user.
        return self.db.execute(
            f"""
            WITH knn AS (
                SELECT rowid AS memory_id, distance
                FROM {_VEC_TABLE[kind]}
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT {_COLUMNS}, knn.distance
            FROM knn
            JOIN memories m ON m.id = knn.memory_id
            WHERE m.user_id = ?
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(embedding), max(k * 8, k), user_id),
        ).fetchall()

    def _has_facts(self, user_id: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM memories WHERE user_id = ? AND kind = ? LIMIT 1", (user_id, KIND_FACT)
        ).fetchone()
        return row is not None

    def search(
        self, embedding: list[float], user_id: str, k: int, kind: str | None = None
    ) -> list[tuple[MemoryRecord, float]]:
        """Return the ``k`` nearest memories for ``user_id`` with cosine similarity.

        ``kind`` scopes the search to round records or fact records (the dedup
        screens compare like with like — PHASE2 D3); ``None`` searches both
        and merges by distance, rounds first on a tie. With no fact records in
        the store the unscoped search is the v1 query, row for row.

        Vectors are unit-length (embeddings.py normalizes at the boundary), so
        L2 distance converts exactly: cos = 1 − d²/2. This is the ONE place the
        conversion happens; callers compare genuine cosine numbers.
        """
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown record kind {kind!r}; expected one of {KINDS}")
        if kind is None:
            rows = self._knn(KIND_ROUND, embedding, user_id, k)
            if self._has_facts(user_id):
                rows = sorted(
                    [*rows, *self._knn(KIND_FACT, embedding, user_id, k)],
                    key=lambda row: row["distance"],
                )
        else:
            rows = self._knn(kind, embedding, user_id, k)
        return [
            (self._row_to_record(row), 1.0 - (row["distance"] ** 2) / 2.0) for row in rows[:k]
        ]

    def search_rounds(
        self, embedding: list[float], user_id: str, k: int
    ) -> list[tuple[MemoryRecord, float]]:
        """The ``k`` nearest *rounds*: every record of one round counts once.

        A round stored as several records (``round_key`` set — R4 windows, or
        the extraction era's round record plus its fact records) can put
        several records into a top-k that the reader then sees as one round;
        left uncollapsed, a long round crowds other rounds out of the context
        (the R4 probe: ANY@10 fell 91 → 86 with windows counted separately).
        ``k`` is defined over rounds, so this over-fetches records and keeps
        each round's best-ranked record until ``k`` distinct rounds are in
        hand or the store is exhausted. With no keyed records the result is
        exactly :meth:`search` — the v1 read path, byte for byte.
        """
        fetch = k
        while True:
            hits = self.search(embedding, user_id=user_id, k=fetch)
            seen: set = set()
            distinct: list[tuple[MemoryRecord, float]] = []
            for record, cosine in hits:
                key = record.round_key if record.round_key is not None else ("id", record.id)
                if key in seen:
                    continue
                seen.add(key)
                distinct.append((record, cosine))
            if len(distinct) >= k or len(hits) < fetch:
                return distinct[:k]
            fetch *= 4

    def contents(self, user_id: str, kind: str | None = None) -> list[str]:
        """Stored content strings for a user, optionally one kind — dedup's exact screen."""
        return [content for _created_at, content in self.contents_with_ts(user_id, kind)]

    def contents_with_ts(self, user_id: str, kind: str | None = None) -> list[tuple[str, str]]:
        """``(created_at, content)`` pairs — the session-scoped exact screen's key."""
        if kind is None:
            rows = self.db.execute(
                "SELECT created_at, content FROM memories WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT created_at, content FROM memories WHERE user_id = ? AND kind = ? "
                "ORDER BY id",
                (user_id, kind),
            ).fetchall()
        return [(row["created_at"], row["content"]) for row in rows]

    def facts_of(self, user_id: str, round_key: str) -> list[MemoryRecord]:
        """The fact records of one round, in insertion order — the render header."""
        rows = self.db.execute(
            f"SELECT {_COLUMNS} FROM memories m WHERE m.user_id = ? AND m.round_key = ? "
            f"AND m.kind = ? ORDER BY m.id",
            (user_id, round_key, KIND_FACT),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def active_facts_by_pair(self, user_id: str, pair_key: str) -> list[MemoryRecord]:
        """The user's ACTIVE fact records on one normalized pair — conflict candidates (PHASE3 D2).

        Found through the pair index, store-wide, never through a vector.
        ``salience > 0``: a superseded fact is never a candidate again.
        """
        rows = self.db.execute(
            f"SELECT {_COLUMNS} FROM memories m WHERE m.user_id = ? AND m.kind = ? "
            f"AND m.pair_key = ? AND m.salience > 0 ORDER BY m.id",
            (user_id, KIND_FACT, pair_key),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def facts_with_pair_key(self, user_id: str) -> list[MemoryRecord]:
        """Every fact record with a pair key, active or not, in id order (``consolidate``)."""
        rows = self.db.execute(
            f"SELECT {_COLUMNS} FROM memories m WHERE m.user_id = ? AND m.kind = ? "
            f"AND m.pair_key IS NOT NULL ORDER BY m.id",
            (user_id, KIND_FACT),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def supersede(self, loser_id: int, winner_id: int) -> None:
        """Mark ``loser_id`` superseded by ``winner_id`` (SPEC write path step 4).

        The loser's ``salience`` becomes 0 — the value reserved for supersession
        and nothing else — and the winner's ``supersedes`` points at the last
        loser it defeated (the highest id; it is one integer). History is kept,
        never deleted. Idempotent.
        """
        self.db.execute("UPDATE memories SET salience = 0 WHERE id = ?", (loser_id,))
        self.db.execute(
            "UPDATE memories SET supersedes = ? WHERE id = ? "
            "AND (supersedes IS NULL OR supersedes < ?)",
            (loser_id, winner_id, loser_id),
        )
        self.db.commit()

    def count(self, user_id: str | None = None, kind: str | None = None) -> int:
        """Number of stored memories, optionally scoped to a user and a kind."""
        clauses, params = [], []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.db.execute(f"SELECT COUNT(*) AS n FROM memories{where}", params).fetchone()
        return int(row["n"])

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        # Embeddings are not re-hydrated on read; callers work from content.
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            embedding=None,
            created_at=row["created_at"],
            salience=row["salience"],
            source=row["source"],
            supersedes=row["supersedes"],
            turns=json.loads(row["turns"]) if row["turns"] else None,
            round_key=row["round_key"],
            kind=row["kind"],
            fact=row["fact"],
            raw=row["raw"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            valid_time=row["valid_time"],
            time_mention=row["time_mention"],
            pair_key=row["pair_key"],
        )
