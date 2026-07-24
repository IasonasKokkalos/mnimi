"""sqlite-vec persistence: schema, insert, and vector KNN search.

The only module that talks to SQLite. Metadata lives in a normal ``memories``
table; embeddings live in a ``vec0`` virtual table keyed by the same rowid. A
KNN query against the vector table is joined back to metadata and scoped to a
user. Everything is one file on disk — that is the whole zero-infra pitch.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import sqlite_vec

from .models import MemoryRecord


class Store:
    """Owns the SQLite connection and the mnimi schema."""

    def __init__(self, db_path: str, dim: int) -> None:
        self.dim = dim
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._load_extension()
        self._create_schema()

    def _load_extension(self) -> None:
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

    def _create_schema(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                salience   REAL    NOT NULL DEFAULT 1.0,
                source     TEXT    NOT NULL DEFAULT 'message',
                supersedes INTEGER,
                pinned     INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
        # vec0 virtual table; rowid is shared with the memories table so the two
        # stay joined without an extra key column.
        self.db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                embedding float[{self.dim}]
            )
            """
        )
        self.db.commit()

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a record, assigning it an id and (if unset) a timestamp."""
        if record.created_at is None:
            record.created_at = datetime.now(timezone.utc).isoformat()
        cur = self.db.execute(
            """
            INSERT INTO memories
                (user_id, content, created_at, salience, source, supersedes, pinned)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.user_id,
                record.content,
                record.created_at,
                record.salience,
                record.source,
                record.supersedes,
                int(record.pinned),
            ),
        )
        record.id = int(cur.lastrowid)
        if record.embedding is not None:
            self.db.execute(
                "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                (record.id, sqlite_vec.serialize_float32(record.embedding)),
            )
        self.db.commit()
        return record

    def search(self, embedding: list[float], user_id: str, k: int = 5) -> list[MemoryRecord]:
        """Return the ``k`` nearest memories for ``user_id`` by vector distance.

        KNN over the vector index is global, so we over-fetch and then filter by
        user — otherwise a busy neighbour could crowd out the queried user.
        """
        rows = self.db.execute(
            """
            WITH knn AS (
                SELECT rowid AS memory_id, distance
                FROM vec_memories
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT m.id, m.user_id, m.content, m.created_at, m.salience,
                   m.source, m.supersedes, m.pinned, knn.distance
            FROM knn
            JOIN memories m ON m.id = knn.memory_id
            WHERE m.user_id = ?
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(embedding), max(k * 8, k), user_id),
        ).fetchall()
        return [self._row_to_record(row) for row in rows[:k]]

    def count(self, user_id: str | None = None) -> int:
        """Number of stored memories, optionally scoped to a user."""
        if user_id is None:
            row = self.db.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        else:
            row = self.db.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id = ?", (user_id,)
            ).fetchone()
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
            pinned=bool(row["pinned"]),
        )
