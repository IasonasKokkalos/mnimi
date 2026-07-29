"""The ``Memory`` facade: four methods over an embedder and a store.

Today these are deliberately thin. The store is real (real schema, real vector
search); the write-side intelligence — extraction, salience, conflict
resolution, decay — lands behind ``add`` and ``consolidate`` in later weeks. The
public surface stays at four methods regardless.
"""

from __future__ import annotations

import re

from .config import MemoryConfig
from .embeddings import Embedder
from .models import MemoryRecord
from .store import Store

_PUNCT_RE = re.compile(r"[^\w\s]")

_DEFAULT_CONFIG = MemoryConfig()


class Memory:
    """Embeddable agent memory backed by a single SQLite file."""

    def __init__(
        self, db_path: str, embedder: Embedder, config: MemoryConfig = _DEFAULT_CONFIG
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.store = Store(
            db_path,
            dim=embedder.dim,
            embedder_name=embedder.name,
            embedder_revision=embedder.revision,
        )

    def add(self, messages, user_id: str) -> None:
        """Write path: one record per user+assistant round, deduped.

        ``messages`` is ``list[dict]`` with ``role`` / ``content`` / ``ts`` —
        nothing else. Granularity is per-round (a user turn and its assistant
        reply), which must match ``naive_rag`` exactly or the comparison is
        confounded. The session date is folded into content so it reaches the
        reader; role stays metadata and never enters the embedded string.

        v1 dedup is exact-normalize collapse followed by ONE cosine-threshold
        probe against the store. No negation screen, no entropy gate — those
        arrive with extraction, post-v1.
        """
        rounds = _messages_to_rounds(messages)
        if not rounds:
            return
        seen = {_normalize(content) for content in self.store.contents(user_id)}
        embeddings = self.embedder.embed([content for content, _, _ in rounds])
        for (content, roles, ts), embedding in zip(rounds, embeddings, strict=True):
            normalized = _normalize(content)
            if normalized in seen:
                continue
            hits = self.store.search(embedding, user_id=user_id, k=1)
            if hits and hits[0][1] >= self.config.dedup_cosine_threshold:
                continue
            self.store.insert(
                MemoryRecord(
                    user_id=user_id,
                    content=content,
                    embedding=embedding,
                    created_at=ts,
                    source=roles,
                )
            )
            seen.add(normalized)

    def recall(self, query: str, user_id: str) -> list[MemoryRecord]:
        """Raw retrieval: the nearest stored memories, no assembly."""
        (query_embedding,) = self.embedder.embed([query])
        hits = self.store.search(query_embedding, user_id=user_id, k=self.config.top_k)
        return [record for record, _cosine in hits]

    def get_context(self, query: str, user_id: str) -> str:
        """Assemble recalled memories into a single context string for a reader.

        Ordered oldest-first, not by relevance. Retrieval order is a similarity
        ranking; handing a reader dated blocks in ranking order asks it to
        reconstruct a chronology from a shuffle, and temporal questions are 27%
        of the benchmark. The paper's own pipeline sorts retrieved items by
        timestamp before reading (§5.1).
        """
        records = _time_ordered(self.recall(query, user_id))
        return "\n".join(record.content for record in records)

    def consolidate(self, user_id: str) -> None:
        """Merge duplicates, resolve conflicts, decay stale memories.

        Stub today — not exercised by the Day 1 baselines. This is where the
        write-side policy will live.
        """
        return None


def _time_ordered(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Oldest first, ties broken by insertion order.

    Sorts on the ``created_at`` string directly: session timestamps are
    zero-padded date-first (``2023-05-20``, ``2023/05/20``), so lexicographic
    order is chronological order without parsing — and parsing would mean
    guessing a format the caller owns. The ``id`` tiebreak keeps two records
    from one session in a deterministic order.
    """
    return sorted(records, key=lambda r: (r.created_at or "", r.id or 0))


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the exact-dup key."""
    return " ".join(_PUNCT_RE.sub("", text.lower()).split())


def _messages_to_rounds(messages) -> list[tuple[str, str, str | None]]:
    """Group ``{"role", "content", "ts"}`` dicts into per-round records.

    Returns ``(content, roles, ts)`` per round: a user turn paired with the
    assistant reply that follows it, or a solo turn when no pairing exists.
    Content carries the folded session date (reader-visible; temporal questions
    die without it) and NO role labels — role is metadata, the embedded string
    is bare content (SPEC CHANGELOG #15). ``ts`` comes from the round's first
    turn and is the record's only clock.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"add() takes list[dict] messages, got {type(messages).__name__}"
        )
    turns: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(
                "add() takes list[dict] messages with 'role'/'content'/'ts', "
                f"got a list containing {type(message).__name__}"
            )
        if str(message.get("content", "")).strip():
            turns.append(message)

    rounds: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        follower = turns[index + 1] if index + 1 < len(turns) else None
        if (
            turn.get("role") == "user"
            and follower is not None
            and follower.get("role") == "assistant"
        ):
            batch = [turn, follower]
        else:
            batch = [turn]
        index += len(batch)

        ts = batch[0].get("ts")
        text = "\n".join(str(t["content"]).strip() for t in batch)
        content = f"[Session date: {_date_of(ts)}] {text}" if ts else text
        roles = "+".join(str(t.get("role")) for t in batch)
        rounds.append((content, roles, ts))
    return rounds


def _date_of(ts: str) -> str:
    """The date part of an ISO timestamp; the fold is a date, not a time."""
    return ts.split("T")[0].split(" ")[0]
