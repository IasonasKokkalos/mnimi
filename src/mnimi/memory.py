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
        """Write path: store each entry unless dedup says it is already known.

        v1 dedup is exact-normalize collapse followed by ONE cosine-threshold
        probe against the store. No negation screen, no entropy gate — those
        arrive with extraction, post-v1.
        """
        entries = _messages_to_texts(messages)
        if not entries:
            return
        seen = {_normalize(content) for content in self.store.contents(user_id)}
        embeddings = self.embedder.embed([text for text, _ in entries])
        for (text, ts), embedding in zip(entries, embeddings, strict=True):
            normalized = _normalize(text)
            if normalized in seen:
                continue
            hits = self.store.search(embedding, user_id=user_id, k=1)
            if hits and hits[0][1] >= self.config.dedup_cosine_threshold:
                continue
            self.store.insert(
                MemoryRecord(
                    user_id=user_id,
                    content=text,
                    embedding=embedding,
                    created_at=ts,
                    source="message",
                )
            )
            seen.add(normalized)

    def recall(self, query: str, user_id: str) -> list[MemoryRecord]:
        """Raw retrieval: the nearest stored memories, no assembly."""
        (query_embedding,) = self.embedder.embed([query])
        hits = self.store.search(query_embedding, user_id=user_id, k=self.config.top_k)
        return [record for record, _cosine in hits]

    def get_context(self, query: str, user_id: str) -> str:
        """Assemble recalled memories into a single context string for a reader."""
        records = self.recall(query, user_id)
        return "\n".join(record.content for record in records)

    def consolidate(self, user_id: str) -> None:
        """Merge duplicates, resolve conflicts, decay stale memories.

        Stub today — not exercised by the Day 1 baselines. This is where the
        write-side policy will live.
        """
        return None


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the exact-dup key."""
    return " ".join(_PUNCT_RE.sub("", text.lower()).split())


def _messages_to_texts(messages) -> list[tuple[str, str | None]]:
    """Normalise the shapes ``add`` might be handed into ``(text, ts)`` pairs.

    Accepts a bare string, a list of strings, or a list of
    ``{"role", "content", "ts"}`` message dicts (the LongMemEval turn shape).
    Only dicts can carry a ``ts``; the store refuses records without one.
    """
    if isinstance(messages, str):
        return [(messages, None)] if messages.strip() else []

    entries: list[tuple[str, str | None]] = []
    for message in messages:
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = message.get("role")
            text = f"{role}: {content}" if role else content
            entries.append((text, message.get("ts")))
        else:
            text = str(message).strip()
            if text:
                entries.append((text, None))
    return entries
