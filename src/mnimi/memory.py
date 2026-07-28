"""The ``Memory`` facade: four methods over an embedder and a store.

Today these are deliberately thin. The store is real (real schema, real vector
search); the write-side intelligence — extraction, salience, conflict
resolution, decay — lands behind ``add`` and ``consolidate`` in later weeks. The
public surface stays at four methods regardless.
"""

from __future__ import annotations

from .config import MemoryConfig
from .embeddings import Embedder
from .models import MemoryRecord
from .store import Store

_DEFAULT_CONFIG = MemoryConfig()


class Memory:
    """Embeddable agent memory backed by a single SQLite file."""

    def __init__(
        self, db_path: str, embedder: Embedder, config: MemoryConfig = _DEFAULT_CONFIG
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.store = Store(db_path, dim=embedder.dim)

    def add(self, messages, user_id: str) -> None:
        """Write path. Thin today: store each message's text as a memory.

        Real extraction / dedup / salience scoring will replace this body without
        changing the signature.
        """
        texts = _messages_to_texts(messages)
        if not texts:
            return
        embeddings = self.embedder.embed(texts)
        for text, embedding in zip(texts, embeddings, strict=True):
            self.store.insert(
                MemoryRecord(
                    user_id=user_id,
                    content=text,
                    embedding=embedding,
                    source="message",
                )
            )

    def recall(self, query: str, user_id: str) -> list[MemoryRecord]:
        """Raw retrieval: the nearest stored memories, no assembly."""
        (query_embedding,) = self.embedder.embed([query])
        return self.store.search(query_embedding, user_id=user_id, k=5)

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


def _messages_to_texts(messages) -> list[str]:
    """Normalise the many shapes ``add`` might be handed into a list of strings.

    Accepts a bare string, a list of strings, or a list of ``{"role", "content"}``
    message dicts (the LongMemEval turn shape).
    """
    if isinstance(messages, str):
        return [messages] if messages.strip() else []

    texts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = message.get("role")
            texts.append(f"{role}: {content}" if role else content)
        else:
            text = str(message).strip()
            if text:
                texts.append(text)
    return texts
