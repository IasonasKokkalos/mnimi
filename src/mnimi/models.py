"""Storage shapes for mnimi.

`MemoryRecord` is the schema of record. Its fields constrain the entire write
path (extraction, salience, conflict resolution, decay), so the shape matters
more than anything else in this file today.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class MemoryRecord:
    """A single stored memory.

    Keyword-only so the field order below can mirror the spec without fighting
    dataclass default-ordering rules.
    """

    id: int | None = None
    """Rowid assigned by the store on insert; ``None`` until persisted."""

    user_id: str
    """Owner of the memory. All retrieval is scoped to a user."""

    content: str
    """The memory text itself — the EMBED text. This exact string is what gets
    embedded and what the dedup screens key on; rendering for a reader happens
    from ``turns``, never from here. Changing how this string is built moves
    every vector (see ``memory.EMBED_TEMPLATE``)."""

    embedding: list[float] | None = None
    """Dense vector for similarity search. Not re-hydrated on read."""

    turns: list[dict] | None = None
    """The verbatim ``{"role", "content"}`` turns of the source round, kept for
    RENDERING reader context with speaker attribution. Never embedded, never
    part of the dedup key — the embed/render split is the safety property that
    lets the rendered format change without moving a single vector."""

    created_at: str | None = None
    """The session timestamp (``ts``) of the source message, ISO-8601. The
    caller must set it before insert; the store refuses a record without one
    and never stamps wall-clock time."""

    salience: float = 1.0
    """How much this memory matters. Drives ranking and decay later."""

    source: str = "message"
    """Provenance, e.g. ``"message"``, ``"summary"``, ``"fact"``."""

    supersedes: int | None = None
    """Id of a memory this one replaces (conflict resolution / updates)."""
