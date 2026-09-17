"""Storage shapes for mnimi.

`MemoryRecord` is the schema of record. Its fields constrain the entire write
path (extraction, salience, conflict resolution, decay), so the shape matters
more than anything else in this file today.

Two kinds of record share the table (PHASE2 D1, the hybrid store): a *round*
record — the v1 shape, the verbatim round embedded through ``EMBED_TEMPLATE``
— and a *fact* record — one atomic fact the extractor read out of that round,
embedded through ``FACT_EMBED_TEMPLATE``. Every record of one round carries
the same ``round_key``, so retrieval collapses them to the round and the
reader sees the round once (``Store.search_rounds``).
"""

from __future__ import annotations

from dataclasses import dataclass

KIND_ROUND = "round"
KIND_FACT = "fact"
KINDS = (KIND_ROUND, KIND_FACT)


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
    """The EMBED text of this record — for a round the ``EMBED_TEMPLATE``
    rendering, for a fact the ``FACT_EMBED_TEMPLATE`` rendering (``raw`` +
    newline + ``fact``). This exact string is what gets embedded and what the
    dedup screens key on; rendering for a reader happens from ``turns``,
    ``fact`` and ``raw``, never from here. Changing how this string is built
    moves every vector of its kind (see the two template hashes)."""

    embedding: list[float] | None = None
    """Dense vector for similarity search. Not re-hydrated on read."""

    turns: list[dict] | None = None
    """The verbatim ``{"role", "content"}`` turns of the source round, kept for
    RENDERING reader context with speaker attribution. Never embedded, never
    part of the dedup key — the embed/render split is the safety property that
    lets the rendered format change without moving a single vector. A fact
    record carries its source round's turns too."""

    created_at: str | None = None
    """The session timestamp (``ts``) of the source message, ISO-8601. The
    caller must set it before insert; the store refuses a record without one
    and never stamps wall-clock time. Plays SPEC's ``system_time`` role."""

    salience: float = 1.0
    """How much this memory matters, in [0, 1]: the ranking multiplier under
    ``ranking="score"`` (PHASE4 D4). Written at insert (the extractor's
    grammar-restricted estimate for a fact, 1.0 for a round), lowered toward
    ``decay_floor`` by ``consolidate()`` (D3), set to 0 by supersession and
    only by supersession."""

    initial_salience: float | None = None
    """The salience the record was inserted with; never updated (PHASE4 D3).
    Decay recomputes ``salience`` from it, so a pass is a pure function of
    stored fields (twice = once) and an access restores the full value at the
    next pass. ``Store.insert`` fills it from ``salience`` when unset."""

    last_accessed: str | None = None
    """The session timestamp this record was last returned by ``recall()`` at
    (``now_logical``), initialized to ``created_at`` at insert (SPEC
    §MemoryRecord, PHASE4 D2). Decay counts days from it."""

    source: str = "message"
    """Provenance, e.g. ``"user+assistant"`` (the round's roles)."""

    supersedes: int | None = None
    """Id of a memory this one replaces (conflict resolution / updates)."""

    round_key: str | None = None
    """Identity of the source round, shared by every record made from it: the
    R4 windows of a round, and in the extraction era the round record and its
    fact records. Rendering shows a round once, and ``k`` counts rounds.
    ``None`` for a round embedded whole with no extractor — the v1 shape,
    byte for byte."""

    kind: str = KIND_ROUND
    """``"round"`` (the verbatim round) or ``"fact"`` (one extracted fact)."""

    fact: str | None = None
    """The human-readable fact text (SPEC's ``content`` of a fact); fact
    records only."""

    raw: str | None = None
    """The verbatim, role-prefixed excerpt the fact was extracted from
    (``"user: …"`` / ``"assistant: …"``); immutable; fact records only."""

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    """The normalized triple, nullable as a unit; fact records only. Consumed
    by the value-substitution screen and conflict resolution (Phase 3)."""

    valid_time: str | None = None
    """When the fact holds in the world, resolved deterministically from
    ``time_mention`` against the message ``ts``: an ISO date, or a month or
    year prefix when the mention is that coarse; ``None`` = a standing fact.
    Temporal metadata, never an expiry (SPEC CHANGELOG #3)."""

    time_mention: str | None = None
    """The temporal expression the extractor copied verbatim, kept beside its
    resolution so an unresolved mention is auditable."""

    pair_key: str | None = None
    """The normalized ``subject|predicate`` of a fact's triple
    (``mnimi.conflict.normalize``), indexed so conflict candidates are found
    store-wide without a vector (PHASE3 D2). ``None`` for round records and
    for facts whose triple is null or empties under normalization. Derived,
    never rendered, never embedded."""
