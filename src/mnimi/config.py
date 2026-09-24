"""Tunable parameters, passed to ``Memory`` at construction.

Every threshold the library reads must come from here — never hardcoded in
write- or read-path logic. Only knobs for stages that exist in code are
present; a config field nothing reads is dead weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .embeddings import BGE_QUERY_INSTRUCTION


@dataclass(frozen=True)
class MemoryConfig:
    dedup_cosine_threshold: float = 0.95
    """Cosine similarity at or above which an incoming record is a duplicate.

    0.95, not the 0.85 v1 shipped with. Measured on a stratified LongMemEval
    slice with the pinned BGE embedder: 0.85 discarded 37.7% of the corpus and
    44.2% of annotated evidence rounds (12/12 questions lost at least one),
    because the max-cosine-to-any-earlier-round distribution for whole
    conversational rounds has median 0.839 — 0.85 sits at the median of the
    noise. At 0.95 evidence loss falls to 2.3%. See docs/DECISIONS.md.
    """

    top_k: int = 10
    """How many records ``recall`` retrieves."""

    dedup_entropy_gate: float = 2.0
    """Token-level Shannon entropy (bits) below which a cosine-gate-pass pair
    of FACT records is not auto-merged (SPEC §Dedup strategy step 5, PHASE3
    D5). The shorter side's entropy is compared: with fewer than four distinct
    tokens (2.0 bits) a cosine is not trusted to say "same fact", and both
    records are kept. SPEC's default, unmeasured — reported, never tuned on
    the benchmark. A harness flag (``--dedup-entropy-gate``) and a pin."""

    conflict_resolution: bool = True
    """The Phase 3 write-side stage as one switch: the negation and
    value-substitution screens and the entropy gate over cosine-gate-pass fact
    pairs (SPEC steps 3–5), and supersession between conflicting facts (SPEC
    write path step 4). ``False`` is the v1.9 write path byte for byte — a
    fact that trips the cosine gate is dropped, nothing is ever superseded —
    which is gate 3-ii's baseline and the fallback the pre-registration keeps
    the code behind on a gate loss (DECISIONS "Phase 3 pre-registration"). It
    never touches a round record or the read path. A harness flag
    (``--conflict-resolution``) and a pin."""

    decay_half_life_days: float = 30.0
    """Logical days over which ``consolidate()`` halves an active record's
    salience since its ``last_accessed`` (SPEC write path step 5, PHASE4 D3).
    SPEC's default, unmeasured and never tuned on the benchmark; also the time
    constant of the ``recency`` component (D4). Must be > 0. A harness flag
    (``--decay-half-life-days``) and a pin."""

    decay_floor: float = 0.15
    """The salience decay never goes below (SPEC CHANGELOG #11): decay
    down-ranks, never excludes; 0 is reserved for supersession. A record
    inserted below the floor keeps its initial value (decay never raises).
    Must lie in (0, 1]. A harness flag (``--decay-floor``) and a pin."""

    ranking: str = "score"
    """How ``recall`` orders rounds (PHASE4 D4; ``"rerank"`` added by PHASE6 D5 —
    the ``"score"`` top-``rerank_pool`` reordered by the injected cross-encoder,
    ``mnimi.rerank``). ``"score"`` — the default since
    v1.11.0, gate 4-iii: 87 vs 86, b=2, c=1 — is SPEC §Ranking: every record
    scored ``(w_sim * relevance + w_rec * recency) * salience``, a round by its
    best record, the exact top-k by score (``mnimi.ranking.rank_rounds``); the
    extractor's salience (0.5 on most of the assistant's facts) and decay reach
    the ranking only here. ``"similarity"`` is the v1.10 read path byte for
    byte: ``Store.search_rounds``, a round by its most similar record, stored
    salience not read. A harness flag (``--ranking``) and a pin."""

    active_only: bool = True
    """Whether the read path sees only ACTIVE records, ``salience > 0`` (SPEC
    §Retriever, §get_context; PHASE4 D5): a superseded fact neither ranks nor
    renders under a round's ``facts:`` header. Rounds are never superseded, so
    every round stays. ``True`` is the default since v1.11.0 — gate 4-ii: 0
    evidence rounds left the top-10, ANY@10 93/95 and ALL@10 83/95 unchanged.
    ``False`` is the v1.10 read path. A harness flag (``--active-only``) and a
    pin."""

    recall_min_relevance: float = 0.0
    """A relevance floor on ``recall``'s returned rounds (SPEC §MemoryConfig,
    PHASE4 D7): after the top-k, a round whose ``relevance`` is below it is
    dropped and nothing fills its place. ``0.0`` is off by definition — the drop
    is skipped, whatever the cosines. Unmeasured, never set in a run (the FUTURE
    trigger, distractor-driven reading failures, has not fired). Must lie in
    [-1, 1]. A harness flag (``--recall-min-relevance``) and a pin."""

    salience_weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"similarity": 1.0, "recency": 0.0})
    )
    """SPEC §Ranking's two weights, read only under ``ranking="score"``:
    ``similarity`` on the cosine, ``recency`` on
    ``0.5 ** (days since the record's session / decay_half_life_days)``. SPEC's
    defaults: similarity alone (LongMemEval §5.1); a recency weight > 0 is a
    FUTURE item with a trigger and is never set in a run. Exactly those two keys,
    each finite and >= 0. A harness flag (``--salience-weights``) and a pin."""

    rerank_pool: int = 50
    """Under ``ranking="rerank"`` (PHASE6 D5): how many score-ranked rounds the
    cross-encoder reorders before the top-``top_k`` are taken. 50 is the
    probe's own candidate width (``SEARCH_K``), fixed in the pre-registration
    and never swept; must be >= ``top_k``. The reranker itself is injected
    (``Memory(..., reranker=)``) and pinned by name and revision. A harness
    flag (``--rerank-pool``) and a pin."""

    time_weight: float = 0.05
    """The weight of the time-aware term in SPEC's score (PHASE6 D3):
    ``(w_sim * relevance + w_rec * recency + time_weight * time_match) * salience``,
    where ``time_match`` is 1 when the record's effective time (``valid_time``,
    else its session date) falls inside the window the query's own relative-date
    expression names, anchored on the question date the query carries as a
    documented prefix (``mnimi.temporal``, D2). ``0.0`` is the v2.3 ranking
    byte for byte (the published ``mnimi__500q_gpt4o`` configuration). The
    default is **0.05** since v2.12.0 — fixed before any probe from a
    design-time reading of the rank-10/rank-20 cosine gap, never swept, and
    adopted at gate 6-iv arm 1 (2026-09-24: 429 vs the published 422, b=15,
    c=8; DECISIONS "Gate 6-iv arm 1 read"). Without a dated query the term is
    inert whatever the weight. A harness flag (``--time-weight``) and a pin,
    beside ``temporal_rules_hash``. Must be finite and >= 0."""

    dedup_scope: str = "session"
    """Where the cosine dedup screen looks for a duplicate.

    ``"session"`` (the default since R3, 2026-09-12): only a record carrying
    the same ``ts`` — a near-duplicate from another session is a repeat or an
    update, and the date it carries is information the reader needs.
    ``"store"``: any earlier record of the user (v1 as first shipped, and the
    published local-family configuration). Measured on the gpt-4o family:
    store scope had dropped an annotated evidence round on five questions,
    four of them against another session; session scope recovered those four
    (probe) and scored 79 against 77 for the same commit's store-scope arm
    (b=5, c=3 — DECISIONS "R3 read"). The exact-normalize collapse is
    unaffected by the scope (its key already folds the session date in). The
    extraction era revisits this with ``valid_time``: a cross-session
    duplicate becomes a supersede, not a drop.
    """

    query_instruction: str = BGE_QUERY_INSTRUCTION
    """Text prepended to every query before it is embedded; never to stored text.

    The default is the BGE v1.5 model card's retrieval instruction (R5,
    adopted 2026-09-13: on the gpt-4o family 81 against 79 for the same
    commit's bare-question arm, b=5, c=3; probe: recall equal at k=10, better
    at k=20/50). ``""`` embeds the bare question — v1 as first shipped and the
    published local-family configuration. Query side only: stored vectors,
    the dedup key and ``memory_meta`` do not move, so it is a harness pin
    (``query_instruction``) and not a store guard.
    """

    chunk_tokens: int = 0
    """Embed a round longer than this many embedder tokens as overlapping
    windows, one record per window (R4, 2026-09-12). ``0`` (v1 as shipped)
    embeds each round once and lets the embedder truncate it — 41.95% of
    LongMemEval rounds exceed BGE's 512-token window (09-08_analysis §C13),
    so their tails are invisible to retrieval. Every window of a round shares
    its ``round_key`` and the reader still sees the round once. Changes which
    strings are embedded, so it is a ``memory_meta`` key and a harness pin;
    the ``EMBED_TEMPLATE`` is unchanged (each window is templated the same
    way). ``510`` leaves BGE room for ``[CLS]``/``[SEP]``."""

    chunk_overlap: int = 64
    """Tokens shared by consecutive windows when ``chunk_tokens`` > 0. In
    ``memory_meta`` and the pins beside ``chunk_tokens``."""

    render_unit: str = "round+facts"
    """What ``get_context`` shows a retrieved round as (PHASE2 D5).

    ``"round+facts"`` (the default since v1.9.0 — gate 4-iii, 2026-09-15: 84
    vs 80 over ``turns`` at n=100, b=7, c=3 — and again since v2.13.0, the
    maintainer's choice of 2026-09-25 after Phase 6): the round's verbatim
    turns under a ``facts:`` header listing its extracted facts with their
    resolved dates. ``"turns"``: the turns alone — v1's unit, and what every
    other arm renders; with an extractor the facts are still stored and
    retrieved, just not rendered. Phase 6 measured it at n=500 (gate 6-iv arm
    2: 424 vs the published 422, b=22, c=20, adopted by the pre-registered
    rule; 10 of the 11 rows the analysis blamed on the header were right
    without it, and the dated lines cost temporal-reasoning 111 → 106) and
    was the default for one commit (v2.12.0); the shipped configuration is
    L1 alone — ``time_weight=0.05`` over this unit, the arm that read 429 —
    by the maintainer's ruling, three above the L1 + L3 headline on a
    descriptive pair (DECISIONS "The shipped default is L1 alone").
    ``"facts"``: the round's facts alone
    (``fact:`` / ``source:`` lines; a round with no facts falls back to its
    turns) — the one pre-registered alternative, measured worse (78 vs 84,
    b=2, c=8: it drops the verbatim turns single-session questions need). A
    system-level pin
    (``render_unit_template_hash`` in mnimi's retrieval pins), not a harness
    parity field: the render *format* stays identical across arms, the unit
    is what the memory layer hands the reader. Without an extractor no record
    has facts and every unit renders the turns. Decided by the gate 4-iii
    sitting (DECISIONS "Phase 2 pre-registration"), never on taste.
    """

    render_format: str = "text"
    """How ``get_context`` frames the recalled turns for a reader.

    ``"text"`` (default): one ``[Session date: …]`` header per timestamp
    change, then ``role: content`` lines. ``"json"``: the same blocks as a JSON
    array (LongMemEval §5.5's structured presentation). The framing is the
    only difference; neither touches a vector or the dedup key. The eval
    harness pins ``render_template_hash(render_format)`` so a format change is
    a different configuration, never a silent edit. The default is decided by
    the gpt-4o era's presentation pair (``docs/DECISIONS.md``, 2026-09-12).
    """
