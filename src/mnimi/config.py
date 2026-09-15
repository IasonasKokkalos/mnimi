"""Tunable parameters, passed to ``Memory`` at construction.

Every threshold the library reads must come from here — never hardcoded in
write- or read-path logic. Only knobs for stages that exist in code are
present; a config field nothing reads is dead weight.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    ``"round+facts"`` (default since v1.9.0 — gate 4-iii, 2026-09-15: 84 vs
    80 over ``turns`` in one sitting, b=7, c=3): the round's verbatim turns
    under a ``facts:`` header listing its extracted facts with their
    resolved dates. ``"turns"``: the turns alone — v1's unit, and what every
    other arm renders. ``"facts"``: the round's facts alone
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
