"""Tunable parameters, passed to ``Memory`` at construction.

Every threshold the library reads must come from here — never hardcoded in
write- or read-path logic. Only knobs for stages that exist in code are
present; a config field nothing reads is dead weight.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    dedup_scope: str = "store"
    """Where the cosine dedup screen looks for a duplicate.

    ``"store"``: any earlier record of the user (v1 as shipped). ``"session"``:
    only a record carrying the same ``ts`` — a near-duplicate from another
    session is a repeat or an update, and the date it carries is information
    the reader needs. R3 (2026-09-12): on the gpt-4o family three of the six
    rows naive_rag won were rows where the store-scope screen had dropped an
    annotated evidence round. The exact-normalize collapse is unaffected by
    the scope (its key already folds the session date in). The extraction era
    revisits this with ``valid_time``: a cross-session duplicate becomes a
    supersede, not a drop.
    """

    query_instruction: str = ""
    """Text prepended to every query before it is embedded; never to stored text.

    ``""`` (v1 as shipped) embeds the bare question. The BGE v1.5 model card's
    retrieval instruction is ``mnimi.embeddings.BGE_QUERY_INSTRUCTION``
    (R5, 2026-09-12; 09-08_analysis §B11 found it absent). Query side only:
    stored vectors, the dedup key and ``memory_meta`` do not move, so it is a
    harness pin (``query_instruction``) and not a store guard.
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
