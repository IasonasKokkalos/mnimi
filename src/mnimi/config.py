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
