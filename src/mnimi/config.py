"""Tunable parameters, passed to ``Memory`` at construction.

Every threshold the library reads must come from here — never hardcoded in
write- or read-path logic. Only knobs for stages that exist in code are
present; a config field nothing reads is dead weight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    dedup_cosine_threshold: float = 0.85
    """Cosine similarity at or above which an incoming record is a duplicate."""

    top_k: int = 10
    """How many records ``recall`` retrieves."""
