"""Logical time and decay: SPEC write path step 5 as pure functions (PHASE4 D1-D3).

``now_logical`` is the latest session timestamp present in a user's store —
never the wall clock (SPEC §Logical time). ``logical_days`` counts whole days
between two session timestamps through the resolver's anchor parse, the same
parse the conflict ordering reads. ``decayed_salience`` is the half-life
formula with the floor clamp: decay lowers a salience toward
``decay_floor``, never below it and never above the value the record was
written with; salience 0 belongs to supersession alone.

The rules are frozen as text and hashed into ``memory_meta``
(``decay_rules_hash``): a store whose stored salience was produced under
different rules is refused at open. The decay *values* (half-life, floor) are
configuration, not guard rows — the next ``consolidate()`` rewrites every
active salience from ``initial_salience`` under whatever values it is given.
"""

from __future__ import annotations

from collections.abc import Iterable

from .conflict.supersede import created_at_key
from .extract.protocol import canonical, sha256_text
from .extract.resolver import anchor_date

DECAY_RULES_VERSION = "v1"

#: The decay and access rules as frozen text, hashed into ``memory_meta``
#: (``decay_rules_hash``, PHASE4 D9). An edit is a version bump, a new hash, a
#: re-ingest and a dated DECISIONS entry — never an edit under a run.
DECAY_RULES = (
    "now_logical = the user's created_at with the greatest created_at_key; ties: greater string",
    "days(later, earlier) = (anchor_date(later) - anchor_date(earlier)).days; 0 if undated or < 0",
    "insert: last_accessed = created_at and initial_salience = salience unless given",
    "insert: salience and initial_salience lie in [0, 1], else ValueError",
    "recall: last_accessed = now_logical on every record it returns",
    "consolidate: after the conflict pass, for every record with salience > 0:",
    "factor = 0.5 ** (days(now_logical, last_accessed) / decay_half_life_days)",
    "salience = max(min(decay_floor, initial_salience), initial_salience * factor)",
    "salience 0 is written by supersession only; never decayed, never restored",
)


def decay_rules_hash() -> str:
    """Digest of the frozen decay rules (``memory_meta``)."""
    return sha256_text(canonical({"version": DECAY_RULES_VERSION, "rules": list(DECAY_RULES)}))


def now_logical(timestamps: Iterable[str | None]) -> str | None:
    """The latest dated session timestamp, verbatim; ``None`` when none carries a date."""
    dated = [ts for ts in timestamps if ts and created_at_key(ts) != (0, 0, 0)]
    if not dated:
        return None
    return max(dated, key=lambda ts: (created_at_key(ts), ts))


def logical_days(later: str | None, earlier: str | None) -> int:
    """Whole days from ``earlier`` to ``later``; 0 when either is undated or it is negative."""
    a, b = anchor_date(later), anchor_date(earlier)
    if a is None or b is None:
        return 0
    return max(0, (a - b).days)


def decayed_salience(initial: float, days: int, half_life_days: float, floor: float) -> float:
    """``max(min(floor, initial), initial * 0.5 ** (days / half_life_days))`` (SPEC step 5)."""
    return max(min(floor, initial), initial * 0.5 ** (days / half_life_days))
