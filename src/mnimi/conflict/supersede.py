"""Conflict resolution between two facts: which one stays active (PHASE3 D2, D6).

``conflict_between`` says whether two stored fact records with one ``pair_key``
contradict each other — by one of three rules — and ``beats`` says which of
the two is current. Both are pure functions of stored fields; nothing here
reads a clock (SPEC §Logical time) or a model.

The three rules (D2), applied only to facts whose subject is not the
assistant (its statements are reports, not state):

* ``negation`` — the same object with opposite polarity ("likes jazz" /
  "dislikes jazz", "lives in Boston" / "no longer lives in Boston");
* ``functional`` — a predicate in the frozen ``FUNCTIONAL`` groups with two
  different value-sized objects, both positive ("lives in Boston" / "moved
  to Seattle");
* ``numeric`` — two objects that carry different numbers on the same
  residue, both positive ("2 of Emma's recipes" / "3 of Emma's recipes").

The ordering (D6, SPEC write path step 4): one *effective time* per fact —
``valid_time`` when the fact is dated, else the session date of
``created_at`` — later wins; then the later session date, the later raw
session timestamp, the user's own span over the assistant's, the later id.
A standing fact is an assertion current as of its session, which is what
lets the mixed dated-vs-standing case use the same key.
"""

from __future__ import annotations

import re

from ..extract.resolver import anchor_date
from ..models import KIND_FACT, MemoryRecord
from .normalize import (
    MAX_VALUE_TOKENS,
    NormalizedTriple,
    normalize_triple,
    numeric_signature,
    value_tokens,
)

RULES = ("negation", "functional", "numeric")

_VALID_TIME_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def valid_time_key(valid_time: str | None) -> tuple[int, int, int] | None:
    """``"2023"`` → (2023, 0, 0); ``"2023-05"`` → (2023, 5, 0); ``"2023-05-20"`` → (2023, 5, 20)."""
    if not valid_time:
        return None
    m = _VALID_TIME_RE.match(valid_time.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0)


def created_at_key(created_at: str | None) -> tuple[int, int, int]:
    """The session date of a ``created_at`` timestamp; ``(0, 0, 0)`` when unparseable."""
    d = anchor_date(created_at)
    return (d.year, d.month, d.day) if d is not None else (0, 0, 0)


def effective_time_key(record: MemoryRecord) -> tuple[int, int, int]:
    """When the fact is current as of: its ``valid_time``, else its session date."""
    return valid_time_key(record.valid_time) or created_at_key(record.created_at)


def trust(record: MemoryRecord) -> int:
    """1 when the user said it (``raw`` starts with ``user:``), 0 for the assistant's span."""
    return 1 if (record.raw or "").startswith("user:") else 0


def order_key(record: MemoryRecord) -> tuple:
    return (
        effective_time_key(record),
        created_at_key(record.created_at),
        record.created_at or "",
        trust(record),
        record.id or 0,
    )


def beats(a: MemoryRecord, b: MemoryRecord) -> bool:
    """Whether ``a`` stays active over ``b`` when the two conflict."""
    return order_key(a) > order_key(b)


def _triple(record: MemoryRecord) -> NormalizedTriple | None:
    return normalize_triple(record.subject, record.predicate, record.object)


def conflict_between(a: MemoryRecord, b: MemoryRecord) -> str | None:
    """The rule under which ``a`` and ``b`` conflict, or ``None``."""
    if a.kind != KIND_FACT or b.kind != KIND_FACT:
        return None
    if not a.pair_key or a.pair_key != b.pair_key:
        return None
    ta, tb = _triple(a), _triple(b)
    if ta is None or tb is None or ta.subject == "assistant":
        return None
    if ta.object == tb.object:
        return "negation" if ta.polarity != tb.polarity else None
    if ta.polarity < 0 or tb.polarity < 0:
        return None  # a negative fact asserts no value
    if value_tokens(ta.object) > MAX_VALUE_TOKENS or value_tokens(tb.object) > MAX_VALUE_TOKENS:
        return None
    if ta.functional is not None:
        return "functional"
    na, nb = numeric_signature(ta.object), numeric_signature(tb.object)
    if na is not None and nb is not None and na[0] != nb[0] and na[1] == nb[1]:
        return "numeric"
    return None


def reason(rule: str, loser: MemoryRecord, winner: MemoryRecord) -> str:
    """The logged reason: ``"functional user|lives in: boston -> seattle"``."""
    lo, wi = _triple(loser), _triple(winner)
    lo_obj = lo.object if lo is not None else (loser.object or "")
    wi_obj = wi.object if wi is not None else (winner.object or "")
    return f"{rule} {winner.pair_key or loser.pair_key}: {lo_obj} -> {wi_obj}"
