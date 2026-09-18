"""``export()`` — the human-readable store dump (SPEC §Human-readable memory, PHASE5 D11).

One format, no options. The rules, each pinned by a test in ``tests/test_export.py``:

- **Rounds render through the one shared renderer.** This module never formats a
  round itself; it is handed ``render`` (``mnimi.memory.render_records``) and uses
  its bytes. A second round format would be a second renderer, and cross-arm
  comparability requires exactly one.
- **Nothing is hidden.** A superseded fact (``salience = 0``) is shown and marked.
  A dump whose omissions are invisible is not an audit trail.
- **The original fact is recoverable without a reverse lookup** (SPEC): each fact
  line carries its text, its salience, and — when set — the resolved ``valid_time``,
  its ``pair_key`` and what it superseded.
- **No clock.** All time is logical: the header's "now" is ``now_logical``, the
  latest dated session timestamp in the user's own store.
- **No embedder, no model, no query.** ``export`` is a read over stored rows, which
  is why no eval arm can reach it and it cannot move a benchmark number.
"""

from __future__ import annotations

from .models import KIND_FACT, KIND_ROUND, MemoryRecord

#: The dump's first line; a test pins it.
HEADER = "# memory export"

#: What the header prints for ``now_logical`` when no record carries a parseable date.
NO_NOW = "-"

_INDENT = "  "


def _fact_line(record: MemoryRecord) -> str:
    """One fact, with everything needed to recover it without a reverse lookup."""
    parts = [f"fact: {record.fact or record.content}"]
    parts.append(f"salience {record.salience:.4f}")
    initial = record.initial_salience
    if initial is not None and abs(initial - record.salience) > 1e-9:
        parts.append(f"(initial {initial:.4f})")
    if record.salience == 0:
        parts.append("superseded")
    if record.valid_time:
        parts.append(f"valid_time {record.valid_time}")
    if record.pair_key:
        parts.append(f"pair {record.pair_key}")
    if record.supersedes is not None:
        parts.append(f"supersedes {record.supersedes}")
    return _INDENT + "  ".join(parts)


def _ordered(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Oldest first, ties by insertion order — ``_time_ordered``'s rule, kept local.

    Session timestamps are zero-padded date-first, so lexicographic order is
    chronological without parsing a format the caller owns.
    """
    return sorted(records, key=lambda r: (r.created_at or "", r.id or 0))


def export_store(
    user_id: str,
    records: list[MemoryRecord],
    now_logical: str | None,
    render,
) -> str:
    """Render a user's whole store as text. ``render`` is ``render_records``."""
    rounds = [r for r in records if r.kind == KIND_ROUND]
    facts = [r for r in records if r.kind == KIND_FACT]

    by_round: dict[str | None, list[MemoryRecord]] = {}
    for fact in facts:
        by_round.setdefault(fact.round_key, []).append(fact)

    lines = [
        HEADER,
        f"user_id: {user_id}",
        f"now_logical: {now_logical or NO_NOW}",
        f"records: {len(records)} (rounds {len(rounds)}, facts {len(facts)})",
        "",
    ]

    for record in _ordered(rounds):
        lines.append(render([record]))
        for fact in by_round.pop(record.round_key, []) if record.round_key else []:
            lines.append(_fact_line(fact))
        lines.append("")

    # A fact whose round is gone cannot be silently dropped: it is still evidence.
    orphans = [f for group in by_round.values() for f in group]
    if orphans:
        lines.append("## facts with no round record")
        for fact in _ordered(orphans):
            lines.append(_fact_line(fact))
        lines.append("")

    return "\n".join(lines)
