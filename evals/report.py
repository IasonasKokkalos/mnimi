"""Aggregate and print the score table: system x overall x per-category."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .dataset import CATEGORIES
from .runner import Result


def aggregate(results: Iterable[Result]) -> tuple[dict[str, list[int]], list[int]]:
    """Return ``({category: [correct, total]}, [correct, total])``."""
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for r in results:
        by_cat[r.category][0] += int(r.correct)
        by_cat[r.category][1] += 1
        overall[0] += int(r.correct)
        overall[1] += 1
    return by_cat, overall


def _acc(correct: int, total: int) -> str:
    return f"{correct / total:6.1%}" if total else "     —"


def truncation_caveat(results: list[Result]) -> str | None:
    """Describe how much history the reader never saw, or ``None`` if it saw all.

    A system whose context is cut to fit the reader's window is not the baseline
    its name claims. ``full_history`` in particular stops being a ceiling and
    becomes "the most recent N tokens" — reporting it unqualified overstates the
    ceiling every other system is measured against.
    """
    truncated = [r for r in results if r.truncated]
    if not truncated:
        return None
    fed = [r.reader_prompt_tokens for r in results if r.reader_prompt_tokens is not None]
    mean_fed = round(sum(fed) / len(fed)) if fed else None
    dropped = sum(r.tokens_dropped for r in truncated)
    total = sum(r.tokens_dropped for r in truncated) + sum(fed or [0])
    pct = f"~{100 * dropped / total:.0f}%" if total else "?"
    fed_txt = f"{mean_fed:,}" if mean_fed is not None else "?"
    return (
        f"TRUNCATED on {len(truncated)}/{len(results)} questions: mean {fed_txt} "
        f"prompt tokens fed, ~{dropped:,} dropped ({pct} of history). "
        f"This row is 'most recent ~{fed_txt} tokens', NOT full history."
    )


def format_table(system_name: str, results: list[Result]) -> str:
    by_cat, overall = aggregate(results)
    # Known categories first (stable order), then any unexpected ones.
    cats = [c for c in CATEGORIES if c in by_cat]
    cats += [c for c in by_cat if c not in CATEGORIES]

    caveat = truncation_caveat(results)
    abstentions = sum(1 for r in results if r.is_abstention)

    width = max([len("overall"), *(len(c) for c in cats)]) + 2
    # The caveat is appended to every category row, not just the footer: a row
    # copied out of this table on its own must still carry the qualifier.
    mark = " *" if caveat else ""
    lines = [
        f"System: {system_name}",
        f"{'category'.ljust(width)}  {'acc':>7}  {'n':>5}",
        f"{'-' * width}  {'-' * 7}  {'-' * 5}",
    ]
    for c in cats:
        correct, total = by_cat[c]
        lines.append(f"{c.ljust(width)}  {_acc(correct, total):>7}  {total:>5}{mark}")
    lines.append(f"{'-' * width}  {'-' * 7}  {'-' * 5}")
    lines.append(f"{'overall'.ljust(width)}  {_acc(*overall):>7}  {overall[1]:>5}{mark}")
    lines.append(f"abstention questions in slice: {abstentions}/{len(results)}")
    if caveat:
        lines.append(f"* {caveat}")
    return "\n".join(lines)


def print_report(system_name: str, results: list[Result]) -> None:
    print(format_table(system_name, results))
