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


def format_table(system_name: str, results: list[Result]) -> str:
    by_cat, overall = aggregate(results)
    # Known categories first (stable order), then any unexpected ones.
    cats = [c for c in CATEGORIES if c in by_cat]
    cats += [c for c in by_cat if c not in CATEGORIES]

    width = max([len("overall"), *(len(c) for c in cats)]) + 2
    lines = [
        f"System: {system_name}",
        f"{'category'.ljust(width)}  {'acc':>7}  {'n':>5}",
        f"{'-' * width}  {'-' * 7}  {'-' * 5}",
    ]
    for c in cats:
        correct, total = by_cat[c]
        lines.append(f"{c.ljust(width)}  {_acc(correct, total):>7}  {total:>5}")
    lines.append(f"{'-' * width}  {'-' * 7}  {'-' * 5}")
    lines.append(f"{'overall'.ljust(width)}  {_acc(*overall):>7}  {overall[1]:>5}")
    return "\n".join(lines)


def print_report(system_name: str, results: list[Result]) -> None:
    print(format_table(system_name, results))
