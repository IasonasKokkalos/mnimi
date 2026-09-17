"""Row identity between two probe files: did a change move retrieval at all?

``python -m evals.probes.identity A.json B.json`` prints ``identical N/M`` over
each question's evidence ranks, top-50 sessions, dedup drops and stored count,
then the ids that differ. It is the deterministic half of a "nothing moved"
claim (PHASE2 Task 6 read it by hand; PHASE3 gates read it here): the read
path is untouched in Phase 3, so a supersession cannot change a rank, and the
no-extractor path is untouched by the screens, so the v1 probe cannot move.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from .retrieval import QuestionProbe, read_rows

COMPARED = ("evidence_ranks", "top_sessions", "drops", "stored")


def _view(row: QuestionProbe) -> dict:
    return {key: asdict(row)[key] for key in COMPARED}


def compare(a: list[QuestionProbe], b: list[QuestionProbe]) -> tuple[list[str], list[str]]:
    """``(identical ids, differing ids)`` over the questions both files hold, in A's order."""
    b_by_id = {row.question_id: row for row in b}
    identical, differing = [], []
    for row in a:
        other = b_by_id.get(row.question_id)
        if other is None:
            differing.append(row.question_id)
        elif _view(row) == _view(other):
            identical.append(row.question_id)
        else:
            differing.append(row.question_id)
    return identical, differing


def evidence_moves(a: list[QuestionProbe], b: list[QuestionProbe], k: int = 10) -> dict:
    """Evidence rounds that crossed the top-k boundary from A to B (PHASE4 gate 4-ii).

    ``{"left": [...], "entered": [...]}``, each item ``(question_id, evidence index,
    rank in A, rank in B)`` with -1 for "not in the top-50".
    """
    b_by_id = {row.question_id: row for row in b}
    left, entered = [], []
    for row in a:
        other = b_by_id.get(row.question_id)
        if other is None:
            continue
        for index, (rank_a, rank_b) in enumerate(
            zip(row.evidence_ranks, other.evidence_ranks, strict=True)
        ):
            inside_a, inside_b = 0 < rank_a <= k, 0 < rank_b <= k
            if inside_a and not inside_b:
                left.append((row.question_id, index, rank_a, rank_b))
            elif inside_b and not inside_a:
                entered.append((row.question_id, index, rank_a, rank_b))
    return {"left": left, "entered": entered}


def format_report(a: list[QuestionProbe], b: list[QuestionProbe]) -> str:
    identical, differing = compare(a, b)
    lines = [f"identical {len(identical)}/{len(a)} (on {', '.join(COMPARED)})"]
    if len(b) != len(a):
        lines.append(f"row counts differ: {len(a)} vs {len(b)}")
    for qid in differing:
        lines.append(f"  differs: {qid}")
    moves = evidence_moves(a, b)
    lines.append(f"evidence rounds leaving the top-10: {len(moves['left'])}")
    for qid, index, rank_a, rank_b in moves["left"]:
        lines.append(f"  left: {qid} evidence #{index} rank {rank_a} -> {rank_b}")
    lines.append(f"evidence rounds entering the top-10: {len(moves['entered'])}")
    for qid, index, rank_a, rank_b in moves["entered"]:
        lines.append(f"  entered: {qid} evidence #{index} rank {rank_a} -> {rank_b}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m evals.probes.identity <a.json> <b.json>", file=sys.stderr)
        return 2
    a, b = read_rows(Path(argv[0])), read_rows(Path(argv[1]))
    print(format_report(a, b))
    return 0 if not compare(a, b)[1] and len(a) == len(b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
