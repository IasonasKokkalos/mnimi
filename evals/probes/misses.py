"""Classify a run's wrong rows: retrieval miss or reading miss (Phase 1, task 1.5).

Joins a probe file (``evals.probes.retrieval``) with the run's ``results.json``
under the same configuration. A wrong row is a *retrieval-miss* when no
annotated evidence round sits inside the run's top-k (``k`` from the pins) —
the reader never saw the evidence — and a *reading-miss* otherwise. The paper
found reading errors are 40–50% of all errors even with correct retrieval
(§E.5); which half a system's misses fall in decides where the next hour goes:
R7 (hybrid FTS5) and R8 (bge-base) fire only on retrieval misses.

Optional ``--oracle`` / ``--naive`` results files add, per row, whether those
arms got it right — a reading-miss oracle also missed is a question the reader
cannot do; a retrieval-miss naive_rag got right is a retrieval-side difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .retrieval import read_rows

RETRIEVAL_MISS = "retrieval-miss"
READING_MISS = "reading-miss"


def _correctness(results_path: Path) -> tuple[dict[str, bool], int | None]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = {r["question_id"]: bool(r["correct"]) for r in payload["results"]}
    return rows, payload.get("pins", {}).get("k")


def classify(probe_rows, correct: dict[str, bool], k: int) -> list[dict]:
    """One entry per wrong row of ``correct`` that the probe covers."""
    by_id = {r.question_id: r for r in probe_rows}
    out = []
    for qid, ok in correct.items():
        if ok or qid not in by_id:
            continue
        r = by_id[qid]
        in_k = [x for x in r.evidence_ranks if 0 < x <= k]
        kind = READING_MISS if in_k else RETRIEVAL_MISS
        if r.is_abstention or r.n_evidence_rounds == 0:
            kind = READING_MISS  # nothing to retrieve: the reader had to abstain
        out.append({
            "question_id": qid,
            "category": r.category,
            "kind": kind,
            "evidence_ranks": r.evidence_ranks,
            "evidence_in_top_k": len(in_k),
            "lost_evidence": r.lost_evidence,
        })
    return out


def annotate(entries: list[dict], name: str, correct: dict[str, bool]) -> None:
    for e in entries:
        e[f"{name}_correct"] = correct.get(e["question_id"])


def format_table(entries: list[dict], k: int, extra: list[str]) -> str:
    kinds = Counter(e["kind"] for e in entries)
    lines = [
        f"wrong rows {len(entries)}: {kinds.get(RETRIEVAL_MISS, 0)} {RETRIEVAL_MISS} "
        f"(no evidence round in the top-{k}), {kinds.get(READING_MISS, 0)} {READING_MISS}"
    ]
    for name in extra:
        got = Counter((e["kind"], bool(e.get(f"{name}_correct"))) for e in entries)
        lines.append(
            f"  {name} right on: {got[(RETRIEVAL_MISS, True)]} of the retrieval misses, "
            f"{got[(READING_MISS, True)]} of the reading misses"
        )
    by_cat = Counter((e["category"], e["kind"]) for e in entries)
    for cat in sorted({e["category"] for e in entries}):
        lines.append(
            f"  {cat:28s} retrieval {by_cat[(cat, RETRIEVAL_MISS)]:<3} "
            f"reading {by_cat[(cat, READING_MISS)]}"
        )
    for e in entries:
        flags = " ".join(
            f"{name}={'✓' if e.get(f'{name}_correct') else '✗'}" for name in extra
        )
        lines.append(
            f"  {e['question_id']:22s} {e['category']:28s} {e['kind']:15s} "
            f"ranks={e['evidence_ranks']} {flags}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.probes.misses")
    parser.add_argument("probe", help="probe.json from evals.probes.retrieval")
    parser.add_argument("results", help="the run's results.json (same configuration)")
    parser.add_argument("--oracle", default=None, help="oracle results.json")
    parser.add_argument("--naive", default=None, help="naive_rag results.json")
    parser.add_argument("--k", type=int, default=None, help="override the pinned k")
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.probe))
    correct, pinned_k = _correctness(Path(args.results))
    k = args.k if args.k is not None else pinned_k
    if k is None:
        print("ERROR: no k in the results pins; pass --k", file=sys.stderr)
        return 2
    entries = classify(rows, correct, k)
    extra = []
    for name, path in (("oracle", args.oracle), ("naive", args.naive)):
        if path:
            annotate(entries, name, _correctness(Path(path))[0])
            extra.append(name)
    print(format_table(entries, k, extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
