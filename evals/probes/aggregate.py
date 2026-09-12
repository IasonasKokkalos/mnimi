"""Tables from a probe file: recall@k (ANY / ALL), per category, dedup drops."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from .retrieval import QuestionProbe, read_rows

KS = (1, 5, 10, 20, 50)


def _hit(ranks: list[int], k: int, require_all: bool) -> bool:
    inside = [0 < x <= k for x in ranks]
    return all(inside) if require_all else any(inside)


def summarize(rows: list[QuestionProbe]) -> dict:
    answerable = [r for r in rows if not r.is_abstention and r.n_evidence_rounds > 0]
    any_at = {k: sum(1 for r in answerable if _hit(r.evidence_ranks, k, False)) for k in KS}
    all_at = {k: sum(1 for r in answerable if _hit(r.evidence_ranks, k, True)) for k in KS}
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "any10": 0, "all10": 0})
    for r in answerable:
        c = by_cat[r.category]
        c["n"] += 1
        c["any10"] += _hit(r.evidence_ranks, 10, False)
        c["all10"] += _hit(r.evidence_ranks, 10, True)
    drops = Counter(d.screen for r in rows for d in r.drops)
    evidence_lost = [(r.question_id, lost[2]) for r in rows for lost in r.lost_evidence]
    cross_session = sum(
        1
        for r in rows
        for d in r.drops
        if d.screen == "cosine" and d.neighbour_session_id not in (None, d.session_id)
    )
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "any_at": any_at,
        "all_at": all_at,
        "by_category": dict(by_cat),
        "drops": {
            "exact": drops.get("exact", 0),
            "cosine": drops.get("cosine", 0),
            "evidence_lost": len(evidence_lost),
        },
        "cosine_drops_cross_session": cross_session,
        "evidence_lost_rows": evidence_lost,
        "rounds": sum(r.n_rounds for r in rows),
        "stored": sum(r.stored for r in rows),
    }


def format_summary(s: dict) -> str:
    n = s["n_answerable"]
    lines = [
        f"questions {s['n']} (answerable {n}); rounds {s['rounds']}, stored {s['stored']}",
        "k     " + "".join(f"{k:>10}" for k in KS),
        "ANY@k " + "".join(f"{s['any_at'][k]:>6}/{n:<3}" for k in KS),
        "ALL@k " + "".join(f"{s['all_at'][k]:>6}/{n:<3}" for k in KS),
        f"ANY@10 {s['any_at'][10]}/{n}   ALL@10 {s['all_at'][10]}/{n}",
    ]
    for cat, c in sorted(s["by_category"].items()):
        lines.append(f"  {cat:28s} n={c['n']:<3} ANY@10 {c['any10']:<3} ALL@10 {c['all10']}")
    d = s["drops"]
    lines.append(
        f"drops: exact {d['exact']}, cosine {d['cosine']} "
        f"({s['cosine_drops_cross_session']} cross-session), "
        f"evidence rounds lost {d['evidence_lost']}"
    )
    for qid, screen in s["evidence_lost_rows"]:
        lines.append(f"  evidence lost: {qid} ({screen})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m evals.probes.aggregate <probe.json>", file=sys.stderr)
        return 2
    print(format_summary(summarize(read_rows(Path(argv[0])))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
