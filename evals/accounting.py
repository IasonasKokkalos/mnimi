"""C2 and C1 over committed files (PHASE8 D2; ``python -m evals.accounting``).

- ``retest``: judge test-retest for one run — the first verdicts against every
  ``judge_replay_*.json`` graded by the same judge. A row is **judge-unstable**
  when any of its gradings differs from the others.
- ``buckets``: the oracle-paired accounting (paper docs/PLAN.md §1.1) over the
  rows that are stable under both the system and the oracle: retrieval-bound
  (oracle right, system wrong), both wrong, system right / oracle wrong, both
  right — per category and overall.

Nothing here re-judges; every number comes from files under ``results/published``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stats

RETEST_SCHEMA = "mnimi-judge-retest/1"
ACCOUNTING_SCHEMA = "mnimi-accounting/1"
BUCKETS = ("both_right", "oracle_right_system_wrong", "both_wrong", "system_right_oracle_wrong")


def _replays(run_dir: Path, judge_model: str) -> list[dict[str, bool]]:
    """The run's replays under ``judge_model`` — each a fresh grading, or refused.

    A replay that read the verdict cache reproduces the first grading by
    construction (PHASE8 D2 needs ``--judge-cache off``); one under another
    judge prompt measures a different instrument. Both are refused by name.
    """
    out = []
    first = json.loads((run_dir / "results.json").read_text(encoding="utf-8")).get("judge") or {}
    for path in sorted(run_dir.glob("judge_replay_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        judge = payload.get("judge") or {}
        if judge.get("judge_model") != judge_model:
            continue
        run = payload.get("run") or {}
        if run.get("judge_cache") != "off" or (run.get("judge_cache_hits") or 0) > 0:
            raise ValueError(
                f"{path.name}: a retest replay must be graded with --judge-cache off and no "
                f"cache hit (found judge_cache={run.get('judge_cache')!r}, "
                f"hits={run.get('judge_cache_hits')!r})"
            )
        first_hash, this_hash = first.get("judge_prompt_hash"), judge.get("judge_prompt_hash")
        if first_hash and this_hash and first_hash != this_hash:
            raise ValueError(f"{path.name}: judge_prompt_hash differs from the first grading's")
        out.append({r["question_id"]: bool(r["correct"]) for r in payload["results"]})
    return out


def retest(run_dir: str | Path, judge_model: str) -> dict:
    """The first verdicts of ``run_dir`` against every replay under ``judge_model``."""
    run_dir = Path(run_dir)
    first = stats.load_correctness(run_dir)
    replays = _replays(run_dir, judge_model)
    if not replays:
        raise ValueError(f"{run_dir}: no judge_replay of {judge_model}")
    for i, replay in enumerate(replays, start=1):
        if set(replay) != set(first):
            raise ValueError(f"{run_dir}: replay {i} graded a different question set")
    unstable = sorted(q for q in first if len({first[q], *[r[q] for r in replays]}) > 1)
    to_wrong = sum(1 for r in replays for q in first if first[q] and not r[q])
    to_right = sum(1 for r in replays for q in first if not first[q] and r[q])
    ci = stats.wilson(len(unstable), len(first))
    return {
        "analysis_schema": RETEST_SCHEMA,
        "run_id": run_dir.name,
        "judge_model": judge_model,
        "replays": len(replays),
        "n": len(first),
        "unstable_ids": unstable,
        "unstable": len(unstable),
        "rate": ci.point, "rate_low": ci.low, "rate_high": ci.high,
        "score_by_grading": [sum(first.values()), *[sum(r.values()) for r in replays]],
        "flips_to_wrong": to_wrong,
        "flips_to_right": to_right,
    }


def _categories(run_dir: Path) -> dict[str, str]:
    out = {}
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            category = "abstention" if row.get("is_abstention") else row.get("category")
            out[row["question_id"]] = category
    return out


def buckets(system_dir: str | Path, oracle_dir: str | Path, unstable_ids: set[str]) -> dict:
    """The four buckets over the rows not in ``unstable_ids``, per category and overall."""
    system_dir, oracle_dir = Path(system_dir), Path(oracle_dir)
    system = stats.load_correctness(system_dir)
    oracle = stats.load_correctness(oracle_dir)
    if set(system) != set(oracle):
        raise ValueError(
            f"question sets differ: {len(set(system) ^ set(oracle))} ids in only one of "
            f"{system_dir.name} and {oracle_dir.name}"
        )
    categories = _categories(system_dir)

    def empty() -> dict:
        return {b: 0 for b in BUCKETS} | {"n_stable": 0, "n_unstable": 0}

    overall = empty()
    by_category: dict[str, dict] = {}
    for q in sorted(system):
        cell = by_category.setdefault(categories.get(q, "unknown"), empty())
        if q in unstable_ids:
            cell["n_unstable"] += 1
            overall["n_unstable"] += 1
            continue
        if system[q] and oracle[q]:
            bucket = "both_right"
        elif oracle[q]:
            bucket = "oracle_right_system_wrong"
        elif system[q]:
            bucket = "system_right_oracle_wrong"
        else:
            bucket = "both_wrong"
        for target in (cell, overall):
            target[bucket] += 1
            target["n_stable"] += 1
    return {
        "analysis_schema": ACCOUNTING_SCHEMA,
        "system": system_dir.name,
        "oracle": oracle_dir.name,
        "unstable_ids": sorted(unstable_ids),
        "overall": overall,
        "by_category": by_category,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.accounting")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("retest")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--judge", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("buckets")
    p.add_argument("--system", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--retest", required=True, help="retest.json; the union of the system's and "
                   "the oracle's unstable ids is removed")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "retest":
            out = Path(args.out) / "retest.json"
            if out.exists():
                raise FileExistsError(f"{out} exists; never overwritten")
            arms = {Path(d).name: retest(d, args.judge) for d in args.run_dirs}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"analysis_schema": RETEST_SCHEMA, "judge_model": args.judge,
                                       "arms": arms}, indent=2, sort_keys=True), encoding="utf-8")
            for name, r in arms.items():
                print(f"{name}: unstable {r['unstable']}/{r['n']} "
                      f"[{r['rate_low']:.3f}, {r['rate_high']:.3f}]; "
                      f"scores {r['score_by_grading']}; "
                      f"flips to wrong {r['flips_to_wrong']}, to right {r['flips_to_right']}",
                      file=sys.stderr)
            print(f"wrote {out}", file=sys.stderr)
        else:
            out = Path(args.out) / "buckets.json"
            if out.exists():
                raise FileExistsError(f"{out} exists; never overwritten")
            retest_payload = json.loads(Path(args.retest).read_text(encoding="utf-8"))
            arms = retest_payload["arms"]
            unstable: set[str] = set()
            for name in (Path(args.system).name, Path(args.oracle).name):
                if name not in arms:
                    raise ValueError(f"{args.retest} has no retest of {name}")
                unstable |= set(arms[name]["unstable_ids"])
            result = buckets(args.system, args.oracle, unstable)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(f"overall: {result['overall']}", file=sys.stderr)
            print(f"wrote {out}", file=sys.stderr)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
