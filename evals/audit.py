"""Tier 1 audit records (Phase 7 D2/D3): what a re-grade of committed predictions found.

``python -m evals --stage judge --predictions <file> --audit-out <path>`` grades the
rows again and writes one record: the recomputed score beside the published one,
and every row whose fresh verdict differs from the committed first verdict. A
cache-hit replay that differs is a defect (the predictions or the cache were
edited); a fresh re-grade that differs is the judge's instrument error at
temperature 0: reported, never "fixed". The record is written where the caller
points, never beside the artifact it audits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import artifacts

AUDIT_SCHEMA = "mnimi-tier1-audit/2"


def committed_verdicts(predictions_path: Path) -> dict[str, bool]:
    """The first verdict of every row in the file; else the ``results.json`` rows beside it."""
    predictions_path = Path(predictions_path)
    out: dict[str, bool] = {}
    with open(predictions_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdicts = row.get("verdicts") or []
            if verdicts:
                out[row["question_id"]] = bool(verdicts[0]["correct"])
    if out:
        return out
    results = predictions_path.parent / "results.json"
    if results.exists():
        rows = json.loads(results.read_text(encoding="utf-8"))["results"]
        return {row["question_id"]: bool(row["correct"]) for row in rows}
    return {}


def flips(committed: dict[str, bool], regraded: dict[str, bool]) -> dict:
    """Rows graded both times whose verdict changed, split by direction."""
    shared = sorted(set(committed) & set(regraded))
    to_correct = [q for q in shared if regraded[q] and not committed[q]]
    to_wrong = [q for q in shared if committed[q] and not regraded[q]]
    return {
        "compared": len(shared),
        "count": len(to_correct) + len(to_wrong),
        "to_correct": to_correct,
        "to_wrong": to_wrong,
    }


def verdict(record: dict) -> str:
    """MATCHES; WITHIN RE-GRADE (flips on fresh calls); DIVERGES (a defect); NO REFERENCE."""
    flipped = record["flips"]
    if record.get("published") is None and not flipped.get("compared"):
        return "NO REFERENCE"
    if flipped["count"] == 0:
        if record.get("published") in (None, record.get("recomputed")):
            return "MATCHES"
        return "DIVERGES"
    return "WITHIN RE-GRADE" if record["cache"]["misses"] > 0 else "DIVERGES"


def audit_record(
    *,
    predictions_path: Path,
    results,
    pins: dict,
    judge_info: dict,
    cache_hits: int,
    cache_misses: int,
    published: tuple[int, int] | None,
    cost: dict,
    commit: str | None,
    clean_tree: bool | None,
    graded_utc: str,
) -> dict:
    """One audit, as ``analyses/audits/`` stores it."""
    predictions_path = Path(predictions_path)
    regraded = {r.question_id: bool(r.correct) for r in results}
    record = {
        "analysis_schema": AUDIT_SCHEMA,
        "run_id": predictions_path.parent.name,
        "predictions": predictions_path.as_posix(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "pins_hash": artifacts.pins_hash(pins) if pins else None,
        "artifact_harness_git_sha": (pins or {}).get("harness_git_sha"),
        "audited_at": {"commit": commit, "clean_tree": clean_tree, "graded_utc": graded_utc},
        "judge": judge_info,
        "judge_hash": artifacts.fingerprint(artifacts.canonical(judge_info)),
        "cache": {"hits": cache_hits, "misses": cache_misses},
        # Since /2 (PHASE8 D3) an audit never reads or writes the verdict cache.
        "judge_cache": "off",
        "recomputed": {"correct": sum(regraded.values()), "n": len(regraded)},
        "published": None if published is None else {"correct": published[0], "n": published[1]},
        "flips": flips(committed_verdicts(predictions_path), regraded),
        "cost": cost,
    }
    record["verdict"] = verdict(record)
    return record


def write_record(path: Path, record: dict) -> Path:
    """Write once: an audit record is evidence and is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; an audit record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path
