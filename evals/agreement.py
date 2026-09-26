"""Inter-judge agreement and the blind label sheet (Phase 7 D8; ``python -m evals.agreement``).

Committed-file readers only: the first judge's verdicts come from ``predictions.jsonl``
(else ``results.json``), the second judge's from the one ``judge_replay_*.json`` graded
by that model. Nothing here re-judges; nothing here reads a key it also writes.

- ``report``: per-arm and pooled agreement between the two judges, with Wilson
  intervals and Cohen's kappa.
- ``sheet``: a blind label sheet — ``n_disagree`` rows the judges disagree on and
  ``n_control`` rows they agree on, drawn with a seed, no verdict column; the key
  sits in a separate file that only ``score`` reads.
- ``score``: each judge's agreement with the human labels, by kind.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from . import stats
from .judge import build_judge_prompt

REPORT_SCHEMA = "mnimi-judge-agreement/1"
LABELS_SCHEMA = "mnimi-judge-labels/1"
SHEET_FIELDS = ("row_id", "question_type", "instruction", "question", "gold", "response", "human")
SHEET_FILE = "label_sheet.csv"
KEY_FILE = "label_key.json"
YES = {"yes", "y"}
NO = {"no", "n"}


def cohen_kappa(yy: int, yn: int, ny: int, nn: int) -> float:
    """Cohen's kappa from the 2x2 table (first judge yes/no x second judge yes/no)."""
    n = yy + yn + ny + nn
    if n == 0:
        return 1.0
    observed = (yy + nn) / n
    p1 = (yy + yn) / n
    p2 = (yy + ny) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(first: dict[str, bool], second: dict[str, bool]) -> dict:
    """Agreement between two verdict dicts over the same keys."""
    if set(first) != set(second):
        raise ValueError("the two judges graded different question sets")
    yy = sum(1 for q in first if first[q] and second[q])
    yn = sum(1 for q in first if first[q] and not second[q])
    ny = sum(1 for q in first if not first[q] and second[q])
    nn = sum(1 for q in first if not first[q] and not second[q])
    n = len(first)
    agree = yy + nn
    rate = stats.wilson(agree, n) if n else None
    return {
        "n": n, "yy": yy, "yn": yn, "ny": ny, "nn": nn, "agree": agree,
        "rate": rate.point if rate else None,
        "rate_low": rate.low if rate else None,
        "rate_high": rate.high if rate else None,
        "kappa": cohen_kappa(yy, yn, ny, nn),
    }


def _first_judge(run_dir: Path) -> str | None:
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return (payload.get("judge") or {}).get("judge_model")


def report(run_dirs, judge_model: str) -> dict:
    """Per-arm and pooled agreement between each run's first verdicts and its replay."""
    run_dirs = [Path(d) for d in run_dirs]
    firsts = {_first_judge(d) for d in run_dirs}
    if len(firsts) != 1:
        raise ValueError(
            f"the arms were first graded by different judges: {sorted(map(str, firsts))}"
        )
    (judge_first,) = firsts
    arms: dict[str, dict] = {}
    pooled_first: dict[str, bool] = {}
    pooled_second: dict[str, bool] = {}
    for d in run_dirs:
        first = stats.load_correctness(d)
        second = stats.load_correctness(d, judge_model)
        arms[d.name] = agreement(first, second)
        for q in first:
            pooled_first[f"{d.name}:{q}"] = first[q]
            pooled_second[f"{d.name}:{q}"] = second[q]
    return {
        "analysis_schema": REPORT_SCHEMA,
        "judge_first": judge_first,
        "judge_second": judge_model,
        "arms": arms,
        "pooled": agreement(pooled_first, pooled_second),
    }


def _rows_of(run_dir: Path) -> list[dict]:
    rows = []
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(run_dirs, judge_model: str) -> tuple[list[dict], list[dict]]:
    """Every row of every run with both verdicts, split into disagreements and agreements."""
    disagree, agree = [], []
    for d in map(Path, run_dirs):
        first = stats.load_correctness(d)
        second = stats.load_correctness(d, judge_model)
        for row in _rows_of(d):
            qid = row["question_id"]
            item = {
                "run_id": d.name, "question_id": qid, "category": row.get("category"),
                "is_abstention": bool(row.get("is_abstention")), "question": row.get("question"),
                "answer": row.get("answer"), "predicted": row.get("predicted"),
                "first": first[qid], "second": second[qid],
            }
            (disagree if first[qid] != second[qid] else agree).append(item)
    return disagree, agree


def _dedupe(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """One entry per (question_id, sha256(predicted)); run_ids merged, sorted."""
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        digest = hashlib.sha256((row["predicted"] or "").encode("utf-8")).hexdigest()
        key = (row["question_id"], digest)
        if key in out:
            out[key]["run_ids"] = sorted(set(out[key]["run_ids"]) | {row["run_id"]})
        else:
            out[key] = {**row, "run_ids": [row["run_id"]]}
    return out


def _instruction(row: dict) -> str:
    category = row.get("category") or ""
    prompt = build_judge_prompt(category, "", "", "", row["is_abstention"])
    return prompt.split("\n\nQuestion:")[0]


def draw_sheet(disagree, agree, n_disagree=50, n_control=10, seed=0) -> tuple[list[dict], dict]:
    """The blind sheet rows (no verdicts) and the key, both deterministic in ``seed``."""
    pool_d = _dedupe(disagree)
    pool_a = {k: v for k, v in _dedupe(agree).items() if k not in pool_d}
    rng = random.Random(seed)
    chosen_d = rng.sample(sorted(pool_d), min(n_disagree, len(pool_d)))
    chosen_a = rng.sample(sorted(pool_a), min(n_control, len(pool_a)))
    union = [("disagree", k) for k in chosen_d] + [("control", k) for k in chosen_a]
    rng.shuffle(union)
    rows, key = [], {}
    for i, (kind, k) in enumerate(union, start=1):
        item = pool_d[k] if kind == "disagree" else pool_a[k]
        row_id = f"L{i:02d}"
        rows.append({
            "row_id": row_id,
            "question_type": (
                "abstention" if item["is_abstention"] else (item.get("category") or "")
            ),
            "instruction": _instruction(item),
            "question": item.get("question") or "",
            "gold": item.get("answer") or "",
            "response": item.get("predicted") or "",
            "human": "",
        })
        key[row_id] = {"kind": kind, "question_id": item["question_id"], "run_ids": item["run_ids"],
                       "first": item["first"], "second": item["second"]}
    return rows, key


def write_sheet(out_dir, rows: list[dict], key: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, key_path = out_dir / SHEET_FILE, out_dir / KEY_FILE
    for path in (sheet, key_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; a label sheet is never overwritten")
    with open(sheet, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
    return sheet, key_path


def read_labels(sheet) -> dict[str, bool]:
    """The human column as booleans; a blank or non-yes/no label names its row and raises."""
    labels: dict[str, bool] = {}
    bad: list[str] = []
    with open(sheet, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("human") or "").strip().lower()
            if value in YES:
                labels[row["row_id"]] = True
            elif value in NO:
                labels[row["row_id"]] = False
            else:
                bad.append(row["row_id"])
    if bad:
        raise ValueError("unlabelled or invalid rows (yes/no expected): " + ", ".join(bad))
    return labels


def score(labels: dict[str, bool], key: dict) -> dict:
    """Each judge's agreement with the human, by kind (disagree / control)."""
    by_kind: dict[str, dict] = {}
    for kind in ("disagree", "control"):
        ids = [r for r, k in key.items() if k["kind"] == kind]
        entry: dict = {"n": len(ids)}
        for judge in ("first", "second"):
            agrees = sum(1 for r in ids if key[r][judge] == labels[r])
            side: dict = {"agrees_with_human": agrees}
            if ids:
                ci = stats.wilson(agrees, len(ids))
                side.update(rate=ci.point, rate_low=ci.low, rate_high=ci.high)
            entry[judge] = side
        by_kind[kind] = entry
    return {"analysis_schema": LABELS_SCHEMA, "n": len(labels), "by_kind": by_kind}


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.agreement")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "sheet"):
        p = sub.add_parser(name)
        p.add_argument("run_dirs", nargs="+")
        p.add_argument("--judge", required=True)
        p.add_argument("--out", required=True)
        if name == "sheet":
            p.add_argument("--disagree", type=int, default=50)
            p.add_argument("--controls", type=int, default=10)
            p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            out = Path(args.out) / "agreement.json"
            _refuse_existing(out)
            rep = report(args.run_dirs, args.judge)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
            pooled = rep["pooled"]
            print(f"wrote {out}: pooled agreement {pooled['agree']}/{pooled['n']} "
                  f"(kappa {pooled['kappa']:.3f})", file=sys.stderr)
        elif args.command == "sheet":
            disagree, agree = split_rows(args.run_dirs, args.judge)
            rows, key = draw_sheet(disagree, agree, args.disagree, args.controls, args.seed)
            sheet, key_path = write_sheet(args.out, rows, key)
            print(f"wrote {sheet} ({len(rows)} rows) and {key_path}; do not open the key "
                  "before labelling", file=sys.stderr)
        else:
            directory = Path(args.directory)
            out = directory / "label_score.json"
            _refuse_existing(out)
            labels = read_labels(directory / SHEET_FILE)
            key = json.loads((directory / KEY_FILE).read_text(encoding="utf-8"))
            result = score(labels, key)
            out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote {out}", file=sys.stderr)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
