"""``python -m evals.backfill <run_dir> --purpose ... [--set a.b=v] [--also DIR]``.

A manifest for a run that predates the run-documentation rule (2026-09-22),
built from what its directory holds: ``results.json`` (pins, run block,
judge), ``reader_resolved.json``, ``predictions.jsonl`` (the question ids)
and the API ledger line booked under the run directory (the judge's cost and
tokens). Every required field the artifacts cannot supply is written
``UNKNOWN`` — never a guess. ``--set`` carries a value recovered from a
written record (a results document, a run log); the record is named in the
manifest's ``backfilled.sources`` so the value stays auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import artifacts
from . import manifest as manifest_mod
from .manifest import REQUIRED_FIELDS, UNKNOWN, _get


def backfill(
    run_dir: Path,
    *,
    purpose: str,
    claim: str = "none",
    rule_commit: str | None = None,
    ledger: Path | None = None,
    overrides: dict | None = None,
    sources: str | None = None,
) -> dict:
    run_dir = Path(run_dir)
    overrides = dict(overrides or {})
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    pins, run = payload["pins"], payload.get("run", {})
    resolved = artifacts.read_reader_resolved_optional(run_dir) or {}
    question_ids = [
        json.loads(line)["question_id"]
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judge_cost: dict = {}
    if ledger is not None and Path(ledger).exists():
        for line in Path(ledger).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            booked = entry.get("run_dir")
            if booked and Path(booked).name == run_dir.name and (
                entry.get("judge_actual_usd") is not None
            ):
                judge_cost = entry
    usage = resolved.get("usage") or {}
    served = resolved.get("served_models") or None
    m = manifest_mod.build(
        run_id=run_dir.name,
        purpose=purpose,
        claim=claim,
        rule_commit=rule_commit,
        pins=pins,
        porcelain=None if str(pins.get("harness_git_sha", "")).endswith("-dirty") else "",
        served_models=served or {UNKNOWN: len(question_ids)},
        served_fingerprints=resolved.get("system_fingerprints") or None,
        judge=payload.get("judge"),
        replay_count=len(list(run_dir.glob("judge_replay_*.json"))),
        question_ids=question_ids,
        environment=run.get("environment"),
        cost={
            "reader_usd": resolved.get("actual_usd"),
            "judge_usd": judge_cost.get("judge_actual_usd"),
            "reader_prompt_tokens": usage.get("prompt_tokens"),
            "reader_completion_tokens": usage.get("completion_tokens"),
            "judge_prompt_tokens": judge_cost.get("judge_prompt_tokens"),
            "judge_completion_tokens": judge_cost.get("judge_completion_tokens"),
            "judge_wall_s": run.get("elapsed_s"),
            "llm_calls_at_write": None,
        },
        outputs={
            "predictions": "predictions.jsonl",
            "results": "results.json",
            "pins": "pins.json",
            "reader_resolved": "reader_resolved.json" if resolved else None,
        },
    )
    # The mnimi version OF THE RUN is the one in pyproject.toml at the run's
    # commit, not today's: recovered from git, UNKNOWN when git cannot show it.
    commit = m["code"].get("commit")
    shown = artifacts._git("show", f"{commit}:pyproject.toml") if commit else None
    m["code"]["mnimi_version"] = UNKNOWN
    if shown:
        import re

        found = re.search(r'^version\s*=\s*"([^"]+)"', shown, re.M)
        if found:
            m["code"]["mnimi_version"] = found.group(1)
    state = artifacts.read_batch_state_optional(run_dir) or {}
    if state.get("submitted_at"):
        # The one timestamp the artifacts hold: the Batch submission, local
        # time as the harness wrote it. Named as such rather than passed off
        # as UTC.
        m["created_utc"] = f"{state['submitted_at']} (local, batch_state.json submitted_at)"
    m["backfilled"] = {
        "at": manifest_mod.utc_now(),
        "from": "results.json, reader_resolved.json, predictions.jsonl, the API ledger",
        "sources": sources,
        "note": "written after the run under the run-documentation rule of 2026-09-22; "
        "UNKNOWN marks a field the artifacts do not hold and no record supplies",
    }
    # The machine running the backfill is NOT the machine that ran: only what
    # the run block recorded stands, and the rest is UNKNOWN unless a record
    # supplies it through --set.
    m["environment"]["python"] = UNKNOWN
    m["environment"]["lockfile_hash"] = UNKNOWN
    m["environment"]["lockfile_source"] = "not recorded at run time"
    for key in ("processor", "cpu_count", "platform"):
        m["environment"]["hardware"][key] = UNKNOWN
    for dotted, value in overrides.items():
        node = m
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    cost = m["cost"]

    def total(*keys):
        known = [cost.get(k) for k in keys if isinstance(cost.get(k), (int, float))]
        return round(sum(known), 1) if known else None

    if cost.get("tokens_in") is None:
        cost["tokens_in"] = total("reader_prompt_tokens", "judge_prompt_tokens")
    if cost.get("tokens_out") is None:
        cost["tokens_out"] = total("reader_completion_tokens", "judge_completion_tokens")
    if cost.get("wall_clock_s") is None:
        cost["wall_clock_s"] = total("predict_wall_s", "judge_wall_s")
    for field in REQUIRED_FIELDS:
        if _get(m, field) is None:
            node = m
            parts = field.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = UNKNOWN
    return m


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.backfill")
    parser.add_argument("run_dir")
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--claim", default="none")
    parser.add_argument("--rule-commit", default=None)
    parser.add_argument("--ledger", default=".cache/api_ledger.jsonl")
    parser.add_argument("--sources", default=None,
                        help="the written records the --set values were recovered from")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                        help="a recovered value (dotted path; JSON or text)")
    parser.add_argument("--also", default=None, metavar="DIR",
                        help="write a second copy of the manifest here (the published dir)")
    parser.add_argument("--status", default=None, choices=manifest_mod.STATUSES,
                        help="registry status to record (default: from the checks)")
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"ERROR: --set expects PATH=VALUE, got {item!r}", file=sys.stderr)
            return 2
        overrides[key.strip()] = _parse_value(value)
    run_dir = Path(args.run_dir)
    m = backfill(
        run_dir, purpose=args.purpose, claim=args.claim, rule_commit=args.rule_commit,
        ledger=Path(args.ledger), overrides=overrides, sources=args.sources,
    )
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    manifest_mod.finalize(m, payload.get("provisional") or [])
    if args.status:
        m["status"] = args.status
    manifest_mod.write(run_dir, m)
    if args.also:
        manifest_mod.write(Path(args.also), m)
    if not args.no_index:
        correct = sum(1 for r in payload["results"] if r["correct"])
        row = manifest_mod.index_row(m, f"{correct}/{len(payload['results'])}")
        manifest_mod.index_append(row, runs_dir=run_dir.parent)
    line = manifest_mod.format_missing(m)
    print(f"wrote {run_dir / manifest_mod.MANIFEST_FILE}  |  status {m['status']}"
          + (f"  |  {line}" if line else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
