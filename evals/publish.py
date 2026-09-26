"""``python -m evals.publish <run_dir>`` — promote a run to ``results/published/``.

The promotion gate of the run-documentation rule (2026-09-22): only a
clean-tree run with a complete ``manifest.json`` may be copied. The copy is
the five documents a published number needs — ``pins.json``,
``predictions.jsonl``, ``results.json``, ``summary.json``, ``manifest.json``
(plus ``reader_resolved.json`` and any judge replays when present) — and the
registry row's status becomes ``published``. Nothing is ever deleted or
overwritten: an existing published directory refuses.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from . import manifest as manifest_mod

PUBLISHED_DIR = Path("results") / "published"
PROMOTED_FILES = (
    "pins.json", "predictions.jsonl", "results.json", "summary.json",
    manifest_mod.MANIFEST_FILE, "reader_resolved.json",
)


def promote(run_dir: Path, published_dir: Path = PUBLISHED_DIR, name: str | None = None) -> Path:
    """Copy a run into ``published_dir/<name>``; raise with the reasons when it may not be."""
    run_dir = Path(run_dir)
    m = manifest_mod.read_optional(run_dir)
    reasons = manifest_mod.promotable(m)
    if not (run_dir / "results.json").exists():
        reasons.append("no results.json (the run was never judged)")
    if reasons:
        raise ValueError(f"{run_dir} may not be promoted: " + "; ".join(reasons))
    target = Path(published_dir) / (name or run_dir.name)
    if target.exists():
        raise FileExistsError(f"{target} exists; a published artifact is never overwritten")
    target.mkdir(parents=True)
    for filename in PROMOTED_FILES:
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, target / filename)
    for replay in sorted(run_dir.glob("judge_replay_*.json")):
        shutil.copy2(replay, target / replay.name)
    m["status"] = "published"
    m["published_as"] = str(target)
    manifest_mod.write(run_dir, m)
    manifest_mod.write(target, m)
    try:
        manifest_mod.index_update(run_dir.name, status="published", runs_dir=run_dir.parent)
    except KeyError:
        row = manifest_mod.index_row(m, _score(run_dir))
        manifest_mod.index_append(row, runs_dir=run_dir.parent)
    return target


def promote_replay(
    run_dir: Path, replay_file: str, published_dir: Path = PUBLISHED_DIR, name: str | None = None,
) -> Path:
    """Copy one judge replay beside its published run and record it there (Phase 7 D6).

    The published predictions, results, summary and pins are never touched: the
    replay is its own file, and the published manifest gains the entry the run's
    manifest carries. Refused when the run was not published, when the file is
    already there, when the replay graded other pins, or when the run's
    predictions no longer match the published ones.
    """
    from . import drift

    run_dir = Path(run_dir)
    source = run_dir / replay_file
    target_dir = Path(published_dir) / (name or run_dir.name)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    if not (target_dir / "results.json").exists():
        raise ValueError(f"{target_dir} is not a published run; promote the run first")
    target = target_dir / replay_file
    if target.exists():
        raise FileExistsError(f"{target} exists; a published file is never overwritten")
    replay = json.loads(source.read_text(encoding="utf-8"))
    published = json.loads((target_dir / "results.json").read_text(encoding="utf-8"))
    if replay.get("pins_hash") != published.get("pins_hash"):
        raise ValueError(f"{replay_file} graded pins {replay.get('pins_hash')}, "
                         f"not the published {published.get('pins_hash')}")
    report = drift.compare(target_dir, run_dir)
    if report["changed"]:
        raise ValueError(
            f"{run_dir} predictions differ from the published ones on {report['changed']} rows"
        )
    replays = (manifest_mod.read_optional(run_dir) or {}).get("judge", {}).get("replays", [])
    entry = next((r for r in replays if r.get("file") == replay_file), None)
    if entry is None:
        raise ValueError(f"{run_dir / manifest_mod.MANIFEST_FILE} records no replay {replay_file}")
    published_manifest = manifest_mod.read_optional(target_dir)
    if published_manifest is None:
        raise ValueError(f"{target_dir} has no manifest.json")
    shutil.copy2(source, target)
    manifest_mod.write(target_dir, manifest_mod.record_replay(published_manifest, entry))
    return target


def _score(run_dir: Path) -> str | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("score")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--replay"]:
        if len(argv) not in (3, 4):
            print("usage: python -m evals.publish --replay <run_dir> <judge_replay_N.json> "
                  "[<published name>]", file=sys.stderr)
            return 2
        try:
            target = promote_replay(
                Path(argv[1]), argv[2], name=argv[3] if len(argv) == 4 else None
            )
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"published {argv[1]}/{argv[2]} -> {target}", file=sys.stderr)
        return 0
    if not argv or len(argv) > 2:
        print("usage: python -m evals.publish <run_dir> [<published name>]", file=sys.stderr)
        return 2
    try:
        target = promote(Path(argv[0]), name=argv[1] if len(argv) == 2 else None)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"promoted {argv[0]} -> {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
