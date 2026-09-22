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


def _score(run_dir: Path) -> str | None:
    import json

    path = run_dir / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("score")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
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
