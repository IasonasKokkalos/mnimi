"""``--verify-drift``: the divergence count Tier 2 actually claims.

Two predict runs of the *same* pins are compared row by row: how many
predictions changed, which ones, and the first byte at which each diverged.
That count is the reproducibility statement — "N/100 changed" — for the family
that produced the rows: 0/100 across a daemon restart for the local family
(``results/published/mnimi__100q_restart_2026-09-10``, compared by hand at
the time), and "reproducible within N/100 measured drift" for the gpt-4o
family, whose served backend (``system_fingerprint``) cannot be requested.

Before this existed the signal was a side effect: the judge verdict cache keys
on ``sha256(predicted)``, so a re-run that missed the cache proved the reader
had moved, mute about where. This promotes it to a first-class instrument
(FUTURE.md, "``--verify-drift``", trigger: the next drift pair).

Two entry points, one comparison:

* ``python -m evals ... --stage predict --verify-drift <reference>`` compares
  the fresh ``predictions.jsonl`` against ``<reference>`` (a run directory or
  a predictions file) the moment it is written — sync, batch, or a batch
  resume — and leaves ``drift.json`` beside it.
* ``python -m evals.drift <reference> <fresh>`` compares two existing
  artifacts and writes ``drift.json`` into ``<fresh>`` when it is a directory.

A drift pair is the same configuration by definition: when both sides carry a
``pins.json`` and any shared pin differs, the comparison is refused, not
softened — a changed prediction under changed pins is a configuration
difference, and reporting it as drift would launder it. Two pins are exempt
from the refusal and reported instead: ``artifact_schema`` (a bump that only
adds a pin is not a configuration change) and ``harness_git_sha`` (the
published restart pair spans three commits; a code change that moves nothing
is exactly what a drift pair is allowed to show, and one that moves rows is
named in the report so the reader does not call it drift). ``drift.json`` is
a diagnostic, never part of ``pins_hash``, and is not one of the three
promoted artifact files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import artifacts

DRIFT_FILE = "drift.json"


class DriftPairError(ValueError):
    """The two sides are not a drift pair (different pins, or disjoint rows)."""


def _predictions_path(target: Path) -> Path:
    return target if target.is_file() else target / "predictions.jsonl"


def _rows(target: Path) -> dict[str, dict]:
    path = _predictions_path(target)
    if not path.exists():
        raise FileNotFoundError(f"no predictions file at {path}")
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def _pins(target: Path) -> tuple[dict | None, str | None]:
    """``(pins, pins_hash)`` beside a predictions file; ``(None, None)`` for a bare file."""
    directory = target.parent if target.is_file() else target
    path = directory / "pins.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    pins = payload["pins"]
    return pins, payload.get("pins_hash") or artifacts.pins_hash(pins)


SCHEMA_KEY = "artifact_schema"
HARNESS_KEY = "harness_git_sha"
#: Shared pins that may differ across a drift pair. They are reported, never
#: refused: a schema bump that only adds a pin, and the harness commit — the
#: published restart pair (0/100 changed) spans /4 -> /5 and three commits.
REPORTED_NOT_REFUSED = frozenset({SCHEMA_KEY, HARNESS_KEY})
#: The extraction-era pins (schema /8, 2026-09-13) are written as the literal
#: ``"none"`` on an arm without an extractor; before /8 the same keys were
#: placeholders, ``None``, because nothing existed to hash. One absent
#: extractor, two spellings: a pair that differs only in that is one
#: configuration (the 2026-09-14 p2base arm vs the published k10 arm).
ABSENT = frozenset({None, "none"})


def pins_differences(reference: dict, fresh: dict) -> list[str]:
    """Keys present on both sides whose values differ, the reported pins aside.

    A drift pair is judged on shared pins, not on the hash: a schema bump
    that only *adds* a pin (``/4`` → ``/5`` added the Ollama build the same
    daemon had always served) leaves every shared key equal and the hashes
    different. Any shared pin outside :data:`REPORTED_NOT_REFUSED` that
    differs is a refusal — except an ``extractor_*`` pin that is absent on
    both sides under the two spellings of absent (:data:`ABSENT`).
    """
    return sorted(
        key
        for key in set(reference) & set(fresh)
        if key not in REPORTED_NOT_REFUSED
        and reference[key] != fresh[key]
        and not (
            key.startswith("extractor_")
            and reference[key] in ABSENT
            and fresh[key] in ABSENT
        )
    )


def _fingerprints(target: Path) -> dict | None:
    directory = target.parent if target.is_file() else target
    resolved = artifacts.read_reader_resolved_optional(directory)
    if not resolved:
        return None
    return resolved.get("system_fingerprints")


def first_divergence(a: str, b: str) -> int | None:
    """Byte offset (UTF-8) of the first difference, ``None`` when identical."""
    if a == b:
        return None
    ab, bb = a.encode("utf-8"), b.encode("utf-8")
    for i, (x, y) in enumerate(zip(ab, bb, strict=False)):
        if x != y:
            return i
    return min(len(ab), len(bb))  # one is a proper prefix of the other


def compare(reference: Path, fresh: Path) -> dict:
    """The drift report for ``fresh`` against ``reference``.

    Rows are matched on ``question_id``; a mismatch in the row set is an
    error, not a partial report — a drift pair answers the same questions.
    """
    reference, fresh = Path(reference), Path(fresh)
    ref_pins, ref_hash = _pins(reference)
    new_pins, new_hash = _pins(fresh)
    pins_checked = ref_pins is not None and new_pins is not None
    if pins_checked:
        differing = pins_differences(ref_pins, new_pins)
        if differing:
            raise DriftPairError(
                f"not a drift pair: pins differ on {differing} "
                f"({ref_hash[:12]}... in {reference} vs {new_hash[:12]}... in {fresh}). "
                "A prediction that changes under different pins is a configuration "
                "difference, not drift; compare runs of one configuration."
            )
    ref_rows, new_rows = _rows(reference), _rows(fresh)
    if set(ref_rows) != set(new_rows):
        only_ref = sorted(set(ref_rows) - set(new_rows))[:5]
        only_new = sorted(set(new_rows) - set(ref_rows))[:5]
        raise DriftPairError(
            f"not a drift pair: the two sides answered different questions "
            f"({len(ref_rows)} vs {len(new_rows)} rows; only in reference: {only_ref}; "
            f"only in fresh: {only_new})."
        )
    changed = []
    prompt_tokens_changed = 0
    for qid in sorted(ref_rows):
        before, after = ref_rows[qid], new_rows[qid]
        offset = first_divergence(str(before.get("predicted", "")), str(after.get("predicted", "")))
        if before.get("reader_prompt_tokens") != after.get("reader_prompt_tokens"):
            prompt_tokens_changed += 1
        if offset is not None:
            changed.append(
                {
                    "question_id": qid,
                    "category": after.get("category"),
                    "first_divergence_byte": offset,
                    "reader_prompt_tokens": [
                        before.get("reader_prompt_tokens"),
                        after.get("reader_prompt_tokens"),
                    ],
                }
            )
    return {
        "reference": str(reference),
        "fresh": str(fresh),
        "pins_hash": {"reference": ref_hash, "fresh": new_hash},
        "artifact_schema": {
            "reference": (ref_pins or {}).get(SCHEMA_KEY),
            "fresh": (new_pins or {}).get(SCHEMA_KEY),
        },
        "harness_git_sha": {
            "reference": (ref_pins or {}).get(HARNESS_KEY),
            "fresh": (new_pins or {}).get(HARNESS_KEY),
        },
        "pins_checked": pins_checked,
        "n": len(ref_rows),
        "changed": len(changed),
        "prompt_tokens_changed": prompt_tokens_changed,
        "rows": changed,
        "system_fingerprints": {
            "reference": _fingerprints(reference),
            "fresh": _fingerprints(fresh),
        },
    }


def statement(report: dict) -> str:
    """The one-line Tier 2 sentence the report supports."""
    n, changed = report["n"], report["changed"]
    if changed == 0:
        return f"drift: 0/{n} predictions changed (byte-identical rows)"
    return f"drift: {changed}/{n} predictions changed"


def format_report(report: dict) -> str:
    lines = [statement(report)]
    lines.append(
        f"  prompt tokens changed on {report['prompt_tokens_changed']}/{report['n']} rows"
        + ("" if report["pins_checked"] else "  (pins not checked: one side has no pins.json)")
    )
    sha = report["harness_git_sha"]
    if sha["reference"] != sha["fresh"]:
        lines.append(
            f"  harness commit differs: {sha['reference']} -> {sha['fresh']} "
            "(a changed row may be the code, not the reader; say which before quoting)"
        )
    fps = report["system_fingerprints"]
    if fps.get("reference") or fps.get("fresh"):
        lines.append(f"  fingerprints reference={fps.get('reference')} fresh={fps.get('fresh')}")
    for row in report["rows"]:
        lines.append(
            f"  {row['question_id']}  ({row['category']})  first divergence at byte "
            f"{row['first_divergence_byte']}  prompt_tokens {row['reader_prompt_tokens']}"
        )
    return "\n".join(lines)


def write_report(directory: Path, report: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DRIFT_FILE
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def verify(reference: Path, fresh_dir: Path) -> dict:
    """Compare, print, and leave ``drift.json`` in the fresh run directory."""
    report = compare(reference, fresh_dir)
    print(format_report(report), file=sys.stderr)
    path = write_report(fresh_dir, report)
    print(f"wrote {path}", file=sys.stderr)
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(
            "usage: python -m evals.drift <reference run dir | predictions.jsonl> "
            "<fresh run dir | predictions.jsonl>",
            file=sys.stderr,
        )
        return 2
    reference, fresh = Path(argv[0]), Path(argv[1])
    try:
        report = compare(reference, fresh)
    except (DriftPairError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if fresh.is_dir():
        print(f"wrote {write_report(fresh, report)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
