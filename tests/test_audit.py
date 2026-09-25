"""Tier 1 audit records (Phase 7 D3). Pure functions; no judge, no network."""

from __future__ import annotations

import json

import pytest
from evals import audit


def test_flips_are_counted_per_row_in_both_directions():
    got = audit.flips(
        {"a": True, "b": False, "c": True}, {"a": False, "b": True, "c": True, "d": True}
    )
    assert got == {"compared": 3, "count": 2, "to_correct": ["b"], "to_wrong": ["a"]}


def test_the_verdict_separates_a_defect_from_the_judges_flips():
    base = {
        "flips": {"count": 0, "compared": 2}, "cache": {"misses": 0},
        "published": {"correct": 1, "n": 2}, "recomputed": {"correct": 1, "n": 2},
    }
    assert audit.verdict(base) == "MATCHES"
    assert audit.verdict({**base, "flips": {"count": 2, "compared": 2}}) == "DIVERGES"
    fresh = {**base, "flips": {"count": 2, "compared": 2}, "cache": {"misses": 500}}
    assert audit.verdict(fresh) == "WITHIN RE-GRADE"
    assert audit.verdict({**base, "recomputed": {"correct": 2, "n": 2}}) == "DIVERGES"
    unpublished = {**base, "published": None, "flips": {"count": 0, "compared": 0}}
    assert audit.verdict(unpublished) == "NO REFERENCE"


def test_committed_verdicts_prefer_the_first_verdict_in_the_file(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [{"question_id": "a", "verdicts": [{"correct": True}, {"correct": False}]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert audit.committed_verdicts(path) == {"a": True}


def test_committed_verdicts_fall_back_to_results_json(tmp_path):
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"question_id": "a"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "results.json").write_text(
        json.dumps({"results": [{"question_id": "a", "correct": False}]}), encoding="utf-8"
    )
    assert audit.committed_verdicts(tmp_path / "predictions.jsonl") == {"a": False}


def test_an_audit_record_is_never_overwritten(tmp_path):
    path = tmp_path / "audits" / "x.json"
    audit.write_record(path, {"a": 1})
    with pytest.raises(FileExistsError):
        audit.write_record(path, {"a": 2})
