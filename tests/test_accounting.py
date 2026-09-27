"""C2 (judge test-retest) and C1 (the oracle-paired accounting) over committed files (PHASE8 D2)."""

from __future__ import annotations

import json

import pytest
from evals import accounting


def _arm(directory, first, replays, categories=None, judge="gpt-4o-2024-08-06"):
    directory.mkdir(parents=True)
    categories = categories or {q: "multi-session" for q in first}
    (directory / "results.json").write_text(json.dumps({
        "judge": {"judge_model": judge}, "pins_hash": "p",
        "results": [{"question_id": q, "correct": c} for q, c in first.items()],
    }), encoding="utf-8")
    (directory / "predictions.jsonl").write_text("".join(
        json.dumps({"question_id": q, "category": categories[q], "is_abstention": False,
                    "verdicts": [{"correct": c}]}) + "\n" for q, c in first.items()),
        encoding="utf-8")
    for n, verdicts in enumerate(replays, start=1):
        (directory / f"judge_replay_{n}.json").write_text(json.dumps({
            "judge": {"judge_model": judge}, "judge_hash": "h",
            "run": {"judge_cache": "off", "judge_cache_hits": 0},
            "results": [{"question_id": q, "correct": c} for q, c in verdicts.items()],
        }), encoding="utf-8")


def test_retest_marks_a_row_unstable_when_any_grading_differs(tmp_path):
    first = {"q1": True, "q2": True, "q3": False}
    _arm(tmp_path / "a", first, [
        {"q1": True, "q2": False, "q3": False},
        {"q1": True, "q2": True, "q3": False},
        {"q1": True, "q2": True, "q3": True},
    ])
    r = accounting.retest(tmp_path / "a", "gpt-4o-2024-08-06")
    assert r["replays"] == 3 and r["n"] == 3
    assert sorted(r["unstable_ids"]) == ["q2", "q3"] and r["unstable"] == 2
    assert r["score_by_grading"] == [2, 1, 2, 3]
    assert r["flips_to_wrong"] == 1 and r["flips_to_right"] == 1
    assert 0 < r["rate_low"] < r["rate"] < r["rate_high"] <= 1


def test_retest_refuses_a_run_with_no_replay_of_that_judge(tmp_path):
    _arm(tmp_path / "a", {"q1": True}, [])
    with pytest.raises(ValueError, match="no judge_replay"):
        accounting.retest(tmp_path / "a", "gpt-4o-2024-08-06")


def test_buckets_remove_unstable_rows_and_split_per_category(tmp_path):
    cats = {"q1": "multi-session", "q2": "multi-session",
            "q3": "knowledge-update", "q4": "knowledge-update"}
    _arm(tmp_path / "sys", {"q1": True, "q2": False, "q3": False, "q4": True}, [], cats)
    _arm(tmp_path / "ora", {"q1": True, "q2": True, "q3": False, "q4": False}, [], cats)
    b = accounting.buckets(tmp_path / "sys", tmp_path / "ora", unstable_ids={"q4"})
    assert b["overall"] == {"both_right": 1, "oracle_right_system_wrong": 1, "both_wrong": 1,
                            "system_right_oracle_wrong": 0, "n_stable": 3, "n_unstable": 1}
    assert b["by_category"]["multi-session"]["oracle_right_system_wrong"] == 1
    assert b["by_category"]["knowledge-update"]["both_wrong"] == 1
    assert b["by_category"]["knowledge-update"]["n_unstable"] == 1


def test_buckets_refuse_different_question_sets(tmp_path):
    _arm(tmp_path / "sys", {"q1": True}, [])
    _arm(tmp_path / "ora", {"q9": True}, [])
    with pytest.raises(ValueError, match="question sets differ"):
        accounting.buckets(tmp_path / "sys", tmp_path / "ora", unstable_ids=set())

def test_retest_refuses_a_replay_that_read_the_cache(tmp_path):
    _arm(tmp_path / "a", {"q1": True}, [{"q1": True}])
    replay = tmp_path / "a" / "judge_replay_1.json"
    payload = json.loads(replay.read_text(encoding="utf-8"))
    payload["run"] = {"judge_cache": "on", "judge_cache_hits": 1}
    replay.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cache"):
        accounting.retest(tmp_path / "a", "gpt-4o-2024-08-06")
