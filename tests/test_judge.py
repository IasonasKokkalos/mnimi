"""The judge request is what the reference sends, and its hash covers all of it.

Both properties below were broken and untested until 2026-07-29: a
harness-invented system message rode along outside the hash, and the verdict
cache key ignored the judge entirely.
"""

from __future__ import annotations

from evals import judge
from evals.judge_cache import JudgeCache


def test_no_system_message_is_sent():
    """The reference sends exactly one user message
    (evaluate_qa.py: messages=[{"role": "user", "content": prompt}]), and the
    paper's 97% human agreement was measured on that request."""
    messages = judge.judge_messages("grade this")

    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "grade this"


def test_prompt_hash_covers_the_system_message_slot():
    """The hole: a system message could be added, edited or removed without
    moving a hash that only ever digested the five user templates."""
    baseline = judge.judge_prompt_hash()
    original = judge._JUDGE_SYSTEM
    try:
        judge._JUDGE_SYSTEM = "You are a strict grader."
        assert judge.judge_prompt_hash() != baseline
    finally:
        judge._JUDGE_SYSTEM = original
    assert judge.judge_prompt_hash() == baseline


def test_prompt_hash_covers_every_template():
    baseline = judge.judge_prompt_hash()
    original = judge._TEMPLATES["abstention"]
    try:
        judge._TEMPLATES["abstention"] = original + " "
        assert judge.judge_prompt_hash() != baseline
    finally:
        judge._TEMPLATES["abstention"] = original


def test_request_shape_records_the_absence_of_a_system_message():
    shape = judge.judge_request_shape()
    assert shape["system_message"] is None, "absence is a recorded value, not a missing key"
    assert shape["roles"] == ["user"]
    assert set(shape["templates"]) == {
        "standard",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
        "abstention",
    }


def test_verdict_cache_is_scoped_to_the_judge(tmp_path):
    """Reader drift always moved the key (it changes `predicted`); judge drift
    never did, so a re-grade after a prompt change replayed stale verdicts."""
    path = str(tmp_path / "verdicts.json")
    before = JudgeCache(path, judge_fingerprint="gpt-4o:hash-A")
    before.set("q1", "the answer is 42", True)

    after = JudgeCache(path, judge_fingerprint="gpt-4o:hash-B")
    assert after.get("q1", "the answer is 42") is None, "a changed judge must miss"

    same = JudgeCache(path, judge_fingerprint="gpt-4o:hash-A")
    assert same.get("q1", "the answer is 42") is True
