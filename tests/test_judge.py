"""The judge request is what the reference sends, and its hash covers all of it.

Two properties below were broken and untested until 2026-07-29: a
harness-invented system message rode along outside the hash, and the verdict
cache key ignored the judge entirely. The byte-locks against the verbatim
reference strings landed 2026-07-30 with the one-character template fixes.
"""

from __future__ import annotations

import pytest
from evals import judge
from evals.judge_cache import JudgeCache

# The five templates of the reference implementation, VERBATIM — copied
# byte-for-byte from `get_anscheck_prompt` in src/evaluation/evaluate_qa.py
# (github.com/xiaowu0162/LongMemEval), each kept as one uninterrupted string
# literal so no editor or formatter can reshape it without failing this file.
# Note the trailing space before `\n\nQuestion:` in the first two — that space
# is the one-character deviation this lock exists to keep fixed.
REFERENCE = {
    "standard": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",  # noqa: E501
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",  # noqa: E501
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",  # noqa: E501
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",  # noqa: E501
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",  # noqa: E501
}


def test_all_five_templates_are_byte_identical_to_the_reference():
    """Substituting our named fields into the reference's positional slots must
    reproduce our template constants exactly — one byte off and the judge is a
    different instrument than the one validated at 97% human agreement."""
    for key, reference in REFERENCE.items():
        expected = reference.format("{question}", "{answer}", "{response}")
        assert judge._TEMPLATES[key] == expected, f"{key} deviates from the reference"


def test_dispatch_renders_the_reference_prompt_for_every_category():
    """The three standard-list categories all resolve to the standard template,
    exactly as the reference's membership test does."""
    for category in ("single-session-user", "single-session-assistant", "multi-session"):
        rendered = judge.build_judge_prompt(category, "Q?", "A", "R", abstention=False)
        assert rendered == REFERENCE["standard"].format("Q?", "A", "R"), category

    rendered = judge.build_judge_prompt("temporal-reasoning", "Q?", "A", "R", abstention=False)
    assert rendered == REFERENCE["temporal-reasoning"].format("Q?", "A", "R")


def test_unknown_question_type_raises_like_the_reference():
    """No silent fallback to _STANDARD: a new category must fail loudly, not be
    graded by a template never validated for it."""
    with pytest.raises(NotImplementedError, match="multi-hop-v9"):
        judge.build_judge_prompt("multi-hop-v9", "q", "a", "r", abstention=False)
    # Abstention dispatches first, whatever the type says — same as the
    # reference, where the abstention branch sits outside the type dispatch.
    rendered = judge.build_judge_prompt("multi-hop-v9", "q", "a", "r", abstention=True)
    assert rendered == REFERENCE["abstention"].format("q", "a", "r")


def test_decode_config_moves_the_judge_hash(monkeypatch):
    """temperature and max_tokens are configuration inputs to every verdict, so
    changing either must be as loud as editing a template."""
    baseline = judge.judge_prompt_hash()
    monkeypatch.setattr(judge, "JUDGE_MAX_TOKENS", 500)
    widened = judge.judge_prompt_hash()
    assert widened != baseline
    monkeypatch.setattr(judge, "JUDGE_MAX_TOKENS", 10)
    monkeypatch.setattr(judge, "JUDGE_TEMPERATURE", 1)
    assert judge.judge_prompt_hash() not in {baseline, widened}


def test_decode_config_change_is_a_cache_miss(tmp_path, monkeypatch):
    """An otherwise identical grading under a different decode config must not
    replay the old verdict — the fingerprint carries judge_prompt_hash, which
    digests decode, so the key moves."""
    path = str(tmp_path / "verdicts.json")
    before = f"gpt-4o-2024-08-06:{judge.judge_prompt_hash()}"
    JudgeCache(path, judge_fingerprint=before).set("q1", "same prediction", True)

    monkeypatch.setattr(judge, "JUDGE_MAX_TOKENS", 500)
    after = f"gpt-4o-2024-08-06:{judge.judge_prompt_hash()}"
    assert after != before
    assert JudgeCache(path, judge_fingerprint=after).get("q1", "same prediction") is None


def test_the_hash_is_computed_from_the_rendering_that_is_sent():
    """One source for what is sent AND what is hashed: the captured request must
    be reconstructable from `judge_request_shape()` alone."""
    captured = {}

    class FakeClient:
        def __init__(self):
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": "yes"})()})()]},
            )()

    judge.Judge("gpt-4o-2024-08-06", client=FakeClient()).is_correct(
        question_id="q1", question="Q?", answer="A", response="R",
        question_type="temporal-reasoning", abstention=False,
    )

    shape = judge.judge_request_shape()
    assert [m["role"] for m in captured["messages"]] == shape["roles"]
    assert captured["temperature"] == shape["decode"]["temperature"]
    assert captured["max_tokens"] == shape["decode"]["max_tokens"]
    assert captured["messages"][-1]["content"] == shape["templates"][
        "temporal-reasoning"
    ].format(question="Q?", answer="A", response="R")


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
