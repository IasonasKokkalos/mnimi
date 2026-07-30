"""LLM judge, faithful to the LongMemEval paper's ``get_anscheck_prompt``.

(src/evaluation/evaluate_qa.py in github.com/xiaowu0162/LongMemEval.) The judge
returns yes/no; correctness is ``'yes' in output.lower()``. Per-type rules:
temporal-reasoning forgives off-by-one day counts; knowledge-update accepts the
updated answer even if prior info is included; preference grades against a rubric;
abstention checks the model correctly refused to answer.

Transport runs on the OpenAI API (``gpt-4o-2024-08-06``, key from
``OPENAI_API_KEY``): the paper's 97% human agreement is validated for that
snapshot with these exact prompts. The five templates are byte-identical to the
reference (locked by tests/test_judge.py against verbatim copies), the dispatch
mirrors its structure including the ``NotImplementedError`` on unknown types,
and the decode config matches its request kwargs.
"""

from __future__ import annotations

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no. \n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, etc., "
    "and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct. \n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\n"
    "Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the "
    "points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\n"
    "Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


_TEMPLATES = {
    "standard": _STANDARD,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
    "abstention": _ABSTENTION,
}

# No system message. The reference implementation sends exactly one user
# message — `messages=[{"role": "user", "content": prompt}]`
# (src/evaluation/evaluate_qa.py, verified against the source) — and the 97%
# human agreement was measured on that request, not on ours. This was a
# harness-invented system message ("You are a strict grader...") until
# 2026-07-29.
#
# It is None rather than deleted so the absence is a recorded value: the hash
# below covers it, and re-introducing a system message moves the hash instead
# of slipping past a digest that only ever looked at the templates.
_JUDGE_SYSTEM: str | None = None

# Decode config, pinned. The reference sends `temperature=0, max_tokens=10`
# (evaluate_qa.py kwargs), so these are part of the instrument, not transport
# detail: max_tokens=10 is what truncates the verdict to a bare yes/no, and a
# different budget invites a hedged sentence the `'yes' in output` check then
# misreads. Digested in `judge_prompt_hash` below — every configuration input
# to a verdict must invalidate the cache when it changes.
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 10

# v3: the five templates are now BYTE-IDENTICAL to the reference (v2 carried a
# known one-character deviation in `_STANDARD` and `_TEMPORAL` — the missing
# space before `\n\nQuestion:` — fixed under an explicit ruling, 2026-07-30),
# unknown question types raise like the reference instead of falling through to
# `_STANDARD`, and the request hash covers decode config alongside the message
# stack. The name stops overclaiming with this version.
JUDGE_PROMPT_VERSION = "longmemeval-paper-v3"


def judge_messages(prompt: str) -> list[dict]:
    """The message stack sent to the judge.

    Single source for what is sent AND what is hashed — the two cannot drift.
    """
    messages = []
    if _JUDGE_SYSTEM is not None:
        messages.append({"role": "system", "content": _JUDGE_SYSTEM})
    messages.append({"role": "user", "content": prompt})
    return messages


def judge_request_shape() -> dict:
    """The judge request as sent, with per-question text left unrendered.

    Covers roles as well as contents: a verdict depends on the whole message
    stack, not on the user template alone. Hashing only the five templates was
    the hole this closes — a system message could be added, edited or removed
    without moving `judge_prompt_hash`, which is how one got added unnoticed.
    """
    return {
        "system_message": _JUDGE_SYSTEM,  # None = no system message is sent
        "roles": [m["role"] for m in judge_messages("")],
        "templates": dict(sorted(_TEMPLATES.items())),
        # Decode config is part of the request the verdicts came from: a wider
        # max_tokens or a nonzero temperature is a different judge.
        "decode": {"temperature": JUDGE_TEMPERATURE, "max_tokens": JUDGE_MAX_TOKENS},
    }


def judge_prompt_hash() -> str:
    """Digest over the entire judge request shape, decode config included."""
    from .artifacts import canonical, fingerprint

    return fingerprint(canonical(judge_request_shape()))


def judge_block(judge_model: str) -> dict:
    """Judge identity, written into ``results.json`` beside the verdicts.

    Refreshed at judge time from the judge about to run — never copied from
    pins, which describe the predict stage and are not rewritten by a judge
    replay. Every verdict set on disk sits beside the exact judge that produced
    it: model, prompt version, prompt hash, decode config.
    """
    return {
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_hash": judge_prompt_hash(),
        "judge_temperature": JUDGE_TEMPERATURE,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
    }


# The reference's standard-template dispatch list (evaluate_qa.py line 26).
_STANDARD_TYPES = frozenset(
    {"single-session-user", "single-session-assistant", "multi-session"}
)


def build_judge_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    """Select and fill the per-type judge template.

    Mirrors the reference dispatch exactly, including its strictness: an
    unknown question type raises rather than borrowing ``_STANDARD``. A silent
    fallback would grade a new category with a template never validated for it
    and report the number as if it meant the same thing.
    """
    if abstention:
        template = _ABSTENTION
    elif question_type in _STANDARD_TYPES:
        template = _STANDARD
    elif question_type == "temporal-reasoning":
        template = _TEMPORAL
    elif question_type == "knowledge-update":
        template = _KNOWLEDGE_UPDATE
    elif question_type == "single-session-preference":
        template = _PREFERENCE
    else:
        raise NotImplementedError(
            f"no judge template for question type {question_type!r} — the "
            "reference implementation raises here too, and grading it with "
            "_STANDARD would be a silent template substitution"
        )
    return template.format(question=question, answer=answer, response=response)


class Judge:
    """OpenAI-backed yes/no grader with an optional call-avoidance cache."""

    def __init__(self, model: str, client=None, cache=None) -> None:
        self.model = model
        if client is None:
            import openai

            client = openai.OpenAI(max_retries=4)
        self.client = client
        # Optional JudgeCache; when present, identical (question_id, predicted)
        # pairs skip the API call. See evals/judge_cache.py.
        self.cache = cache

    def is_correct(
        self,
        *,
        question_id: str,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        abstention: bool,
    ) -> bool:
        if self.cache is not None:
            cached = self.cache.get(question_id, response)
            if cached is not None:
                return cached
        prompt = build_judge_prompt(question_type, question, answer, response, abstention)
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            messages=judge_messages(prompt),
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        verdict = "yes" in text
        if self.cache is not None:
            self.cache.set(question_id, response, verdict)
        return verdict
