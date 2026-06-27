"""LLM judge, faithful to the LongMemEval paper's ``get_anscheck_prompt``.

(src/evaluation/evaluate_qa.py in github.com/xiaowu0162/LongMemEval.) The judge
returns yes/no; correctness is ``'yes' in output.lower()``. Per-type rules:
temporal-reasoning forgives off-by-one day counts; knowledge-update accepts the
updated answer even if prior info is included; preference grades against a rubric;
abstention checks the model correctly refused to answer.

Runs on the Anthropic API (key from ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no.\n\n"
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
    "is 18), the model's response is still correct.\n\n"
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


def build_judge_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    """Select and fill the per-type judge template."""
    if abstention:
        template = _ABSTENTION
    elif question_type == "temporal-reasoning":
        template = _TEMPORAL
    elif question_type == "knowledge-update":
        template = _KNOWLEDGE_UPDATE
    elif question_type == "single-session-preference":
        template = _PREFERENCE
    else:
        template = _STANDARD
    return template.format(question=question, answer=answer, response=response)


class Judge:
    """Anthropic-backed yes/no grader."""

    def __init__(self, model: str, client=None) -> None:
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client

    def is_correct(
        self,
        *,
        question: str,
        answer: str,
        response: str,
        question_type: str,
        abstention: bool,
    ) -> bool:
        prompt = build_judge_prompt(question_type, question, answer, response, abstention)
        # max_tokens=10 matches the paper. No temperature: Opus 4.8 rejects it.
        message = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system="You are a strict grader. Answer with only 'yes' or 'no'.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text").strip().lower()
        return "yes" in text
