"""The pinned extraction prompt and the chat template it is rendered into.

Everything the model reads is a literal in this module: the instruction, the
two examples, and the chat-template skeleton with Qwen3's thinking block left
deliberately empty (``<think>\\n\\n</think>``), which is how the post-trained
hybrid model is put into non-thinking mode. It is rendered here, by
``string.Template``, never by the runtime's template engine, so the bytes the
model sees are exactly the bytes ``extractor_prompt_hash`` covers.

An edit to any constant here is a new extractor era (a different hash, a
different cache file, a re-ingest), never an in-place change under a run.
"""

from __future__ import annotations

from string import Template

from .protocol import canonical, sha256_text
from .schema import FACT_SCHEMA, SCHEMA_VERSION

PROMPT_VERSION = "qwen3-fact-v1"

#: Round text handed to the model is capped at this many of the model's own
#: tokens (user turn first, the assistant turn takes the remainder); the
#: truncation is flagged and its rate reported. Part of the decode pins.
INPUT_CAP_TOKENS = 1536

SYSTEM_PROMPT = """\
You extract memory facts from one round of a chat between a user and an assistant.
Return a JSON array. Each element is one atomic fact with these fields, in this order:
"content": one self-contained sentence stating the fact. Begin with "The user" for a fact \
from the user's turn and "The assistant" for a fact from the assistant's turn. Keep names, \
numbers, amounts, products, places and dates exactly as written.
"raw": a contiguous excerpt of the turn the fact comes from, copied verbatim, prefixed with \
"user: " or "assistant: ".
"when": the time expression in the excerpt that says when the fact happened or holds \
("yesterday", "last month", "on October 15th", "three weeks ago"), copied verbatim; \
null if there is none.
"subject", "predicate", "object": the fact as a short triple ("user", "owns", \
"Fitbit Versa 3"); all three null if the fact is not one relation.
"salience": 1.0 for durable facts about the user (identity, possessions, relationships, \
habits, preferences, dated events, counts), 0.5 for what the assistant recommended or \
decided for this user, 0.25 for anything else.
Extract from BOTH turns: what the user did, owns, likes, plans or is — including asides \
such as "by the way, I ..." — and what the assistant recommended, decided or stated \
specifically for this user. One fact per element; several facts in one sentence become \
several elements. Do not extract generic advice, questions, greetings, or general \
knowledge. Return [] when the round holds no fact about this user or this conversation.

Example round:
user: I'm planning meals for the week, any ideas with chicken? By the way, I went grocery \
shopping last Saturday and spent around $120 at Walmart.
assistant: Happy to help! Try lemon garlic chicken or chicken fajitas. And well done on \
the shopping trip.
Output:
[{"content":"The user went grocery shopping and spent around $120 at Walmart.",\
"raw":"user: I went grocery shopping last Saturday and spent around $120 at Walmart.",\
"when":"last Saturday","subject":"user","predicate":"spent at Walmart",\
"object":"around $120","salience":1.0},\
{"content":"The assistant suggested lemon garlic chicken and chicken fajitas for the \
user's weekly meals.","raw":"assistant: Try lemon garlic chicken or chicken fajitas.",\
"when":null,"subject":"assistant","predicate":"suggested",\
"object":"lemon garlic chicken, chicken fajitas","salience":0.5}]

Example round:
user: Thanks, that was really helpful!
assistant: You're welcome! Let me know if there's anything else I can do.
Output:
[]"""

#: Qwen3 ChatML with the thinking block left empty (non-thinking mode).
CHAT_TEMPLATE = (
    "<|im_start|>system\n${system}<|im_end|>\n"
    "<|im_start|>user\n${round}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
_CHAT_T = Template(CHAT_TEMPLATE)


def render_round(turns: list[dict]) -> str:
    """``role: content`` lines — the round as the model sees it."""
    return "\n".join(
        f"{turn.get('role', '')}: {str(turn.get('content', '')).strip()}" for turn in turns
    )


def build_prompt(turns: list[dict]) -> str:
    """The exact string the model is prompted with for one round."""
    # Template.substitute never rescans substituted values, so `$` inside
    # chat text is safe; the template itself is the hashed constant.
    return _CHAT_T.substitute(system=SYSTEM_PROMPT, round=render_round(turns))


def extractor_prompt_hash(grammar_text: str) -> str:
    """Digest of the pinned prompt surface: instruction, template, schema, grammar.

    ``grammar_text`` is the GBNF the runtime actually samples under (derived
    from ``FACT_SCHEMA``, but the derivation is the runtime's, so the text is
    hashed rather than trusted). SPEC: ``extractor_prompt_hash`` covers the
    prompt *and* the grammar.
    """
    return sha256_text(
        canonical(
            {
                "prompt_version": PROMPT_VERSION,
                "system": SYSTEM_PROMPT,
                "chat_template": CHAT_TEMPLATE,
                "schema_version": SCHEMA_VERSION,
                "schema": FACT_SCHEMA,
                "grammar": grammar_text,
            }
        )
    )
