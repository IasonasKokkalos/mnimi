"""The extraction output contract: a JSON schema, its locked field order, its parser.

The schema is what the grammar is generated from (llama.cpp's JSON-schema →
GBNF converter, in ``.llama``), and the field order is part of the pinned
prompt surface: free prose first (``content``, ``raw``), then the verbatim
time mention, then the constrained slots (SPEC CHANGELOG #12 — premature
serialisation degrades capacity-limited models; PHASE2 D8 — ``when`` is a
mention the resolver reads, not a date the model computes).
"""

from __future__ import annotations

import json

from .protocol import ExtractedFact

SCHEMA_VERSION = "fact-v1"

#: Locked: prose before slots. Part of ``extractor_prompt_hash``'s surface.
FIELD_ORDER = ("content", "raw", "when", "subject", "predicate", "object", "salience")

MAX_FACTS = 8
MAX_TEXT_CHARS = 300
MAX_SLOT_CHARS = 120
SALIENCE_LEVELS = (0.25, 0.5, 1.0)

FACT_SCHEMA: dict = {
    "type": "array",
    "maxItems": MAX_FACTS,
    "items": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
            "raw": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS},
            "when": {"type": ["string", "null"], "maxLength": MAX_SLOT_CHARS},
            "subject": {"type": ["string", "null"], "maxLength": MAX_SLOT_CHARS},
            "predicate": {"type": ["string", "null"], "maxLength": MAX_SLOT_CHARS},
            "object": {"type": ["string", "null"], "maxLength": MAX_SLOT_CHARS},
            "salience": {"enum": list(SALIENCE_LEVELS)},
        },
        "required": list(FIELD_ORDER),
        "additionalProperties": False,
    },
}


def field_order() -> list[str]:
    return list(FIELD_ORDER)


def _encodable(value) -> bool:
    """``False`` when a string carries an unpaired surrogate.

    A Python ``str`` can hold one — ``json.loads`` produces it from a lone
    ``\\uD83C`` escape — but it is not valid Unicode and cannot be encoded as
    UTF-8, so every downstream consumer that crosses into Rust or onto a wire
    rejects it. Measured on the LongMemEval corpus: 1 round in 60,467, where
    the model's output ended mid-emoji.
    """
    if not isinstance(value, str):
        return True
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _optional_text(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("slot is not a string")
    stripped = value.strip()
    return stripped or None


def parse_output(text: str) -> tuple[list[ExtractedFact], bool]:
    """``(facts, truncated)`` from the model's raw output.

    Anything that is not the contract — unparseable JSON (the token budget cut
    the array), a non-array, an item missing a field or carrying a wrong type
    — is ``([], True)``: the grammar makes such output impossible except by
    truncation, and SPEC says a truncated parse is ``[]`` for that turn. The
    parser is strict rather than salvaging partial arrays so the cache never
    holds an output whose reading depends on parser leniency.
    """
    try:
        # strict=False: the runtime's JSON grammar (``char ::= [^"\\] | ...``)
        # admits raw control characters inside strings, and a verbatim ``raw``
        # excerpt of a multi-line assistant turn carries real newlines. The
        # grammar defines the language the model emits; the parser reads it.
        payload = json.loads(text, strict=False)
    except (TypeError, ValueError):
        return [], True
    if not isinstance(payload, list):
        return [], True
    facts: list[ExtractedFact] = []
    try:
        for item in payload:
            if not isinstance(item, dict) or set(item) != set(FIELD_ORDER):
                return [], True
            content, raw = item["content"], item["raw"]
            if not isinstance(content, str) or not isinstance(raw, str):
                return [], True
            salience = item["salience"]
            if isinstance(salience, bool) or not isinstance(salience, (int, float)):
                return [], True
            # A string the model cut mid-emoji carries the HIGH half of a
            # surrogate pair with no low half (``\uD83C`` with nothing after
            # it). ``json.loads`` materialises it faithfully, but the result is
            # not valid Unicode: it cannot be UTF-8 encoded, and the embedder's
            # Rust tokenizer refuses the whole batch. That is the same evidence
            # of a cut-off output as a truncated array, so it takes the same
            # path rather than a repair — the parser salvages nothing, here as
            # everywhere else in this function.
            if not all(_encodable(item[field]) for field in FIELD_ORDER):
                return [], True
            facts.append(
                ExtractedFact(
                    content=content.strip(),
                    raw=raw.strip(),
                    when=_optional_text(item["when"]),
                    subject=_optional_text(item["subject"]),
                    predicate=_optional_text(item["predicate"]),
                    object=_optional_text(item["object"]),
                    salience=float(salience),
                )
            )
    except (KeyError, ValueError):
        return [], True
    if any(not fact.content or not fact.raw for fact in facts):
        return [], True
    return facts, False
