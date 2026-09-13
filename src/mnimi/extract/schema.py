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
        payload = json.loads(text)
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
