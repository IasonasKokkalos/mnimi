"""Model-free extractors for CI and for counting.

``RuleExtractor`` is the extraction-era counterpart of ``HashingEmbedder``: a
deterministic stand-in that exercises the whole write path (fact records,
per-kind dedup, the resolver, the render units, the probe) with no model and
no download. It is crude on purpose — first-person declaratives of the user
turn become facts — and it declares fake pins, so a store built with it is
refused by an open under the real extractor's pins.
"""

from __future__ import annotations

import json
import re

from .prompt import extractor_prompt_hash
from .protocol import ExtractedFact, ExtractionResult, sha256_text

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FIRST_PERSON = re.compile(r"\b(I|I've|I'm|I'd|I'll|My)\b")
_LEADING_ASIDE = re.compile(r"^(?:by the way|also|anyway|oh),?\s+", re.IGNORECASE)
_TIME_MENTION = re.compile(
    r"\b(yesterday|today|tomorrow|tonight|this (?:morning|week|month|year)|"
    r"last (?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"next (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten) "
    r"(?:days?|weeks?|months?|years?) ago)\b",
    re.IGNORECASE,
)
_REWRITES = (
    (re.compile(r"^I've\b"), "The user has"),
    (re.compile(r"^I'm\b"), "The user is"),
    (re.compile(r"^I'd\b"), "The user would"),
    (re.compile(r"^I'll\b"), "The user will"),
    (re.compile(r"^I\b"), "The user"),
    (re.compile(r"^My\b"), "The user's"),
)


def _facts_from_user_turn(text: str) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?") or not _FIRST_PERSON.search(sentence):
            continue
        clause = _LEADING_ASIDE.sub("", sentence)
        mention = _TIME_MENTION.search(clause)
        when = mention.group(0) if mention else None
        if mention and mention.start() == 0:
            clause = clause[mention.end() :].lstrip(" ,")
            clause = clause[:1].upper() + clause[1:]
        content = clause
        for pattern, replacement in _REWRITES:
            if pattern.search(content):
                content = pattern.sub(replacement, content, count=1)
                break
        else:
            content = "The user: " + content
        facts.append(
            ExtractedFact(
                content=content,
                raw=f"user: {sentence}",
                when=when,
                subject=None,
                predicate=None,
                object=None,
                salience=1.0,
            )
        )
    return facts


class RuleExtractor:
    """First-person declaratives of the user turn → facts. Deterministic, no model."""

    name = "fake-rule"

    @property
    def pins(self) -> dict:
        return {
            "extractor_model": self.name,
            "extractor_quant": "none",
            "extractor_runtime": "mnimi.extract.fake RuleExtractor v1",
            "extractor_decode_hash": sha256_text("fake-rule decode v1"),
            "extractor_prompt_hash": extractor_prompt_hash("<no grammar: fake-rule>"),
        }

    def extract(self, turns: list[dict]) -> ExtractionResult:
        facts: list[ExtractedFact] = []
        for turn in turns:
            if turn.get("role") == "user":
                facts.extend(_facts_from_user_turn(str(turn.get("content", ""))))
        raw_output = json.dumps([fact._asdict() for fact in facts], ensure_ascii=False)
        return ExtractionResult(facts=facts, truncated=False, raw_output=raw_output)


class ScriptedExtractor:
    """Pre-authored facts per round, keyed by the round's first user turn (PHASE3 D10).

    The conflict demo set and the Phase 3 tests script exactly which triples
    a round yields, so the write path — pre-filter, resolver, both screens,
    supersession — runs end to end with no model. An unscripted round yields
    ``[]``. Fake pins, so a scripted store is refused by a real open.
    """

    name = "fake-scripted"

    def __init__(self, script: dict[str, list[ExtractedFact]]) -> None:
        self.script = dict(script)

    @property
    def pins(self) -> dict:
        return {
            "extractor_model": self.name,
            "extractor_quant": "none",
            "extractor_runtime": "mnimi.extract.fake ScriptedExtractor v1",
            "extractor_decode_hash": sha256_text("fake-scripted decode v1"),
            "extractor_prompt_hash": extractor_prompt_hash("<no grammar: fake-scripted>"),
        }

    def extract(self, turns: list[dict]) -> ExtractionResult:
        facts: list[ExtractedFact] = []
        for turn in turns:
            if turn.get("role") == "user":
                facts = list(self.script.get(str(turn.get("content", "")), []))
                break
        raw_output = json.dumps([fact._asdict() for fact in facts], ensure_ascii=False)
        return ExtractionResult(facts=facts, truncated=False, raw_output=raw_output)


class CountingExtractor:
    """Wraps any extractor and counts the calls that reach it (the cache's test)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    @property
    def pins(self) -> dict:
        return self.inner.pins

    def extract(self, turns: list[dict]) -> ExtractionResult:
        self.calls += 1
        return self.inner.extract(turns)
