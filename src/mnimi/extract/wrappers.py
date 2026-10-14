"""Composable extractors over an inner ``Extractor`` (TESTFIXESPT1 Task 3).

Each wrapper implements the protocol (``extract``, ``pins``) and hands ``pins`` through
unchanged: the store's ``memory_meta`` describes the model configuration, and none of these
change the model, the prompt, the schema, the grammar or the decode. What they change is what
the model is shown (the escape, the split) or whether it is called at all (client facts).

* :class:`BackslashEscapingExtractor` — the JSON grammar rejects ``\\P``; the 1.7B model does
  not escape backslashes, the rescan substitutes a legal escape and the string closes (issue 4:
  ``Worktree D:\\\\``). The model is shown ``/`` instead; ``mnimi.extract.verify`` restores the
  turn's bytes in ``raw`` (``/`` and ``\\`` are one character there). ``content`` keeps ``/``.
* :class:`SplittingExtractor` — the output budget (``max_tokens`` 1024 ≈ 5,600 chars of JSON)
  and ``MAX_FACTS`` 8 are per call; a long round was cut mid-fact and the strict parser dropped
  every fact (1 of 9 live rounds; 312 of 95,846 cached). Turns longer than ``max_chars`` are cut
  at sentence boundaries and the inner is called once per sub-round; a truncated sub-round
  loses only its own facts.
* :class:`ClientFactsExtractor` — the calling model supplies the facts for the next round
  (the MCP server's ``remember(..., facts=)``); they go through the same resolver, screens and
  conflict pass as the model's, and the inner is not called for that round.
"""

from __future__ import annotations

import json

from .protocol import ExtractedFact, ExtractionResult, sha256_text
from .verify import sentences, verify_raw

DEFAULT_SPLIT_CHARS = 600

#: The pins of a client-facts store with no model behind it (``ClientFactsExtractor(None)``).
CLIENT_FACTS_PINS = {
    "extractor_model": "client-facts",
    "extractor_quant": "none",
    "extractor_runtime": "mnimi.extract.wrappers ClientFactsExtractor v1",
    "extractor_decode_hash": sha256_text("client-facts decode v1"),
    "extractor_prompt_hash": sha256_text("client-facts prompt v1"),
}


def _plain(turns: list[dict]) -> list[dict]:
    return [{"role": str(t.get("role", "")), "content": str(t.get("content", ""))} for t in turns]


class BackslashEscapingExtractor:
    """The inner model sees ``/`` wherever the turn has ``\\``."""

    def __init__(self, inner) -> None:
        self.inner = inner

    @property
    def pins(self) -> dict:
        return self.inner.pins

    def extract(self, turns: list[dict]) -> ExtractionResult:
        safe = [{"role": t["role"], "content": t["content"].replace("\\", "/")}
                for t in _plain(turns)]
        return self.inner.extract(safe)


def _units(text: str, max_chars: int):
    for sentence in sentences(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars + 1)
            cut = cut if cut > 0 else max_chars
            yield sentence[:cut].strip()
            sentence = sentence[cut:].strip()
        if sentence:
            yield sentence


def split_text(text: str, max_chars: int) -> list[str]:
    """Sentences packed in order into chunks of at most ``max_chars``; ``<= 0`` means no split.

    A sentence longer than the budget is cut at the last space inside it (at the budget when
    it has none). Chunks are joined with one space, so a chunk's text differs from the turn's
    only in whitespace — which the verification forgives.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for unit in _units(text, max_chars):
        candidate = f"{current} {unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


class SplittingExtractor:
    """One inner call per sub-round of a long round; the facts concatenated."""

    def __init__(self, inner, max_chars: int = DEFAULT_SPLIT_CHARS) -> None:
        self.inner = inner
        self.max_chars = int(max_chars)
        self.sub_rounds = 0  # diagnostic: inner calls made for split rounds

    @property
    def pins(self) -> dict:
        return self.inner.pins

    def extract(self, turns: list[dict]) -> ExtractionResult:
        per_turn = [split_text(str(t.get("content", "")), self.max_chars) for t in turns]
        if max((len(c) for c in per_turn), default=0) <= 1:
            return self.inner.extract(turns)
        facts: list[ExtractedFact] = []
        outputs: list[str] = []
        truncated = truncated_input = False
        for i in range(max(len(c) for c in per_turn)):
            sub = [
                {"role": str(t.get("role", "")), "content": per_turn[j][i]}
                for j, t in enumerate(turns) if i < len(per_turn[j])
            ]
            result = self.inner.extract(sub)
            self.sub_rounds += 1
            facts.extend(result.facts)
            outputs.append(result.raw_output)
            truncated = truncated or bool(result.truncated)
            truncated_input = truncated_input or bool(getattr(result, "truncated_input", False))
        return ExtractionResult(facts=facts, truncated=truncated, raw_output="\n".join(outputs),
                                truncated_input=truncated_input)


class ClientFactsExtractor:
    """Facts queued by the caller for the next round; otherwise the inner, or nothing."""

    def __init__(self, inner=None) -> None:
        self.inner = inner
        self.pending: list[ExtractedFact] | None = None

    @property
    def pins(self) -> dict:
        return self.inner.pins if self.inner is not None else dict(CLIENT_FACTS_PINS)

    def provide(self, facts: list[ExtractedFact]) -> None:
        self.pending = list(facts)

    def extract(self, turns: list[dict]) -> ExtractionResult:
        if self.pending is None:
            if self.inner is None:
                return ExtractionResult(facts=[], truncated=False, raw_output="[]")
            return self.inner.extract(turns)
        queued, self.pending = self.pending, None
        facts: list[ExtractedFact] = []
        for fact in queued:
            if not fact.raw:
                verdict = verify_raw(fact.content, "", turns)
                filled = verdict.raw if verdict.how != "kept" else f"user: {fact.content}"
                fact = fact._replace(raw=filled)
            facts.append(fact)
        raw_output = json.dumps([f._asdict() for f in facts], ensure_ascii=False)
        return ExtractionResult(facts=facts, truncated=False, raw_output=raw_output)
