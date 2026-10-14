"""Post-hoc verification of a fact's ``raw`` excerpt and role (TESTFIXESPT1 Task 1).

The extractor is asked to copy ``raw`` verbatim from one turn and to prefix it with that
turn's role. Measured on the LongMemEval cache (2026-10-14, 241,555 facts): 79 % are
verbatim in their own turn, 2 % verbatim in the other turn (a wrong prefix), 19 % verbatim
nowhere — the 300-character cap squeezing the tail into CJK tokens, the grammar rescan after
an unescaped backslash, paraphrase. ``raw`` is embedded (``FACT_EMBED_TEMPLATE``) and rendered
(``source:``), so a wrong excerpt reaches the vector and the reader. This module repairs it
deterministically and with no model: the excerpt handed back is always a span of the round's
own text when one can be found, role-prefixed with the role of the turn it was found in.

Order of repair, first that applies:

1. **verbatim** — the body occurs in a turn (case- and whitespace-insensitive; ``/`` and ``\\``
   are one character, for the pre-escape wrapper): the turn's bytes, that turn's role.
2. **role** — the same, found under another role than the prefix claimed.
3. **prefix** — the longest leading run of at least ``MIN_PREFIX_WORDS`` words occurs in a
   turn: the span from there to the end of the sentence it lands in (the cut excerpt grows
   back to a whole sentence).
4. **overlap** — no run matches: the turn sentence with the highest word-set Jaccard against
   ``content``.
5. **kept** — nothing overlaps at all: the model's ``raw`` as it came, role unverified.
"""

from __future__ import annotations

import re
from typing import NamedTuple

ROLE_PREFIX_RE = re.compile(r"^(user|assistant):\s*", re.IGNORECASE)
MIN_PREFIX_WORDS = 5
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|\n")
_WORD_RE = re.compile(r"[\w'-]+")


class RawVerdict(NamedTuple):
    raw: str
    """The role-prefixed excerpt to store: the turn's own bytes when verified."""
    role: str | None
    """The role of the turn the excerpt was verified in; ``None`` when unverified."""
    how: str
    """``verbatim`` | ``role`` | ``prefix`` | ``overlap`` | ``kept``."""


def split_role(raw: str) -> tuple[str | None, str]:
    """``("user", "I moved")`` from ``"user: I moved"``; ``(None, raw)`` without a prefix."""
    m = ROLE_PREFIX_RE.match(raw)
    if m:
        return m.group(1).lower(), raw[m.end():].strip()
    return None, raw.strip()


def sentences(text: str) -> list[str]:
    """The text cut at sentence terminators and newlines, stripped, empties dropped."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _word_pattern(word: str) -> str:
    return "".join(r"[/\\]" if ch in "/\\" else re.escape(ch) for ch in word)


def _locate(words: list[str], text: str) -> tuple[int, int] | None:
    """Span of ``words`` in ``text``: case-insensitive, any whitespace between words."""
    if not words:
        return None
    pattern = r"\s+".join(_word_pattern(w) for w in words)
    m = re.search(pattern, text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def _sentence_end(text: str, pos: int) -> int:
    m = _SENTENCE_END_RE.search(text, pos)
    if m is None:
        return len(text)
    return m.start() if text[m.start()] == "\n" else m.end()


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def verify_raw(content: str, raw: str, turns: list[dict]) -> RawVerdict:
    """The excerpt to store for a fact, verified against the round's turns (module docstring)."""
    role, body = split_role(raw)
    turn_list = [(str(t.get("role", "")), str(t.get("content", ""))) for t in turns]
    words = body.split()
    if words:
        for r, text in turn_list:
            span = _locate(words, text)
            if span is not None:
                excerpt = text[span[0]:span[1]]
                return RawVerdict(f"{r}: {excerpt}", r, "verbatim" if r == role else "role")
        for n in range(len(words) - 1, MIN_PREFIX_WORDS - 1, -1):
            for r, text in turn_list:
                span = _locate(words[:n], text)
                if span is not None:
                    excerpt = text[span[0]:_sentence_end(text, span[1])].strip()
                    return RawVerdict(f"{r}: {excerpt}", r, "prefix")
    target = _words(content)
    best: tuple[str, str] | None = None
    best_score = 0.0
    for r, text in turn_list:
        for sentence in sentences(text):
            ws = _words(sentence)
            if not ws or not target:
                continue
            score = len(target & ws) / len(target | ws)
            if score > best_score:
                best, best_score = (r, sentence), score
    if best is not None:
        return RawVerdict(f"{best[0]}: {best[1]}", best[0], "overlap")
    return RawVerdict(raw, role, "kept")
