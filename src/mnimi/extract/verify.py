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

1. **verbatim** — the body occurs in a turn (case- and whitespace-insensitive, on word
   boundaries; ``/`` and ``\\`` are one character, for the pre-escape wrapper): the turn's
   bytes, that turn's role. The turns of the role the prefix claims are searched first.
2. **role** — the same, found only under another role than the prefix claimed.
3. **prefix** — the longest leading run of at least ``MIN_PREFIX_WORDS`` words occurs in a
   turn: the span from there to the end of the sentence it lands in (the cut excerpt grows
   back to a whole sentence), capped at ``MAX_TEXT_CHARS`` on a whitespace boundary so the
   fact text behind it stays inside the embedder's window.
4. **overlap** — no run matches: the turn sentence with the highest Jaccard over non-stopword
   words against ``content``, if that reaches ``MIN_OVERLAP_JACCARD``.
5. **kept** — nothing qualifies: the model's ``raw`` as it came, role unverified.

The constants here are rules of this module's version, like the lexicons': an edit changes
what every stored fact embeds and is a version bump with a re-ingest, never an in-place change.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from .schema import MAX_TEXT_CHARS

VERIFY_VERSION = "v1"
ROLE_PREFIX_RE = re.compile(r"^(user|assistant):\s*", re.IGNORECASE)
MIN_PREFIX_WORDS = 5
MIN_OVERLAP_JACCARD = 0.2
_TERMINATORS = ".!?。！？"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|[。！？]|\n")
_WORD_RE = re.compile(r"[\w'-]+")
#: Words that carry no identity for the overlap fallback (the fact's own framing included).
STOPWORDS = frozenset(
    "a an and or of to in on at for with by from as is are was were be been am i you he she "
    "it we they me him her us them my your his its our their this that these those the user "
    "assistant user's assistant's".split()
)


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
    """The text cut at sentence terminators (Latin and CJK) and newlines; empties dropped."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _word_pattern(word: str) -> str:
    return "".join(r"[/\\]" if ch in "/\\" else re.escape(ch) for ch in word)


def _locate(words: list[str], text: str) -> tuple[int, int] | None:
    """Span of ``words`` in ``text``: case-insensitive, any whitespace between words, and the
    run starts and ends on a word boundary (``Yes`` never matches inside ``Yesterday``)."""
    if not words:
        return None
    body = r"\s+".join(_word_pattern(w) for w in words)
    head = r"(?<!\w)" if words[0][:1].isalnum() else ""
    tail = r"(?!\w)" if words[-1][-1:].isalnum() else ""
    m = re.search(head + body + tail, text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def _sentence_end(text: str, pos: int) -> int:
    if pos and text[pos - 1] in _TERMINATORS:
        return pos
    m = _SENTENCE_END_RE.search(text, pos)
    if m is None:
        return len(text)
    return m.start() if text[m.start()] == "\n" else m.end()


def _cap(excerpt: str) -> str:
    """At most ``MAX_TEXT_CHARS``, cut on whitespace when there is one to cut on."""
    if len(excerpt) <= MAX_TEXT_CHARS:
        return excerpt
    cut = excerpt.rfind(" ", 0, MAX_TEXT_CHARS + 1)
    return excerpt[: cut if cut > 0 else MAX_TEXT_CHARS].rstrip()


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - STOPWORDS


def verify_raw(content: str, raw: str, turns: list[dict]) -> RawVerdict:
    """The excerpt to store for a fact, verified against the round's turns (module docstring)."""
    role, body = split_role(raw)
    turn_list = [(str(t.get("role", "")), str(t.get("content", ""))) for t in turns]
    # The claimed role's turns first: a sentence both turns carry belongs to the claimed one.
    ordered = sorted(turn_list, key=lambda rt: rt[0] != role)
    words = body.split()
    if words:
        for r, text in ordered:
            span = _locate(words, text)
            if span is not None:
                excerpt = text[span[0]:span[1]]
                return RawVerdict(f"{r}: {excerpt}", r, "verbatim" if r == role else "role")
        for n in range(len(words) - 1, MIN_PREFIX_WORDS - 1, -1):
            for r, text in ordered:
                span = _locate(words[:n], text)
                if span is not None:
                    excerpt = _cap(text[span[0]:_sentence_end(text, span[1])].strip())
                    return RawVerdict(f"{r}: {excerpt}", r, "prefix")
    target = _words(content)
    best: tuple[str, str] | None = None
    best_score = 0.0
    for r, text in ordered:
        for sentence in sentences(text):
            ws = _words(sentence)
            if not ws or not target:
                continue
            score = len(target & ws) / len(target | ws)
            if score > best_score:
                best, best_score = (r, sentence), score
    if best is not None and best_score >= MIN_OVERLAP_JACCARD:
        return RawVerdict(f"{best[0]}: {_cap(best[1])}", best[0], "overlap")
    return RawVerdict(raw, role, "kept")
