"""Triple normalization, polarity and the pair key (PHASE3 D3/D4).

Exact match after normalization — no lemmatization, no fuzzy matching. The
rule is code: every table below and the value-token limit are hashed into
``memory_meta`` as ``conflict_rules_hash``; a change refuses older stores.

What ``normalize_triple`` does to ``(subject, predicate, object)``:

1. Text: lower-case, apostrophes removed, punctuation stripped, whitespace
   collapsed (the same shape ``memory._normalize`` gives the exact screen).
2. Subject: leading articles dropped; every first-person form and "the user"
   become the one token ``user``.
3. Predicate: leading auxiliaries stripped while a token remains ("has been
   using" → "using"); negation markers removed and counted (longest phrase
   first); an antonym form rewritten to its canonical with its side's
   polarity ("dislikes" → ``like``, negative; "no longer likes" → ``like``,
   negative); a functional form rewritten to its group ("moved to" →
   ``lives in``).
4. Object: number words to digits, leading articles dropped, negation markers
   removed and counted ("not vegetarian" → ``vegetarian``, one marker).
5. Polarity: ``-1`` when the markers counted in 3 + 4 (plus an antonym's
   negative side) are odd, else ``+1``. A double negation is positive.

A triple any of whose three parts is empty after this is ``None`` — the
screens abstain on it, exactly as SPEC says for a null triple.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from ..extract.protocol import canonical, sha256_text
from .lexicon import ANTONYMS, CONTRACTIONS, FUNCTIONAL, MARKERS, NEGATION_LEXICON_VERSION

CONFLICT_RULES_VERSION = "v1"

#: Objects longer than this many tokens are clauses, not values: the value
#: screen abstains on them (measured: 37 % of the extractor's objects are ten
#: tokens or more — "the evolution of language is a complex and ...").
MAX_VALUE_TOKENS = 6

ARTICLES: tuple[str, ...] = ("a", "an", "the")
FIRST_PERSON: tuple[str, ...] = (
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "user",
    "this user", "the user",
)
AUXILIARIES: tuple[str, ...] = (
    "is", "are", "am", "was", "were", "be", "been", "being", "has", "have", "had", "will",
    "would", "shall", "should", "can", "could", "may", "might", "must", "do", "does", "did",
    "currently", "now", "still", "also", "just", "recently", "already", "always", "usually",
    "often", "really",
)
#: Words that qualify a number without changing which number it is
#: ("about 1300 followers" and "1300 followers" are one value).
APPROXIMATORS: tuple[str, ...] = (
    "about", "around", "roughly", "approximately", "nearly", "almost", "over", "under",
    "close", "to", "at", "least", "most", "up", "more", "than", "less", "only", "exactly",
    "just", "some", "currently", "now",
)
NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000", "single": "1", "couple": "2",
    "dozen": "12", "half": "0.5",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")

# Lookup tables derived from the lexicon, built once.
_MARKER_PHRASES = tuple(sorted((m for m in MARKERS if " " in m), key=len, reverse=True))
_MARKER_TOKENS = frozenset(m for m in MARKERS if " " not in m)
_ANTONYM_FORM: dict[str, tuple[str, int]] = {}
for _key, (_pos, _neg) in ANTONYMS.items():
    for _form in _pos:
        _ANTONYM_FORM[_form] = (_key, +1)
    for _form in _neg:
        _ANTONYM_FORM[_form] = (_key, -1)
_FUNCTIONAL_FORM: dict[str, str] = {
    form: key for key, forms in FUNCTIONAL.items() for form in forms
}
for _key in FUNCTIONAL:
    _FUNCTIONAL_FORM.setdefault(_key, _key)


class NormalizedTriple(NamedTuple):
    subject: str
    predicate: str
    object: str
    polarity: int  # +1 or -1
    functional: str | None  # the FUNCTIONAL group the predicate belongs to, or None

    @property
    def pair_key(self) -> str:
        return pair_key(self.subject, self.predicate)


def pair_key(subject: str, predicate: str) -> str:
    """The index key of a normalized ``(subject, predicate)`` — ``"user|lives in"``."""
    return f"{subject}|{predicate}"


def normalize_text(text: str) -> str:
    """Lower-case, apostrophes removed, punctuation stripped, whitespace collapsed,
    contractions expanded (``doesnt`` → ``does not``)."""
    bare = _PUNCT_RE.sub("", text.lower().replace("'", "").replace("’", "")).split()
    return " ".join(CONTRACTIONS.get(token, token) for token in bare)


def _drop_leading(tokens: list[str], words: tuple[str, ...]) -> list[str]:
    while len(tokens) > 1 and tokens[0] in words:
        tokens = tokens[1:]
    return tokens


def _strip_markers(text: str) -> tuple[str, int]:
    """Remove negation markers from normalized ``text``; return ``(residue, count)``."""
    padded = f" {text} "
    count = 0
    for phrase in _MARKER_PHRASES:
        needle = f" {phrase} "
        while needle in padded:
            padded = padded.replace(needle, "  ", 1)
            count += 1
    tokens = []
    for token in padded.split():
        if token in _MARKER_TOKENS:
            count += 1
        else:
            tokens.append(token)
    return " ".join(tokens), count


def normalize_subject(subject: str) -> str:
    text = " ".join(_drop_leading(normalize_text(subject).split(), ARTICLES))
    return "user" if text in FIRST_PERSON else text


def normalize_predicate(predicate: str) -> tuple[str, int, str | None]:
    """``(canonical predicate, polarity, functional group)`` for a raw predicate."""
    text = normalize_text(predicate)
    polarity = +1
    if text in _ANTONYM_FORM:  # the predicate IS an antonym form ("stopped", "dislikes")
        text, polarity = _ANTONYM_FORM[text]
    else:
        # Markers first ("is not" → "is", one marker), then the auxiliaries
        # ("does like" → "like"), then the antonym table on what is left
        # ("no longer likes" → "likes" → like, negative).
        residue, count = _strip_markers(text)
        if count % 2:
            polarity = -polarity
        text = " ".join(_drop_leading((residue or text).split(), AUXILIARIES))
        if text in _ANTONYM_FORM:
            text, side = _ANTONYM_FORM[text]
            polarity *= side
    functional = _FUNCTIONAL_FORM.get(text)
    if functional is not None:  # "moved to" and "lives in" share one pair key
        text = functional
    return text, polarity, functional


def normalize_object(obj: str) -> tuple[str, int]:
    """``(normalized object, marker count)``: digits for number words, no leading article."""
    tokens = [NUMBER_WORDS.get(t, t) for t in normalize_text(obj).split()]
    residue, count = _strip_markers(" ".join(tokens))
    return " ".join(_drop_leading(residue.split(), ARTICLES)), count


def normalize_triple(
    subject: str | None, predicate: str | None, obj: str | None
) -> NormalizedTriple | None:
    """The normalized triple, or ``None`` when any part is null or empties out."""
    if not subject or not predicate or not obj:
        return None
    s = normalize_subject(subject)
    p, polarity, functional = normalize_predicate(predicate)
    o, marker_count = normalize_object(obj)
    if not s or not p or not o:
        return None
    if marker_count % 2:
        polarity = -polarity
    return NormalizedTriple(s, p, o, polarity, functional)


def polarity_of_text(text: str) -> int:
    """Polarity of free text (the fallback when a fact has no triple): odd markers → -1."""
    _residue, count = _strip_markers(normalize_text(text))
    negative_forms = sum(
        1
        for form, (_key, side) in _ANTONYM_FORM.items()
        if side < 0 and f" {form} " in f" {normalize_text(text)} "
    )
    return -1 if (count + negative_forms) % 2 else +1


def value_tokens(obj: str) -> int:
    """Token count of a normalized object — the value-size test."""
    return len(obj.split())


def numeric_signature(obj: str) -> tuple[tuple[str, ...], str] | None:
    """``(numbers, residue)`` of a normalized object, or ``None`` without a number.

    "3 of emmas recipes" → ``(("3",), "of emmas recipes")`` after the
    approximators are dropped, so "about 1300 followers" and "1500 followers"
    share the residue ``followers`` and differ on the number.
    """
    tokens = [t for t in obj.split() if t not in APPROXIMATORS]
    numbers = tuple(t for t in tokens if _NUMBER_RE.match(t))
    if not numbers:
        return None
    return numbers, " ".join(t for t in tokens if not _NUMBER_RE.match(t))


def conflict_rules_hash() -> str:
    """Digest of every table the normalization and the conflict rules read."""
    return sha256_text(
        canonical(
            {
                "version": CONFLICT_RULES_VERSION,
                "negation_lexicon_version": NEGATION_LEXICON_VERSION,
                "articles": list(ARTICLES),
                "first_person": sorted(FIRST_PERSON),
                "auxiliaries": sorted(AUXILIARIES),
                "approximators": sorted(APPROXIMATORS),
                "number_words": dict(sorted(NUMBER_WORDS.items())),
                "functional": {key: sorted(forms) for key, forms in FUNCTIONAL.items()},
                "max_value_tokens": MAX_VALUE_TOKENS,
                "rules": ["negation", "functional", "numeric"],
                "ordering": ["effective_time", "created_at", "trust", "id"],
                "entropy": "token-shannon-bits",
            }
        )
    )
