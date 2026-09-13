"""Stage 1 — the deterministic speech-act pre-filter (SPEC §Extraction).

A screen, not a classifier: it rejects turns that cannot contain a stored
fact before any model call, and it errs towards keeping. False negatives
(junk that slips through) cost one model call and come back as ``[]``; a
false positive (a real fact dropped) is the only correctness risk, so every
rule is narrow and short-text only. SPEC's "question → drop" rule is
deliberately NOT here: on LongMemEval the evidence is a question carrying an
aside ("Can you suggest tips? By the way, I've been …"), and the false-drop
check on the slice's evidence rounds must read 0 (PHASE2 D9).

Every drop is logged as ``filtered: {rule}`` on the ``mnimi.extract`` logger.
The lexicon and the rule identifiers are hashed into ``memory_meta`` as
``prefilter_lexicon_hash``: an edit is a versioned migration plus re-ingest,
never a hot fix under a run.
"""

from __future__ import annotations

import logging
import re

from .protocol import canonical, sha256_text

log = logging.getLogger("mnimi.extract")

PREFILTER_VERSION = "v1"

#: Phrase classes. Matched on the normalised text (lower-case, punctuation
#: stripped, whitespace collapsed), as whole words.
LEXICON: dict[str, tuple[str, ...]] = {
    "ack": (
        "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty", "cheers",
        "great", "awesome", "perfect", "got it", "okay", "ok", "sounds good", "cool", "nice",
        "noted", "understood", "that helps", "thats helpful", "makes sense", "will do",
        "appreciate it", "sure", "alright", "good to know", "that works", "fair enough",
    ),
    "greeting": (
        "hi", "hello", "hey", "hi there", "hello there", "hey there", "good morning",
        "good afternoon", "good evening", "howdy", "yo", "greetings",
    ),
    "closing": (
        "bye", "goodbye", "see you", "see ya", "talk later", "talk to you later",
        "have a good one", "have a great day", "take care", "good night", "later",
    ),
    "assistant_boilerplate": (
        "youre welcome", "you are welcome", "happy to help", "glad i could help",
        "glad to help", "let me know if", "feel free to ask", "anything else",
        "is there anything else", "how can i help", "how can i assist", "im here to help",
        "my pleasure", "no problem", "anytime", "dont hesitate", "hope this helps",
        "hope that helps", "if you have any other questions", "if you need anything else",
    ),
    "memory_self_reference": (
        "remember that", "remember this", "keep that in mind", "note that down", "save that",
        "forget that", "forget this", "what did i say", "what did you say", "your last answer",
        "your previous answer", "that statement you made", "as i said before", "as you said",
        "what did i tell you", "do you remember",
    ),
    "imperative": (
        "tell", "give", "show", "explain", "describe", "list", "summarize", "summarise",
        "write", "translate", "repeat", "elaborate", "expand", "continue", "go on", "rephrase",
        "simplify", "clarify", "help",
    ),
    "filler": (
        "so", "much", "very", "really", "a", "lot", "that", "this", "is", "was", "it", "for",
        "the", "help", "info", "information", "all", "your", "you", "of", "and", "too",
        "again", "then", "well", "now", "just", "yes", "yeah", "yep", "no", "nope", "oh",
        "ah", "hmm", "right", "good", "fine", "indeed", "totally", "absolutely", "exactly",
        "definitely", "please", "pls", "man", "dude", "friend",
    ),
    "first_person": ("i", "im", "ive", "id", "ill", "me", "my", "mine", "we", "were", "weve",
                     "our", "ours", "us", "myself"),
    "date_words": (
        "today", "yesterday", "tomorrow", "tonight", "week", "weeks", "month", "months",
        "year", "years", "day", "days", "ago", "last", "next", "recently", "monday",
        "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
        "february", "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "weekend", "morning", "evening", "afternoon",
    ),
}

RULE_IDS = (
    "ack-only",
    "greeting-only",
    "memory-self-reference",
    "imperative-to-assistant",
    "no-candidate-slot",
    "assistant-boilerplate",
)

_SHORT_WORDS = 12       # `no-candidate-slot` never looks past this many words
_SELF_REF_WORDS = 8     # `memory-self-reference` and `imperative-to-assistant` cap
_BOILERPLATE_WORDS = 40  # a longer assistant turn is an answer, whatever it opens with
_ANAPHORA = ("that", "this", "it", "more", "again", "them", "those", "these", "one",
             "another", "me", "us", "please", "pls")

_PUNCT_RE = re.compile(r"[^\w\s]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIGIT_RE = re.compile(r"\d")
_QUESTION_LEADS = frozenset(
    "what why how when where who whom whose which can could would will should shall do does "
    "did is are was were have has had any anything may might".split()
)


def _normalize(text: str) -> str:
    return " ".join(_PUNCT_RE.sub("", text.lower().replace("'", "")).split())


def _words(text: str) -> list[str]:
    return _normalize(text).split()


def _has_phrase(normalized: str, phrase_class: str) -> bool:
    padded = f" {normalized} "
    return any(f" {phrase} " in padded for phrase in LEXICON[phrase_class])


def _residue(normalized: str, *phrase_classes: str) -> list[str]:
    """Words left after removing the phrase classes (longest first) and fillers."""
    padded = f" {normalized} "
    phrases = sorted((p for c in phrase_classes for p in LEXICON[c]), key=len, reverse=True)
    for phrase in phrases:
        padded = padded.replace(f" {phrase} ", "  ")
    return [w for w in padded.split() if w not in LEXICON["filler"]]


def _has_entity_cue(text: str) -> bool:
    """A capitalised word that is not sentence-initial and not the pronoun I."""
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        tokens = sentence.split()
        for token in tokens[1:]:
            bare = _PUNCT_RE.sub("", token.replace("'", ""))
            if bare and bare[0].isupper() and bare not in {"I", "Im", "Ive", "Id", "Ill"}:
                return True
    return False


def _has_slot_cue(text: str, words: list[str]) -> bool:
    if _DIGIT_RE.search(text) or _has_entity_cue(text):
        return True
    return any(w in LEXICON["first_person"] or w in LEXICON["date_words"] for w in words)


def _is_question(text: str, words: list[str]) -> bool:
    """A question is never slotless by this screen: SPEC's question rule is not
    built (PHASE2 D9), and a bare question costs one model call at most."""
    if text.rstrip().endswith("?"):
        return True
    lead = next((w for w in words if w not in LEXICON["filler"]), None)
    return lead in _QUESTION_LEADS


def _is_imperative_to_assistant(words: list[str]) -> bool:
    """`explain that again`, `can you show me more`, `tell me` — nothing but a
    verb from the lexicon, an optional polite frame and anaphora."""
    rest = list(words)
    for frame in (("please",), ("can", "you"), ("could", "you"), ("would", "you"),
                  ("will", "you"), ("can", "you", "please"), ("could", "you", "please")):
        if tuple(rest[: len(frame)]) == frame:
            rest = rest[len(frame):]
            break
    if not rest or rest[0] not in LEXICON["imperative"]:
        return False
    return all(w in _ANAPHORA for w in rest[1:])


def decide(turn: dict) -> str | None:
    """The rule that drops ``turn``, or ``None`` when it may carry a fact."""
    role = str(turn.get("role", ""))
    text = str(turn.get("content", ""))
    normalized = _normalize(text)
    words = normalized.split()
    if not words:
        return "no-candidate-slot"
    if _has_phrase(normalized, "ack") and not _residue(normalized, "ack", "greeting",
                                                        "closing"):
        return "ack-only"
    if _has_phrase(normalized, "greeting") and not _residue(normalized, "greeting",
                                                             "closing", "ack"):
        return "greeting-only"
    if role == "user" and len(words) <= _SELF_REF_WORDS:
        if _has_phrase(normalized, "memory_self_reference"):
            return "memory-self-reference"
        if _is_imperative_to_assistant(words):
            return "imperative-to-assistant"
    if (
        role == "assistant"
        and len(words) <= _BOILERPLATE_WORDS
        and _has_phrase(normalized, "assistant_boilerplate")
        and not _DIGIT_RE.search(text)
        and not _has_entity_cue(text)
    ):
        return "assistant-boilerplate"
    if (
        len(words) <= _SHORT_WORDS
        and not _is_question(text, words)
        and not _has_slot_cue(text, words)
    ):
        return "no-candidate-slot"
    return None


def keep_round(turns: list[dict]) -> tuple[bool, list[str]]:
    """``(send this round to the model?, rule ids of the dropped turns)``.

    A round reaches the model iff at least one turn survives; the whole round
    is then the model's input (the dropped turn travels as context). Every
    drop is logged with the rule that fired.
    """
    dropped: list[str] = []
    kept = 0
    for turn in turns:
        rule = decide(turn)
        if rule is None:
            kept += 1
        else:
            dropped.append(rule)
            log.info("filtered: %s", rule)
    return kept > 0, dropped


def prefilter_lexicon_hash() -> str:
    """Digest of the sorted lexicon and the rule identifiers (`memory_meta`)."""
    return sha256_text(
        canonical(
            {
                "version": PREFILTER_VERSION,
                "lexicon": {name: sorted(phrases) for name, phrases in LEXICON.items()},
                "rules": list(RULE_IDS),
                "limits": {
                    "short_words": _SHORT_WORDS,
                    "self_ref_words": _SELF_REF_WORDS,
                    "boilerplate_words": _BOILERPLATE_WORDS,
                },
            }
        )
    )
