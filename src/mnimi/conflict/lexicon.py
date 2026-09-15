"""The negation lexicon and the functional-predicate groups — frozen, hashed (PHASE3 D4).

Two lexicons, two hashes, two lifetimes:

* ``MARKERS`` and ``ANTONYMS`` decide a fact's *polarity* (SPEC §Dedup strategy
  step 3, the negation screen) and are hashed into ``memory_meta`` as
  ``negation_lexicon_hash`` — the row that carried the literal ``"none"``
  until this module existed.
* ``FUNCTIONAL`` names the predicates for which one subject holds ONE value at
  a time (residence, employer, occupation, origin, partner, ...). Only these
  make a same-pair, different-object fact a *conflict* rather than a second
  fact (the user has a cat AND a sister; the user lives in Boston OR Seattle).
  It is hashed with the normalization tables into ``conflict_rules_hash``
  (``.normalize``).

Every phrase is written in the normalized form ``.normalize`` produces:
lower-case, apostrophes removed (``doesn't`` → ``doesnt``), punctuation
stripped, single spaces. Forms are enumerated explicitly — inflections
included — because the normalization does no lemmatization (D3).

Judgment, disclosed: the lists were written by hand on 2026-09-15 with the
extraction cache's corpus-wide frequency tables in view (top predicates, the
count of negation markers inside fact texts, object lengths); no evidence
round and no question was read (DECISIONS "Phase 3 pre-registration").
The same measurement is why ``uses`` / ``using`` and ``has`` are NOT
functional: the corpus shows one user using many things.

Frozen: an edit here moves a guard hash and refuses every existing store.
"""

from __future__ import annotations

from ..extract.protocol import canonical, sha256_text

NEGATION_LEXICON_VERSION = "v1"

#: Contractions are expanded token by token BEFORE markers are counted, so
#: "doesn't like" becomes "does not like": the auxiliary survives for the
#: predicate ("does" is stripped as an auxiliary, "like" remains) and the one
#: ``not`` is the marker. ``cannot`` is spelled as one token by the model and
#: expands the same way.
CONTRACTIONS: dict[str, str] = {
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "doesnt": "does not",
    "dont": "do not",
    "didnt": "did not",
    "cant": "can not",
    "cannot": "can not",
    "couldnt": "could not",
    "wont": "will not",
    "wouldnt": "would not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "shouldnt": "should not",
    "mustnt": "must not",
    "aint": "is not",
    "neednt": "need not",
}

#: Polarity markers. A fact whose predicate/object (or, with no triple, whose
#: text) carries an ODD number of these is negative. Multi-word phrases are
#: matched before single tokens, longest first, so ``no longer`` counts once
#: and ``not anymore`` counts once (``anymore`` alone is deliberately absent —
#: "doesn't play tennis anymore" is one negation, not two). ``used to`` marks
#: a past state ("used to live in Boston"): negative for the standing fact.
#: Cessation verbs (``stopped``, ``quit``, ``gave up``) act as markers when
#: they PREFIX a predicate ("stopped going to the gym") and as antonym forms
#: when they ARE the predicate (see ``ANTONYMS``).
MARKERS: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "no longer",
    "not anymore",
    "no more",
    "used to",
    "unable to",
    "failed to",
    "fails to",
    "stopped",
    "stops",
    "stop",
    "quit",
    "quits",
    "gave up",
    "gives up",
    "give up",
    "given up",
    "ceased",
)

#: Antonym groups: ``canonical → (positive forms, negative forms)``. A
#: predicate equal to any listed form is rewritten to the canonical with the
#: polarity of its side. ``like`` folds love/enjoy with dislike/hate on
#: purpose — "likes jazz" and "hates jazz" must meet on one pair key. ``own``
#: folds the acquisition verbs with the disposal verbs ("bought X" vs "sold
#: X"). Verbs with a dominant non-polar reading (pass/fail, win/lose, drop)
#: are left out: "passed the salt", "lost my keys", "dropped my phone".
ANTONYMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "like": (
        (
            "like", "likes", "liked", "liking", "love", "loves", "loved", "loving",
            "enjoy", "enjoys", "enjoyed", "enjoying", "fond of", "into", "a fan of", "fan of",
            "big fan of", "a big fan of", "keen on", "prefers", "prefer", "preferred",
        ),
        (
            "dislike", "dislikes", "disliked", "disliking", "hate", "hates", "hated", "hating",
            "loathe", "loathes", "loathed", "detest", "detests", "detested", "can not stand",
            "not a fan of", "not fond of", "not into",
        ),
    ),
    "start": (
        (
            "start", "starts", "started", "starting", "begin", "begins", "began", "begun",
            "beginning", "take up", "takes up", "took up", "taking up", "taken up",
            "resume", "resumes", "resumed", "resuming",
        ),
        (
            "stop", "stops", "stopped", "stopping", "quit", "quits", "quitting", "gave up",
            "give up", "gives up", "given up", "giving up", "cease", "ceases", "ceased",
            "ceasing", "discontinue", "discontinues", "discontinued", "abandon", "abandons",
            "abandoned", "paused", "pauses", "pause",
        ),
    ),
    "hire": (
        ("hire", "hires", "hired", "hiring", "employ", "employs", "employed", "employing"),
        (
            "fire", "fires", "fired", "firing", "lay off", "lays off", "laid off", "let go",
            "lets go", "dismiss", "dismisses", "dismissed", "sack", "sacks", "sacked",
        ),
    ),
    "marry": (
        ("marry", "marries", "married", "marrying", "married to", "wed", "wedded"),
        (
            "divorce", "divorces", "divorced", "divorcing", "divorced from", "separated from",
            "split from", "split up with", "broke up with", "breaks up with", "break up with",
            "broken up with",
        ),
    ),
    "join": (
        (
            "join", "joins", "joined", "joining", "a member of", "member of", "signed up for",
            "sign up for", "signs up for", "signed up", "subscribed to", "subscribes to",
            "subscribe to", "enrolled in", "enrolls in", "enrol in", "enroll in",
        ),
        (
            "leave", "leaves", "left", "leaving", "unsubscribed from", "unsubscribe from",
            "unsubscribes from", "cancelled", "canceled", "cancels", "cancel", "withdrew from",
            "withdrawn from", "withdraw from", "dropped out of", "drops out of", "drop out of",
        ),
    ),
    "accept": (
        (
            "accept", "accepts", "accepted", "accepting", "agreed to", "agrees to", "agree to",
            "approve", "approves", "approved", "approving",
        ),
        (
            "reject", "rejects", "rejected", "rejecting", "decline", "declines", "declined",
            "declining", "turned down", "turns down", "turn down", "refuse", "refuses",
            "refused", "refusing", "disapprove", "disapproves", "disapproved",
        ),
    ),
    "agree": (
        ("agree", "agrees", "agreed", "agreeing", "agree with", "agrees with", "agreed with"),
        (
            "disagree", "disagrees", "disagreed", "disagreeing", "disagree with",
            "disagrees with", "disagreed with",
        ),
    ),
    "trust": (
        ("trust", "trusts", "trusted", "trusting"),
        ("distrust", "distrusts", "distrusted", "mistrust", "mistrusts", "mistrusted"),
    ),
    "enable": (
        (
            "enable", "enables", "enabled", "enabling", "turned on", "turn on", "turns on",
            "switched on", "switches on", "switch on", "activated", "activates", "activate",
        ),
        (
            "disable", "disables", "disabled", "disabling", "turned off", "turn off",
            "turns off", "switched off", "switches off", "switch off", "deactivated",
            "deactivates", "deactivate",
        ),
    ),
    "own": (
        (
            "own", "owns", "owned", "owning", "has", "have", "had", "has got", "have got", "got",
            "possess", "possesses", "possessed", "acquired", "acquires", "acquire", "bought",
            "buys", "buy", "buying", "purchased", "purchases", "purchase", "purchasing",
            "adopted", "adopts", "adopt",
        ),
        (
            "sold", "sells", "sell", "selling", "gave away", "give away", "gives away",
            "giving away", "given away", "got rid of", "gets rid of", "get rid of", "threw away",
            "throws away", "threw out", "throws out", "donated", "donates", "donate", "returned",
            "rehomed", "rehome", "rehomes",
        ),
    ),
}

#: Functional predicates: ``canonical → forms``. A form is matched against the
#: whole predicate AFTER auxiliary stripping and antonym canonicalization, so
#: "is married to" reaches ``partner`` through ``marry``. Two facts of one
#: subject on one of these groups with different value-sized objects conflict
#: (PHASE3 D2); on any other predicate they are two facts.
FUNCTIONAL: dict[str, tuple[str, ...]] = {
    "lives in": (
        "lives in", "live in", "living in", "lived in", "lives at", "live at", "living at",
        "lived at", "resides in", "reside in", "residing in", "resided in", "resides at",
        "based in", "located in", "stays in", "staying in", "moved to", "moves to", "moving to",
        "moved into", "moving into", "relocated to", "relocates to", "relocating to",
        "settled in", "settling in", "address is", "home address is", "zip code is",
        "postal code is", "zip is",
    ),
    "works at": (
        "works at", "work at", "working at", "worked at", "works for", "work for",
        "working for", "worked for", "employed at", "employed by", "employer is", "company is",
        "works with", "working with",
    ),
    "works as": (
        "works as", "work as", "working as", "worked as", "job is", "occupation is",
        "profession is", "employed as", "role is", "title is", "job title is", "position is",
        "career is",
    ),
    "is from": (
        "from", "comes from", "come from", "hails from", "originally from", "grew up in",
        "grows up in", "growing up in", "born in", "raised in", "native of", "a native of",
        "nationality is", "originally hails from", "hometown is",
    ),
    "drives": ("drives", "drive", "driving", "car is", "vehicle is"),
    "partner": (
        "marry", "dating", "dates", "engaged to", "in a relationship with", "partner is",
        "spouse is", "husband is", "wife is", "girlfriend is", "boyfriend is", "seeing",
        "going out with",
    ),
    "name is": (
        "name is", "goes by", "nickname is", "full name is", "first name is", "last name is",
    ),
    "weighs": ("weighs", "weigh", "weighing", "weighed", "weight is"),
    "studies at": (
        "studies at", "studying at", "studied at", "attends", "attending", "enrolled at",
        "goes to school at", "school is", "university is", "college is", "student at",
        "a student at",
    ),
    "age is": ("age is", "aged"),
    "height is": ("height is",),
    "phone number is": ("phone number is", "phone is", "mobile number is", "cell number is"),
    "email is": ("email is", "email address is", "e mail is"),
    "birthday is": (
        "birthday is", "born on", "date of birth is", "dob is", "birthdate is", "birth date is",
    ),
}


def negation_lexicon_hash() -> str:
    """Digest of the sorted markers and antonym groups (``memory_meta``)."""
    return sha256_text(
        canonical(
            {
                "version": NEGATION_LEXICON_VERSION,
                "contractions": dict(sorted(CONTRACTIONS.items())),
                "markers": sorted(MARKERS),
                "antonyms": {
                    key: {"positive": sorted(pos), "negative": sorted(neg)}
                    for key, (pos, neg) in ANTONYMS.items()
                },
            }
        )
    )
