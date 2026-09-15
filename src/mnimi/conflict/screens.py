"""The dedup screens over a cosine-gate-pass pair of facts (SPEC §Dedup strategy 3–5).

A fact that trips the session-scoped cosine gate is no longer dropped on
sight. In SPEC's order the pair is read through:

3. the **negation screen** — opposite polarity on the two sides routes the
   pair away from the merge (PHASE3 D4: polarity from the triple when it
   normalizes, from the fact text otherwise);
4. the **value-substitution screen** — the same normalized ``pair_key`` with
   different, value-sized objects is a substitution, not a restatement (D2);
5. the **entropy gate** — a pair whose shorter side has too few distinct
   tokens to trust a cosine on is kept apart (D5).

Only a pair that passes all three is a duplicate and merges (the incoming
record is dropped — the v1.9 outcome). Everything here is a pure function of
the two facts' stored fields and one threshold from ``MemoryConfig``; it
never touches a round record.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple

from .normalize import (
    MAX_VALUE_TOKENS,
    normalize_text,
    normalize_triple,
    polarity_of_text,
    value_tokens,
)

ACTION_MERGE = "merge"
ACTION_KEEP = "keep"
REASON_DUPLICATE = "duplicate"
KEEP_REASONS = ("negation", "value", "low-entropy")


class Verdict(NamedTuple):
    action: str  # "merge" | "keep"
    reason: str  # "duplicate" | "negation" | "value" | "low-entropy"


def token_entropy_bits(text: str) -> float:
    """Shannon entropy, in bits, of the token-type distribution of ``text``.

    Token-level on purpose: a sentence's character entropy sits near four
    bits whatever it says, while its token distribution counts the distinct
    words that make it identifiable. One token is 0.0; four distinct tokens
    are exactly 2.0 — SPEC's default gate.
    """
    tokens = normalize_text(text).split()
    if not tokens:
        return 0.0
    total = len(tokens)
    return -sum((n / total) * math.log2(n / total) for n in Counter(tokens).values())


def fact_polarity(fact: str, triple: tuple[str | None, str | None, str | None]) -> int:
    """The polarity the negation screen reads: the triple's, else the text's."""
    normalized = normalize_triple(*triple)
    if normalized is not None:
        return normalized.polarity
    return polarity_of_text(fact)


def screen_pair(
    fact: str,
    triple: tuple[str | None, str | None, str | None],
    neighbour_fact: str,
    neighbour_triple: tuple[str | None, str | None, str | None],
    entropy_gate: float,
) -> Verdict:
    """SPEC steps 3 → 4 → 5 over one cosine-gate-pass pair; ``merge`` only past all three."""
    if fact_polarity(fact, triple) != fact_polarity(neighbour_fact, neighbour_triple):
        return Verdict(ACTION_KEEP, "negation")
    incoming = normalize_triple(*triple)
    existing = normalize_triple(*neighbour_triple)
    if (
        incoming is not None
        and existing is not None
        and incoming.pair_key == existing.pair_key
        and incoming.object != existing.object
        and value_tokens(incoming.object) <= MAX_VALUE_TOKENS
        and value_tokens(existing.object) <= MAX_VALUE_TOKENS
    ):
        return Verdict(ACTION_KEEP, "value")
    if min(token_entropy_bits(fact), token_entropy_bits(neighbour_fact)) < entropy_gate:
        return Verdict(ACTION_KEEP, "low-entropy")
    return Verdict(ACTION_MERGE, REASON_DUPLICATE)
