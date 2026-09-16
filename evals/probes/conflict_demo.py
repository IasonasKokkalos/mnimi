"""The conflict demo set and gate 3-ii: where the write-path thesis is falsified (PHASE3 D10).

LongMemEval's history is built non-conflicting (46 supersessions in 54,515
stored facts on the slice, PHASE3-RESULTS § Task 3), so the slice cannot
falsify supersession. This module can. ``generate(seed)`` builds **100
authored pairs** from frozen template tables — value-change 30 (15
``functional``, 15 ``count``), negation 30 (15 ``marker``, 15 ``antonym``),
dated-update 20 (10 ``in-order``, 10 ``reversed``), and two controls that must
NOT conflict: paraphrase 10 and unrelated 10. Each pair is two rounds with
LongMemEval-shaped session timestamps and pre-authored ``ExtractedFact``s
returned by ``ScriptedExtractor``, so the whole write path — pre-filter,
resolver, both screens, supersession — runs end to end with no model, no GPU
and no API, one fresh ``Memory`` per pair.

What "correct" means (D10, verbatim):

* a conflict pair — after ``add(round_a)`` and ``add(round_b)`` exactly one
  ACTIVE fact (``salience > 0``) carries the pair's ``pair_key``, its
  normalized object is the expected value (negation: the expected polarity on
  the expected object), the loser has ``salience == 0`` and the winner's
  ``supersedes`` points at it — and the same holds after ``consolidate()``;
* paraphrase — nothing has ``salience == 0`` and every active fact carries
  the expected object (merged or kept apart, never superseded);
* unrelated — two active facts, none superseded.

The baseline is the same set through ``MemoryConfig(conflict_resolution=False)``
— the v1.9 write path, screens off, no supersede (D11). The pre-registered
criterion (``criterion``) is mnimi ≥ 72/80 on the conflict families and
≥ 27/30, ≥ 27/30, ≥ 18/20 within them, controls 20/20, on both readings, and
strictly above the baseline on conflicts. The exit code of ``main`` is the
gate; the JSON file is the artifact. A ``--extractor qwen3`` pass runs the
natural-language rounds through the real extractor and reports how many
pairs the model's own triples made conflict-able — descriptive, never the
gate (it would measure the 1.7B model's triples, not the screens).

Deterministic: ``random.Random(seed)`` over frozen tables, so ``generate(0)``
is byte-identical across runs (a test asserts it). No wall-clock anywhere;
every threshold is the library default read from ``MemoryConfig``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from mnimi import Memory, MemoryConfig
from mnimi.conflict.normalize import normalize_object, normalize_triple
from mnimi.embeddings import HashingEmbedder
from mnimi.extract.fake import ScriptedExtractor
from mnimi.extract.protocol import ExtractedFact
from mnimi.memory import round_key

FAMILIES = ("value-change", "negation", "dated-update", "paraphrase", "unrelated")
SIZES = {"value-change": 30, "negation": 30, "dated-update": 20, "paraphrase": 10, "unrelated": 10}
CONFLICT_FAMILIES = ("value-change", "negation", "dated-update")
CONTROL_FAMILIES = ("paraphrase", "unrelated")

#: The pre-registered criterion (D10), as numbers.
MIN_CONFLICTS_CORRECT = 72
MIN_PER_FAMILY = {"value-change": 27, "negation": 27, "dated-update": 18}
CONTROLS_REQUIRED = 20

USER_ID = "demo"

# ---------------------------------------------------------------------------
# Frozen template tables. Every value is drawn by ``random.Random(seed)``; the
# tables never change under a run.

CITIES = (
    "Boston", "Seattle", "Denver", "Austin", "Chicago", "Portland", "Atlanta", "Phoenix",
    "Nashville", "Minneapolis", "Detroit", "Baltimore", "Sacramento", "Pittsburgh", "Tucson",
    "Raleigh", "Omaha", "Tampa", "Cleveland", "Milwaukee",
)
EMPLOYERS = (
    "Acme Corp", "Globex", "Initech", "Umbrella Labs", "Vandelay Industries", "Stark Systems",
    "Wayne Logistics", "Hooli", "Pied Piper", "Soylent Foods", "Cyberdyne", "Wonka Industries",
)
OCCUPATIONS = (
    "nurse", "teacher", "plumber", "barista", "paralegal", "carpenter", "pharmacist",
    "librarian", "welder", "dentist", "translator", "firefighter",
)
CARS = (
    "Toyota Prius", "Honda Civic", "Ford Focus", "Subaru Outback", "Mazda 3", "Kia Soul",
    "Volvo XC40", "Nissan Leaf", "Jeep Wrangler", "Chevy Bolt",
)
PARTNERS = ("Anna", "Maria", "Priya", "Chloe", "Daniel", "Marcus", "Elena", "Tom", "Yuki", "Omar")
SCHOOLS = (
    "MIT", "Stanford", "Georgia Tech", "Purdue", "UCLA", "Rice", "Cornell", "Northwestern",
)
COUNTABLES = (
    "cats", "bikes", "houseplants", "guitars", "tattoos", "credit cards", "laptops",
    "chickens", "bookshelves", "siblings",
)
RECIPE_SOURCES = ("Emma", "Nonna", "Chef Lee", "Ottolenghi", "Aunt Rosa")
HOBBIES = (
    "hiking", "yoga", "jazz", "chess", "pottery", "running", "birdwatching", "salsa dancing",
    "rock climbing", "gardening", "baking", "surfing",
)
POSSESSIONS = (
    "Kindle", "telescope", "sewing machine", "treadmill", "record player", "kayak", "drone",
    "piano", "greenhouse", "espresso machine", "road bike", "camper van",
)
FOODS = ("gluten", "dairy", "peanuts", "shellfish", "sushi", "spicy food", "red meat", "eggs")
FOLLOWER_COUNTS = (300, 800, 1300, 1500, 2500, 4000, 6000, 9000)
YEARS = (2015, 2016, 2017, 2018, 2019, 2020)
#: The three dated mentions of a "moved" sentence; all resolve after every
#: ``YEARS`` entry against every session below, so the valid_time rule alone
#: decides the dated pairs.
MOVE_MENTIONS = ("last month", "two weeks ago", "on March 3rd")

#: LongMemEval-shaped session timestamps: an early and a late session, so the
#: cross-session pairs never meet on the session-scoped cosine gate and the
#: pair index (D2) is what finds them.
SESSIONS_EARLY = ("2023/01/10 (Tue) 09:00", "2023/02/14 (Tue) 18:30", "2023/03/10 (Fri) 12:00")
SESSIONS_LATE = ("2023/06/10 (Sat) 09:00", "2023/07/22 (Sat) 20:15", "2023/09/05 (Tue) 08:45")
ASSISTANT_REPLIES = (
    "Got it, thanks for letting me know.",
    "Noted.",
    "Thanks, I'll keep that in mind.",
    "Understood.",
)


@dataclass
class DemoPair:
    id: str  # "value-change/functional/07"
    family: str
    subfamily: str  # functional | count | marker | antonym | in-order | reversed |
    #                 paraphrase | unrelated
    ts_a: str
    ts_b: str
    turns_a: list[dict]
    turns_b: list[dict]
    facts_a: list[ExtractedFact]
    facts_b: list[ExtractedFact]
    pair_key: str
    expected_object: str  # normalized
    expected_polarity: int
    expect_supersede: bool


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def _fact(content: str, sentence: str, predicate: str, obj: str, when: str | None = None,
          ) -> ExtractedFact:
    """One authored fact: the user's own words as ``raw``, the triple as the screens read it."""
    return ExtractedFact(content=content, raw="user: " + sentence, when=when, subject="user",
                         predicate=predicate, object=obj, salience=1.0)


def _turns(rng: random.Random, sentence: str) -> list[dict]:
    return [{"role": "user", "content": sentence},
            {"role": "assistant", "content": rng.choice(ASSISTANT_REPLIES)}]


class _Builder:
    """Draws from the tables in one fixed order so the set is a function of the seed."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.pairs: list[DemoPair] = []
        self._counter: dict[tuple[str, str], int] = {}

    def add(self, family: str, subfamily: str, *, sentence_a: str, sentence_b: str,
            fact_a: ExtractedFact, fact_b: ExtractedFact, pair_key: str, expected_object: str,
            expected_polarity: int, expect_supersede: bool, same_session: bool = False) -> None:
        n = self._counter.get((family, subfamily), 0) + 1
        self._counter[(family, subfamily)] = n
        ts_a = self.rng.choice(SESSIONS_EARLY)
        ts_b = ts_a if same_session else self.rng.choice(SESSIONS_LATE)
        self.pairs.append(DemoPair(
            id=f"{family}/{subfamily}/{n:02d}", family=family, subfamily=subfamily,
            ts_a=ts_a, ts_b=ts_b,
            turns_a=_turns(self.rng, sentence_a), turns_b=_turns(self.rng, sentence_b),
            facts_a=[fact_a], facts_b=[fact_b], pair_key=pair_key,
            expected_object=normalize_object(expected_object)[0],
            expected_polarity=expected_polarity, expect_supersede=expect_supersede,
        ))


def _value_change(b: _Builder) -> None:
    rng = b.rng
    # 15 functional: residence 3, employer 3, occupation 3, vehicle 2, partner 2, school 2.
    for va, vb in zip(*[iter(rng.sample(CITIES, 6))] * 2, strict=True):
        b.add("value-change", "functional",
              sentence_a=f"I live in {va} now.", sentence_b=f"I live in {vb} now.",
              fact_a=_fact(f"The user lives in {va}.", f"I live in {va} now.", "lives in", va),
              fact_b=_fact(f"The user lives in {vb}.", f"I live in {vb} now.", "lives in", vb),
              pair_key="user|lives in", expected_object=vb, expected_polarity=+1,
              expect_supersede=True)
    for va, vb in zip(*[iter(rng.sample(EMPLOYERS, 6))] * 2, strict=True):
        b.add("value-change", "functional",
              sentence_a=f"I work at {va}.", sentence_b=f"I work at {vb}.",
              fact_a=_fact(f"The user works at {va}.", f"I work at {va}.", "works at", va),
              fact_b=_fact(f"The user works at {vb}.", f"I work at {vb}.", "works at", vb),
              pair_key="user|works at", expected_object=vb, expected_polarity=+1,
              expect_supersede=True)
    for va, vb in zip(*[iter(rng.sample(OCCUPATIONS, 6))] * 2, strict=True):
        oa, ob = f"{_article(va)} {va}", f"{_article(vb)} {vb}"
        b.add("value-change", "functional",
              sentence_a=f"I work as {oa}.", sentence_b=f"I work as {ob}.",
              fact_a=_fact(f"The user works as {oa}.", f"I work as {oa}.", "works as", oa),
              fact_b=_fact(f"The user works as {ob}.", f"I work as {ob}.", "works as", ob),
              pair_key="user|works as", expected_object=ob, expected_polarity=+1,
              expect_supersede=True)
    for va, vb in zip(*[iter(rng.sample(CARS, 4))] * 2, strict=True):
        b.add("value-change", "functional",
              sentence_a=f"I drive a {va}.", sentence_b=f"I drive a {vb}.",
              fact_a=_fact(f"The user drives a {va}.", f"I drive a {va}.", "drives", f"a {va}"),
              fact_b=_fact(f"The user drives a {vb}.", f"I drive a {vb}.", "drives", f"a {vb}"),
              pair_key="user|drives", expected_object=vb, expected_polarity=+1,
              expect_supersede=True)
    for va, vb in zip(*[iter(rng.sample(PARTNERS, 4))] * 2, strict=True):
        b.add("value-change", "functional",
              sentence_a=f"I'm dating {va}.", sentence_b=f"I'm dating {vb}.",
              fact_a=_fact(f"The user is dating {va}.", f"I'm dating {va}.", "is dating", va),
              fact_b=_fact(f"The user is dating {vb}.", f"I'm dating {vb}.", "is dating", vb),
              pair_key="user|partner", expected_object=vb, expected_polarity=+1,
              expect_supersede=True)
    for va, vb in zip(*[iter(rng.sample(SCHOOLS, 4))] * 2, strict=True):
        b.add("value-change", "functional",
              sentence_a=f"I study at {va}.", sentence_b=f"I study at {vb}.",
              fact_a=_fact(f"The user studies at {va}.", f"I study at {va}.", "studies at", va),
              fact_b=_fact(f"The user studies at {vb}.", f"I study at {vb}.", "studies at", vb),
              pair_key="user|studies at", expected_object=vb, expected_polarity=+1,
              expect_supersede=True)
    # 15 count: countables 7, recipes 4, followers 4.
    for noun in rng.sample(COUNTABLES, 7):
        na, nb = rng.sample(range(2, 10), 2)
        b.add("value-change", "count",
              sentence_a=f"I have {na} {noun}.", sentence_b=f"I have {nb} {noun}.",
              fact_a=_fact(f"The user has {na} {noun}.", f"I have {na} {noun}.", "has",
                           f"{na} {noun}"),
              fact_b=_fact(f"The user has {nb} {noun}.", f"I have {nb} {noun}.", "has",
                           f"{nb} {noun}"),
              pair_key="user|own", expected_object=f"{nb} {noun}", expected_polarity=+1,
              expect_supersede=True)
    for source in rng.sample(RECIPE_SOURCES, 4):
        na, nb = rng.sample(range(1, 7), 2)
        sa = f"I've tried {na} of {source}'s recipes."
        sb = f"I've tried {nb} of {source}'s recipes."
        b.add("value-change", "count", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user has tried {na} of {source}'s recipes.", sa, "has tried",
                           f"{na} of {source}'s recipes"),
              fact_b=_fact(f"The user has tried {nb} of {source}'s recipes.", sb, "has tried",
                           f"{nb} of {source}'s recipes"),
              pair_key="user|tried", expected_object=f"{nb} of {source}'s recipes",
              expected_polarity=+1, expect_supersede=True)
    for na, nb in zip(*[iter(rng.sample(FOLLOWER_COUNTS, 8))] * 2, strict=True):
        sa, sb = f"I have about {na} followers.", f"I have {nb} followers now."
        b.add("value-change", "count", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user has about {na} followers.", sa, "has",
                           f"about {na} followers"),
              fact_b=_fact(f"The user has {nb} followers.", sb, "has", f"{nb} followers"),
              pair_key="user|own", expected_object=f"{nb} followers", expected_polarity=+1,
              expect_supersede=True)


def _negation(b: _Builder) -> None:
    rng = b.rng
    # 15 marker: five templates, three draws each. The negated side carries a
    # lexicon MARKER ("no longer", "can't", "not", "never"); the predicate
    # form is otherwise the same token after auxiliary stripping.
    for v in rng.sample(CITIES, 3):
        sa, sb = f"These days I live in {v}.", f"I no longer live in {v}."
        b.add("negation", "marker", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user lives in {v}.", sa, "live in", v),
              fact_b=_fact(f"The user no longer lives in {v}.", sb, "no longer live in", v),
              pair_key="user|lives in", expected_object=v, expected_polarity=-1,
              expect_supersede=True)
    for v in rng.sample(FOODS, 3):
        sa, sb = f"I can eat {v}.", f"I can't eat {v} anymore."
        b.add("negation", "marker", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user can eat {v}.", sa, "can eat", v),
              fact_b=_fact(f"The user can't eat {v} anymore.", sb, "can't eat", v),
              pair_key="user|eat", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(POSSESSIONS, 3):
        art = _article(v)
        sa, sb = f"I own {art} {v}.", f"I no longer own {art} {v}."
        b.add("negation", "marker", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user owns {art} {v}.", sa, "own", f"{art} {v}"),
              fact_b=_fact(f"The user no longer owns {art} {v}.", sb, "no longer own",
                           f"{art} {v}"),
              pair_key="user|own", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(HOBBIES, 3):
        sa, sb = f"I'm into {v}.", f"I'm not into {v} anymore."
        b.add("negation", "marker", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user is into {v}.", sa, "am into", v),
              fact_b=_fact(f"The user is not into {v} anymore.", sb, "am not into", v),
              pair_key="user|like", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(FOODS, 3):
        sa, sb = f"I eat {v} regularly.", f"I never eat {v}."
        b.add("negation", "marker", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user eats {v}.", sa, "eat", v),
              fact_b=_fact(f"The user never eats {v}.", sb, "never eat", v),
              pair_key="user|eat", expected_object=v, expected_polarity=-1, expect_supersede=True)
    # 15 antonym: five ANTONYMS groups, three draws each; no marker token
    # anywhere — the lexicon's antonym side alone decides the polarity.
    for v in rng.sample(HOBBIES, 3):
        b.add("negation", "antonym", sentence_a=f"I love {v}.", sentence_b=f"I hate {v}.",
              fact_a=_fact(f"The user loves {v}.", f"I love {v}.", "love", v),
              fact_b=_fact(f"The user hates {v}.", f"I hate {v}.", "hate", v),
              pair_key="user|like", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(HOBBIES, 3):
        b.add("negation", "antonym", sentence_a=f"I started {v}.", sentence_b=f"I stopped {v}.",
              fact_a=_fact(f"The user started {v}.", f"I started {v}.", "started", v),
              fact_b=_fact(f"The user stopped {v}.", f"I stopped {v}.", "stopped", v),
              pair_key="user|start", expected_object=v, expected_polarity=-1,
              expect_supersede=True)
    for v in rng.sample(EMPLOYERS, 3):
        b.add("negation", "antonym", sentence_a=f"I joined {v}.", sentence_b=f"I left {v}.",
              fact_a=_fact(f"The user joined {v}.", f"I joined {v}.", "joined", v),
              fact_b=_fact(f"The user left {v}.", f"I left {v}.", "left", v),
              pair_key="user|join", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(CARS, 3):
        sa, sb = f"I bought a {v}.", f"I sold the {v}."
        b.add("negation", "antonym", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user bought a {v}.", sa, "bought", f"a {v}"),
              fact_b=_fact(f"The user sold the {v}.", sb, "sold", f"the {v}"),
              pair_key="user|own", expected_object=v, expected_polarity=-1, expect_supersede=True)
    for v in rng.sample(PARTNERS, 3):
        b.add("negation", "antonym", sentence_a=f"I trust {v}.", sentence_b=f"I distrust {v}.",
              fact_a=_fact(f"The user trusts {v}.", f"I trust {v}.", "trust", v),
              fact_b=_fact(f"The user distrusts {v}.", f"I distrust {v}.", "distrust", v),
              pair_key="user|trust", expected_object=v, expected_polarity=-1,
              expect_supersede=True)


def _lived(city: str, year: int) -> tuple[str, ExtractedFact]:
    s = f"Back in {year} I lived in {city}."
    return s, _fact(f"The user lived in {city} in {year}.", s, "lived in", city, when=f"in {year}")


def _moved(city: str, mention: str) -> tuple[str, ExtractedFact]:
    s = f"I moved to {city} {mention}."
    return s, _fact(f"The user moved to {city} {mention}.", s, "moved to", city, when=mention)


def _worked(employer: str, year: int) -> tuple[str, ExtractedFact]:
    s = f"Back in {year} I worked at {employer}."
    return s, _fact(f"The user worked at {employer} in {year}.", s, "worked at", employer,
                    when=f"in {year}")


def _working(employer: str) -> tuple[str, ExtractedFact]:
    s = f"I've been working at {employer} since last month."
    return s, _fact(f"The user has been working at {employer} since last month.", s,
                    "have been working at", employer, when="since last month")


def _dated_update(b: _Builder) -> None:
    rng = b.rng
    cities = iter(rng.sample(CITIES, 20))
    employers = iter(rng.sample(EMPLOYERS, 12))
    # in-order: the later-valid fact is also the later session (residence 5, employer 5).
    for _ in range(5):
        ca, cb = next(cities), next(cities)
        sa, fa = _lived(ca, rng.choice(YEARS))
        sb, fb = _moved(cb, rng.choice(MOVE_MENTIONS))
        b.add("dated-update", "in-order", sentence_a=sa, sentence_b=sb, fact_a=fa, fact_b=fb,
              pair_key="user|lives in", expected_object=cb, expected_polarity=+1,
              expect_supersede=True)
    for _ in range(5):
        ea, eb = next(employers), next(employers)
        sa, fa = _worked(ea, rng.choice(YEARS))
        sb, fb = _working(eb)
        b.add("dated-update", "in-order", sentence_a=sa, sentence_b=sb, fact_a=fa, fact_b=fb,
              pair_key="user|works at", expected_object=eb, expected_polarity=+1,
              expect_supersede=True)
    # reversed: the later-valid fact is ingested FIRST, from the earlier
    # session; only the valid_time rule gets it right (residence 5, employer 5;
    # the employer framings differ from in-order's so no user turn repeats).
    for _ in range(5):
        ca, cb = next(cities), next(cities)
        sa, fa = _moved(ca, rng.choice(MOVE_MENTIONS))
        sb, fb = _lived(cb, rng.choice(YEARS))
        b.add("dated-update", "reversed", sentence_a=sa, sentence_b=sb, fact_a=fa, fact_b=fb,
              pair_key="user|lives in", expected_object=ca, expected_polarity=+1,
              expect_supersede=True)
    for ea, eb in zip(*[iter(rng.sample(EMPLOYERS, 10))] * 2, strict=True):
        sa = f"I've been employed at {ea} since last month."
        fa = _fact(f"The user has been employed at {ea} since last month.", sa,
                   "have been employed at", ea, when="since last month")
        year = rng.choice(YEARS)
        sb = f"I was working at {eb} back in {year}."
        fb = _fact(f"The user was working at {eb} in {year}.", sb, "was working at", eb,
                   when=f"in {year}")
        b.add("dated-update", "reversed", sentence_a=sa, sentence_b=sb, fact_a=fa, fact_b=fb,
              pair_key="user|works at", expected_object=ea, expected_polarity=+1,
              expect_supersede=True)


def _paraphrase(b: _Builder) -> None:
    rng = b.rng
    # Two forms of one functional group, one object, one session: must never conflict.
    for v in rng.sample(CITIES, 4):
        sa, sb = f"I live in {v}.", f"I'm living in {v} these days."
        b.add("paraphrase", "paraphrase", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user lives in {v}.", sa, "live in", v),
              fact_b=_fact(f"The user is living in {v}.", sb, "am living in", v),
              pair_key="user|lives in", expected_object=v, expected_polarity=+1,
              expect_supersede=False, same_session=True)
    for v in rng.sample(EMPLOYERS, 3):
        sa, sb = f"I work for {v}.", f"I'm working for {v} at the moment."
        b.add("paraphrase", "paraphrase", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user works for {v}.", sa, "work for", v),
              fact_b=_fact(f"The user is working for {v}.", sb, "am working for", v),
              pair_key="user|works at", expected_object=v, expected_polarity=+1,
              expect_supersede=False, same_session=True)
    for v in rng.sample(CARS, 3):
        sa, sb = f"I drive a {v}.", f"My car is a {v}."
        b.add("paraphrase", "paraphrase", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user drives a {v}.", sa, "drive", f"a {v}"),
              fact_b=_fact(f"The user's car is a {v}.", sb, "car is", f"a {v}"),
              pair_key="user|drives", expected_object=v, expected_polarity=+1,
              expect_supersede=False, same_session=True)


def _unrelated(b: _Builder) -> None:
    rng = b.rng
    # has / likes / bought with different objects — never a functional
    # predicate, never a number: two facts, both must stay (the D2 hazard).
    for va, vb in zip(*[iter(rng.sample(POSSESSIONS, 8))] * 2, strict=True):
        sa, sb = f"I have {_article(va)} {va}.", f"I have {_article(vb)} {vb}."
        b.add("unrelated", "unrelated", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user has {_article(va)} {va}.", sa, "have",
                           f"{_article(va)} {va}"),
              fact_b=_fact(f"The user has {_article(vb)} {vb}.", sb, "have",
                           f"{_article(vb)} {vb}"),
              pair_key="user|own", expected_object=vb, expected_polarity=+1,
              expect_supersede=False)
    for va, vb in zip(*[iter(rng.sample(HOBBIES, 6))] * 2, strict=True):
        b.add("unrelated", "unrelated", sentence_a=f"I like {va}.", sentence_b=f"I like {vb}.",
              fact_a=_fact(f"The user likes {va}.", f"I like {va}.", "like", va),
              fact_b=_fact(f"The user likes {vb}.", f"I like {vb}.", "like", vb),
              pair_key="user|like", expected_object=vb, expected_polarity=+1,
              expect_supersede=False)
    for va, vb in zip(*[iter(rng.sample(POSSESSIONS, 6))] * 2, strict=True):
        sa, sb = f"I bought {_article(va)} {va}.", f"I bought {_article(vb)} {vb}."
        b.add("unrelated", "unrelated", sentence_a=sa, sentence_b=sb,
              fact_a=_fact(f"The user bought {_article(va)} {va}.", sa, "bought",
                           f"{_article(va)} {va}"),
              fact_b=_fact(f"The user bought {_article(vb)} {vb}.", sb, "bought",
                           f"{_article(vb)} {vb}"),
              pair_key="user|own", expected_object=vb, expected_polarity=+1,
              expect_supersede=False)


def _disambiguate(pairs: list[DemoPair]) -> None:
    """Append the pair id as a trailing clause to a user turn ONLY on a collision.

    Every user turn keys a ``ScriptedExtractor`` entry, so two pairs sharing a
    sentence would share a script line. The tables and framings are laid out
    so this never fires at seed 0; it exists so any seed is still well-formed.
    """
    seen: set[str] = set()
    for pair in pairs:
        for turns, facts in ((pair.turns_a, pair.facts_a), (pair.turns_b, pair.facts_b)):
            text = turns[0]["content"]
            if text in seen:
                text = f"{text} ({pair.id})"
                turns[0]["content"] = text
                facts[0] = facts[0]._replace(raw="user: " + text)
            seen.add(text)


def generate(seed: int = 0) -> list[DemoPair]:
    """The 100 demo pairs for ``seed`` — byte-identical per seed, family sizes = ``SIZES``."""
    b = _Builder(seed)
    _value_change(b)
    _negation(b)
    _dated_update(b)
    _paraphrase(b)
    _unrelated(b)
    _disambiguate(b.pairs)
    return b.pairs


def script_for(pairs: list[DemoPair]) -> dict[str, list[ExtractedFact]]:
    """The ``ScriptedExtractor`` script: user-turn text → the authored facts."""
    script: dict[str, list[ExtractedFact]] = {}
    for pair in pairs:
        for turns, facts in ((pair.turns_a, pair.facts_a), (pair.turns_b, pair.facts_b)):
            text = turns[0]["content"]
            if text in script and script[text] != list(facts):
                raise ValueError(f"two pairs script the same user turn differently: {text!r}")
            script[text] = list(facts)
    return script


def _messages(turns: list[dict], ts: str) -> list[dict]:
    return [{"role": t["role"], "content": t["content"], "ts": ts} for t in turns]


def _view(record) -> tuple[str, int]:
    triple = normalize_triple(record.subject, record.predicate, record.object)
    if triple is None:
        return (record.object or "", +1)
    return (triple.object, triple.polarity)


def evaluate(memory: Memory, pair: DemoPair, user_id: str = USER_ID) -> dict:
    """D10's reading of one store: ``{"correct", "active", "superseded", "why"}``."""
    facts = [f for f in memory.store.facts_with_pair_key(user_id) if f.pair_key == pair.pair_key]
    active = [f for f in facts if f.salience > 0]
    inactive = [f for f in facts if f.salience <= 0]
    view = [_view(f) for f in active]
    expected = (pair.expected_object, pair.expected_polarity)
    why = ""
    if pair.family in CONFLICT_FAMILIES:
        if len(facts) < 2:
            why = f"only {len(facts)} fact on the pair key: the other was merged, not stored"
        elif len(active) != 1:
            why = f"{len(active)} active facts on the pair key, expected exactly one"
        elif view[0] != expected:
            why = f"active fact is {view[0]}, expected {expected}"
        elif len(inactive) != 1 or inactive[0].salience != 0.0:
            why = "no loser with salience 0"
        elif active[0].supersedes != inactive[0].id:
            why = f"winner.supersedes is {active[0].supersedes}, loser id is {inactive[0].id}"
    elif pair.family == "paraphrase":
        if inactive:
            why = f"{len(inactive)} fact(s) superseded on a paraphrase"
        elif not active or any(v != expected for v in view):
            why = f"active facts {view}, expected every one to be {expected}"
    else:  # unrelated
        if inactive:
            why = f"{len(inactive)} fact(s) superseded on an unrelated pair"
        elif len(active) != 2:
            why = f"{len(active)} active facts, expected two"
    return {"correct": not why, "active": view, "superseded": len(inactive), "why": why}


def _model_facts(memory: Memory, pair: DemoPair, user_id: str) -> dict:
    """The descriptive pass: what the real extractor made of the two rounds."""
    out = {}
    for name, turns, ts in (("a", pair.turns_a, pair.ts_a), ("b", pair.turns_b, pair.ts_b)):
        key = round_key(ts, "\n".join(str(t.get("content", "")).strip() for t in turns))
        out[name] = [
            {"fact": f.fact, "subject": f.subject, "predicate": f.predicate, "object": f.object,
             "valid_time": f.valid_time, "pair_key": f.pair_key, "salience": f.salience}
            for f in memory.store.facts_of(user_id, key)
        ]
    keys_a = {f["pair_key"] for f in out["a"]} - {None}
    keys_b = {f["pair_key"] for f in out["b"]} - {None}
    out["conflictable"] = bool(keys_a & keys_b)
    out["on_authored_key"] = pair.pair_key in (keys_a & keys_b)
    return out


def run_pair(pair: DemoPair, config: MemoryConfig, embedder, extractor=None) -> dict:
    """One fresh ``Memory``: add both rounds, read; consolidate, read again."""
    scripted = extractor is None
    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory(str(Path(tmp) / "demo.db"), embedder, config,
                        extractor=ScriptedExtractor(script_for([pair])) if scripted else extractor)
        try:
            memory.add(_messages(pair.turns_a, pair.ts_a), user_id=USER_ID)
            memory.add(_messages(pair.turns_b, pair.ts_b), user_id=USER_ID)
            after_add = evaluate(memory, pair)
            memory.consolidate(USER_ID)
            after_consolidate = evaluate(memory, pair)
            result = {
                "id": pair.id, "family": pair.family, "subfamily": pair.subfamily,
                "after_add": after_add, "after_consolidate": after_consolidate,
                "conflict_stats": dict(memory.conflict_stats),
            }
            if not scripted:
                result["model"] = _model_facts(memory, pair, USER_ID)
        finally:
            memory.store.close()
    return result


def run(pairs: list[DemoPair], conflict_resolution: bool, embedder=None, extractor=None,
        ) -> list[dict]:
    """The whole set under one configuration; everything else is the library default."""
    embedder = embedder if embedder is not None else HashingEmbedder()
    config = MemoryConfig(conflict_resolution=conflict_resolution)
    return [run_pair(pair, config, embedder, extractor) for pair in pairs]


def _tally(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows),
        "correct_after_add": sum(r["after_add"]["correct"] for r in rows),
        "correct_after_consolidate": sum(r["after_consolidate"]["correct"] for r in rows),
        "failures": [r["id"] for r in rows
                     if not (r["after_add"]["correct"] and r["after_consolidate"]["correct"])],
    }


def summarize(results: list[dict]) -> dict:
    """Per family (and the two groups): pairs, correct after add / consolidate, failure ids."""
    summary = {"families": {}, "failures": {}}
    for family in FAMILIES:
        rows = [r for r in results if r["family"] == family]
        summary["families"][family] = _tally(rows)
        if any("model" in r for r in rows):
            summary["families"][family]["conflictable"] = sum(
                r["model"]["conflictable"] for r in rows if "model" in r)
            summary["families"][family]["on_authored_key"] = sum(
                r["model"]["on_authored_key"] for r in rows if "model" in r)
    summary["conflicts"] = _tally([r for r in results if r["family"] in CONFLICT_FAMILIES])
    summary["controls"] = _tally([r for r in results if r["family"] in CONTROL_FAMILIES])
    for r in results:
        failed = summary["conflicts"]["failures"] + summary["controls"]["failures"]
        if r["id"] in failed:
            summary["failures"][r["id"]] = {
                "after_add": r["after_add"]["why"],
                "after_consolidate": r["after_consolidate"]["why"],
            }
    return summary


def criterion(summary: dict, baseline: dict | None = None) -> tuple[bool, str]:
    """D10's criterion, verbatim, on both readings; ``baseline`` adds the strict-above check."""
    checks: list[tuple[bool, str]] = []
    for reading in ("correct_after_add", "correct_after_consolidate"):
        tag = reading.removeprefix("correct_")
        n = summary["conflicts"][reading]
        checks.append((n >= MIN_CONFLICTS_CORRECT,
                       f"conflicts {n}/80 >= {MIN_CONFLICTS_CORRECT} {tag}"))
        for family, bar in MIN_PER_FAMILY.items():
            n = summary["families"][family][reading]
            checks.append((n >= bar, f"{family} {n}/{SIZES[family]} >= {bar} {tag}"))
        n = summary["controls"][reading]
        checks.append((n == CONTROLS_REQUIRED, f"controls {n}/20 == 20 {tag}"))
        if baseline is not None:
            b = baseline["conflicts"][reading]
            checks.append((summary["conflicts"][reading] > b,
                           f"conflicts {summary['conflicts'][reading]} > baseline {b} {tag}"))
    passed = all(ok for ok, _ in checks)
    reason = "; ".join(("" if ok else "FAILED ") + text for ok, text in checks)
    return passed, reason


def format_table(on: dict, off: dict) -> str:
    """mnimi vs exact-dedup per family, after ``add()`` and after ``consolidate()``."""
    header = (f"{'family':<14} {'pairs':>5} | {'mnimi add':>9} {'consol.':>7} | "
              f"{'exact add':>9} {'consol.':>7} | conflictable | mnimi failures")
    lines = [header, "-" * len(header)]
    for name in (*FAMILIES, "conflicts", "controls"):
        a = on["families"].get(name, on.get(name))
        b = off["families"].get(name, off.get(name))
        label = name if name in FAMILIES else f"= {name}"
        conflictable = a.get("conflictable")
        conflictable_cell = "-" if conflictable is None else f"{conflictable}/{a['pairs']}"
        failures = ", ".join(a["failures"]) or "-"
        lines.append(
            f"{label:<14} {a['pairs']:>5} | {a['correct_after_add']:>9} "
            f"{a['correct_after_consolidate']:>7} | {b['correct_after_add']:>9} "
            f"{b['correct_after_consolidate']:>7} | {conflictable_cell:>12} | {failures}"
        )
    if on["failures"]:
        lines.append("")
        lines.append("mnimi failures (after add / after consolidate):")
        for pid, why in on["failures"].items():
            lines.append(f"  {pid}: {why['after_add'] or 'ok'} / "
                         f"{why['after_consolidate'] or 'ok'}")
    return "\n".join(lines)


def _build_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    from mnimi.embeddings import BgeSmallEmbedder  # the [embed] extra

    return BgeSmallEmbedder()


def _build_extractor(name: str):
    """``None`` for the scripted pass; the pinned Qwen3 extractor behind its own cache otherwise.

    The cache is the demo's own file (``.cache/extract/conflict_demo_<pins hash>.sqlite``),
    never the corpus cache: the corpus file is the slice's replay evidence and
    nothing else writes to it.
    """
    if name == "none":
        return None
    from mnimi.extract.cache import CachedExtractor, default_cache_path
    from mnimi.extract.llama import QwenLlamaExtractor  # the [extract] extra + CUDA

    inner = QwenLlamaExtractor()
    corpus = default_cache_path(inner.pins)
    return CachedExtractor(inner, corpus.with_name("conflict_demo_" + corpus.name))


def _pair_json(pair: DemoPair) -> dict:
    d = asdict(pair)
    d["facts_a"] = [f._asdict() for f in pair.facts_a]
    d["facts_b"] = [f._asdict() for f in pair.facts_b]
    return d


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedder", choices=["hashing", "bge"], default="hashing")
    parser.add_argument("--extractor", choices=["none", "qwen3"], default="none",
                        help="qwen3: the natural-language rounds through the real extractor "
                             "(GPU); descriptive, never the gate")
    parser.add_argument("--out", default="runs/conflict_demo.json")
    args = parser.parse_args(argv)

    pairs = generate(args.seed)
    embedder = _build_embedder(args.embedder)
    extractor = _build_extractor(args.extractor)
    on = run(pairs, True, embedder, extractor)
    off = run(pairs, False, embedder, extractor)
    summary_on, summary_off = summarize(on), summarize(off)
    passed, reason = criterion(summary_on, summary_off)
    table = format_table(summary_on, summary_off)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "seed": args.seed, "embedder": args.embedder, "extractor": args.extractor,
        "descriptive": args.extractor != "none",
        "criterion": {"passed": passed, "reason": reason},
        "summary_on": summary_on, "summary_off": summary_off,
        "results_on": on, "results_off": off,
        "pairs": [_pair_json(p) for p in pairs],
    }
    if extractor is not None and hasattr(extractor, "stats"):
        artifact["cache_stats"] = dict(extractor.stats)
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"conflict demo: seed {args.seed}, embedder {args.embedder}, extractor {args.extractor}")
    print(table)
    print()
    if args.extractor != "none":
        print(f"DESCRIPTIVE (--extractor qwen3): not the gate. criterion would read: "
              f"{'PASS' if passed else 'FAIL'} — {reason}")
        if "cache_stats" in artifact:
            print(f"extractor cache: {artifact['cache_stats']}")
        print(f"wrote {out}")
        return 0
    print(f"GATE 3-ii: {'PASS' if passed else 'FAIL'} — {reason}")
    print(f"wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
