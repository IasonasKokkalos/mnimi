"""Phase 3: the frozen negation lexicon, triple normalization, the pair key."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mnimi.conflict import lexicon, normalize, supersede
from mnimi.conflict.normalize import NormalizedTriple, normalize_triple
from mnimi.conflict.screens import KEEP_REASONS, Verdict, screen_pair, token_entropy_bits
from mnimi.models import MemoryRecord


def test_importing_mnimi_conflict_loads_no_heavy_dependency():
    code = (
        "import sys, mnimi, mnimi.conflict, mnimi.conflict.normalize; "
        "assert not {'llama_cpp', 'huggingface_hub', 'onnxruntime'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "triple,expected",
    [
        # subject folding and the functional groups (D3)
        (("The user", "lives in", "Boston"), ("user", "lives in", "boston", +1, "lives in")),
        (("I", "moved to", "Seattle"), ("user", "lives in", "seattle", +1, "lives in")),
        (("We", "have", "a cat"), ("user", "own", "cat", +1, None)),
        (("user", "is currently living in", "Tokyo, Japan"),
         ("user", "lives in", "tokyo japan", +1, "lives in")),
        (("user", "works at", "the Acme Corp."), ("user", "works at", "acme corp", +1, "works at")),
        (("user", "is married to", "Anna"), ("user", "partner", "anna", +1, "partner")),
        (("user's sister", "lives in", "Boston"),
         ("users sister", "lives in", "boston", +1, "lives in")),
        (("assistant", "explained", "the tariffs"),
         ("assistant", "explained", "tariffs", +1, None)),
        # polarity: markers, contractions, antonym forms (D4)
        (("user", "no longer lives in", "Boston"), ("user", "lives in", "boston", -1, "lives in")),
        (("user", "used to live in", "Boston"), ("user", "lives in", "boston", -1, "lives in")),
        (("user", "likes", "jazz"), ("user", "like", "jazz", +1, None)),
        (("user", "dislikes", "jazz"), ("user", "like", "jazz", -1, None)),
        (("user", "hates", "jazz"), ("user", "like", "jazz", -1, None)),
        (("user", "doesn't like", "jazz"), ("user", "like", "jazz", -1, None)),
        (("user", "is a fan of", "Coldplay"), ("user", "like", "coldplay", +1, None)),
        (("user", "not a fan of", "Coldplay"), ("user", "like", "coldplay", -1, None)),
        (("user", "is", "vegetarian"), ("user", "is", "vegetarian", +1, None)),
        (("user", "is not", "a vegetarian"), ("user", "is", "vegetarian", -1, None)),
        (("user", "isn't", "a vegetarian"), ("user", "is", "vegetarian", -1, None)),
        (("user", "has", "no car"), ("user", "own", "car", -1, None)),
        (("user", "does not have", "a car"), ("user", "own", "car", -1, None)),
        (("user", "sold", "the car"), ("user", "own", "car", -1, None)),
        (("user", "never tried", "Lebanese cuisine"),
         ("user", "tried", "lebanese cuisine", -1, None)),
        (("user", "stopped going to", "the gym"), ("user", "going to", "gym", -1, None)),
        (("user", "stopped", "yoga"), ("user", "start", "yoga", -1, None)),
        (("user", "started", "yoga"), ("user", "start", "yoga", +1, None)),
        (("user", "divorced", "Anna"), ("user", "partner", "anna", -1, "partner")),
        (("user", "cannot eat", "gluten"), ("user", "eat", "gluten", -1, None)),
        (("user", "can eat", "gluten"), ("user", "eat", "gluten", +1, None)),
        # numbers in the object (the count-update case)
        (("user", "has tried", "3 of Emma's recipes"),
         ("user", "tried", "3 of emmas recipes", +1, None)),
        (("user", "has tried", "two of Emma's recipes"),
         ("user", "tried", "2 of emmas recipes", +1, None)),
    ],
)
def test_normalize_triple_table(triple, expected):
    assert normalize_triple(*triple) == NormalizedTriple(*expected)


def test_normalize_triple_abstains_on_a_null_or_empty_part():
    assert normalize_triple(None, "lives in", "Boston") is None
    assert normalize_triple("user", "", "Boston") is None
    assert normalize_triple("user", "lives in", "...") is None
    assert normalize_triple("user", "lives in", None) is None


def test_pair_key_is_the_normalized_subject_and_predicate():
    t = normalize_triple("The user", "moved to", "Seattle")
    assert t.pair_key == "user|lives in" == normalize.pair_key("user", "lives in")
    assert normalize_triple("user", "dislikes", "jazz").pair_key == "user|like"


def test_numeric_signature_drops_approximators_and_keeps_the_residue():
    assert normalize.numeric_signature("3 of emmas recipes") == (("3",), "of emmas recipes")
    assert normalize.numeric_signature("about 1300 followers") == (("1300",), "followers")
    assert normalize.numeric_signature("close to 1300 followers") == (("1300",), "followers")
    assert normalize.numeric_signature("1300 followers") == (("1300",), "followers")
    assert normalize.numeric_signature("boston") is None
    assert normalize.value_tokens("tokyo japan") == 2


def test_polarity_of_text_is_the_fallback_for_a_null_triple():
    assert normalize.polarity_of_text("The user lives in Boston.") == +1
    assert normalize.polarity_of_text("The user no longer lives in Boston.") == -1
    assert normalize.polarity_of_text("The user dislikes jazz.") == -1
    assert normalize.polarity_of_text("The user doesn't play tennis anymore.") == -1
    assert normalize.polarity_of_text("The user has never been to Lebanon.") == -1
    assert normalize.polarity_of_text("The user is not unable to drive.") == +1, "double negation"


def test_lexicon_hashes_are_frozen_distinct_and_move_with_an_edit(monkeypatch):
    negation = lexicon.negation_lexicon_hash()
    rules = normalize.conflict_rules_hash()
    assert len(negation) == 64 and len(rules) == 64 and negation != rules
    assert negation != "none", "the guard row goes live in Phase 3"
    # Frozen 2026-09-15 (PHASE3 D4/D12). A change here is a versioned
    # migration — bump the version, re-ingest — never an edit under a run.
    assert negation.startswith("330604b5772e")
    assert rules.startswith("7d19c48828c8")
    monkeypatch.setattr(lexicon, "MARKERS", (*lexicon.MARKERS, "nope"))
    assert lexicon.negation_lexicon_hash() != negation
    monkeypatch.setattr(normalize, "MAX_VALUE_TOKENS", 7)
    assert normalize.conflict_rules_hash() != rules


def test_lexicon_forms_are_normalized_and_unambiguous():
    """Every listed form is already in normalized shape (so a lookup can hit
    it) and no form sits on two sides or in two groups."""
    seen: dict[str, str] = {}
    for key, (pos, neg) in lexicon.ANTONYMS.items():
        for side, forms in (("+", pos), ("-", neg)):
            for form in forms:
                assert form == normalize.normalize_text(form), form
                assert form not in seen, f"{form!r} in {seen[form]} and {key}{side}"
                seen[form] = f"{key}{side}"
    functional: dict[str, str] = {}
    for key, forms in lexicon.FUNCTIONAL.items():
        for form in forms:
            assert form == normalize.normalize_text(form), form
            assert form not in functional, f"{form!r} in {functional[form]} and {key}"
            functional[form] = key
    for marker in lexicon.MARKERS:
        assert marker == normalize.normalize_text(marker), marker
    assert "uses" not in functional and "using" not in functional and "has" not in functional


# -- the screens (PHASE3 Task 2) -------------------------------------------------------

def test_token_entropy_bits_is_token_level_shannon():
    assert token_entropy_bits("") == 0.0
    assert token_entropy_bits("Miso") == 0.0
    assert token_entropy_bits("The user is 30.") == pytest.approx(2.0)  # 4 distinct tokens
    assert token_entropy_bits("the the the user") == pytest.approx(0.8112781)  # (3/4, 1/4)
    assert token_entropy_bits("The user adopted a cat named Miso.") == pytest.approx(2.8073549)


def _t(s, p, o):
    return (s, p, o)


def test_screen_pair_routes_negation_value_and_low_entropy_away_from_the_merge():
    assert KEEP_REASONS == ("negation", "value", "low-entropy")
    # negation: opposite polarity on a near-duplicate — with a triple, and without one
    assert screen_pair("The user likes jazz.", _t("user", "likes", "jazz"),
                       "The user dislikes jazz.", _t("user", "dislikes", "jazz"),
                       2.0) == Verdict("keep", "negation")
    assert screen_pair("The user no longer lives in Boston.", _t(None, None, None),
                       "The user lives in Boston.", _t(None, None, None),
                       2.0) == Verdict("keep", "negation")
    # value: same pair, different value-sized objects
    assert screen_pair("The user lives in Seattle.", _t("user", "lives in", "Seattle"),
                       "The user lives in Boston.", _t("user", "moved to", "Boston"),
                       2.0) == Verdict("keep", "value")
    assert screen_pair("The user has tried 3 of Emma's recipes.",
                       _t("user", "has tried", "3 of Emma's recipes"),
                       "The user has tried 2 of Emma's recipes.",
                       _t("user", "has tried", "two of Emma's recipes"),
                       2.0) == Verdict("keep", "value")
    # low entropy: too few distinct tokens to trust a cosine
    assert screen_pair("Miso.", _t(None, None, None), "Miso", _t(None, None, None),
                       2.0) == Verdict("keep", "low-entropy")
    assert screen_pair("Miso.", _t(None, None, None), "Miso", _t(None, None, None),
                       0.0) == Verdict("merge", "duplicate")


def test_screen_pair_abstains_on_null_triples_and_merges_true_duplicates():
    dup = Verdict("merge", "duplicate")
    # same pair, same object, same polarity: a restatement
    assert screen_pair("The user lives in Boston.", _t("user", "lives in", "Boston"),
                       "The user is living in Boston.", _t("user", "living in", "Boston"),
                       2.0) == dup
    # a clause-sized object is not a value: the value screen abstains, the pair merges
    long_obj = "the evolution of language is a complex and fascinating process"
    assert screen_pair("The user mentioned that " + long_obj + ".",
                       _t("user", "mentioned", long_obj),
                       "The user mentioned language evolution.",
                       _t("user", "mentioned", "language evolution"), 2.0) == dup
    # one triple null and no polarity difference: abstain → entropy → merge
    assert screen_pair("The user adopted a cat named Miso.", _t(None, None, None),
                       "The user adopted a cat called Miso.",
                       _t("user", "adopted", "a cat called Miso"), 2.0) == dup


# -- supersede: the ordering and the three rules (PHASE3 Task 3) -------------------------


def _rec(id, created_at, valid_time=None, raw="user: x", subject="user", predicate="lives in",
         obj="Boston", fact=None, salience=1.0):
    t = normalize_triple(subject, predicate, obj)
    return MemoryRecord(id=id, user_id="u", content="c", created_at=created_at, kind="fact",
                        fact=fact or f"The user {predicate} {obj}.", raw=raw, subject=subject,
                        predicate=predicate, object=obj, valid_time=valid_time, salience=salience,
                        pair_key=t.pair_key if t else None)


def test_time_keys_parse_every_precision_and_never_read_the_clock():
    assert supersede.valid_time_key("2023") == (2023, 0, 0)
    assert supersede.valid_time_key("2023-05") == (2023, 5, 0)
    assert supersede.valid_time_key("2023-05-20") == (2023, 5, 20)
    assert supersede.valid_time_key(None) is None
    assert supersede.created_at_key("2023/05/20 (Sat) 09:00") == (2023, 5, 20)
    assert supersede.created_at_key("2023-05-20") == (2023, 5, 20)
    assert supersede.created_at_key("yesterday") == (0, 0, 0)
    import inspect
    src = inspect.getsource(supersede)
    assert "datetime.now" not in src and "date.today" not in src and "time.time" not in src


def test_ordering_dated_facts_by_valid_time_regardless_of_session_order():
    boston = _rec(1, "2023/06/10 (Sat) 09:00", valid_time="2019", obj="Boston")
    seattle = _rec(2, "2023/01/10 (Tue) 09:00", valid_time="2022-12", obj="Seattle")
    assert supersede.beats(seattle, boston) and not supersede.beats(boston, seattle)


def test_ordering_standing_facts_by_the_later_session_then_trust_then_id():
    a = _rec(1, "2023/01/10 (Tue) 09:00", obj="Boston")
    b = _rec(2, "2023/06/10 (Sat) 09:00", obj="Seattle")
    assert supersede.beats(b, a)
    same_session_user = _rec(3, "2023/06/10 (Sat) 09:00", obj="Tacoma", raw="user: x")
    same_session_assistant = _rec(4, "2023/06/10 (Sat) 09:00", obj="Olympia", raw="assistant: x")
    assert supersede.beats(same_session_user, same_session_assistant), "the user's span outranks"
    later_id = _rec(5, "2023/06/10 (Sat) 09:00", obj="Everett", raw="user: x")
    assert supersede.beats(later_id, same_session_user), "full tie: the later id"


def test_ordering_mixed_dated_vs_standing_uses_one_effective_time():
    standing_2023 = _rec(1, "2023/03/01 (Wed) 09:00", obj="Boston")
    dated_2023_06 = _rec(2, "2023/06/10 (Sat) 09:00", valid_time="2023-06", obj="Seattle")
    assert supersede.beats(dated_2023_06, standing_2023)
    dated_2015 = _rec(3, "2023/09/01 (Fri) 09:00", valid_time="2015", obj="Denver")
    assert supersede.beats(standing_2023, dated_2015), "a 2015 fact does not beat a 2023 assertion"


def test_conflict_between_applies_the_three_rules_and_abstains_elsewhere():
    cb = supersede.conflict_between
    assert cb(_rec(1, "2023-01-01", obj="Boston"),
              _rec(2, "2023-01-02", obj="Seattle")) == "functional"
    assert cb(_rec(1, "2023-01-01", predicate="likes", obj="jazz"),
              _rec(2, "2023-01-02", predicate="dislikes", obj="jazz")) == "negation"
    assert cb(_rec(1, "2023-01-01", predicate="has", obj="2 cats"),
              _rec(2, "2023-01-02", predicate="has", obj="3 cats")) == "numeric"
    assert cb(_rec(1, "2023-01-01", predicate="has", obj="a cat"),
              _rec(2, "2023-01-02", predicate="has", obj="a sister")) is None
    assert cb(_rec(1, "2023-01-01", predicate="likes", obj="jazz"),
              _rec(2, "2023-01-02", predicate="likes", obj="hiking")) is None
    assert cb(_rec(1, "2023-01-01", subject="assistant", predicate="lives in", obj="Boston"),
              _rec(2, "2023-01-02", subject="assistant", predicate="lives in",
                   obj="Seattle")) is None
    assert cb(_rec(1, "2023-01-01", predicate="lives in", obj="Boston"),
              _rec(2, "2023-01-02", predicate="does not live in", obj="Seattle")) is None, \
        "a negative value asserts nothing"
    assert cb(_rec(1, "2023-01-01", obj="Boston"),
              _rec(2, "2023-01-02", predicate="likes", obj="Seattle")) is None, "different pairs"
    assert supersede.reason("functional", _rec(1, "2023-01-01", obj="Boston"),
                            _rec(2, "2023-01-02", obj="Seattle")) == \
        "functional user|lives in: boston -> seattle"
    assert supersede.RULES == ("negation", "functional", "numeric")
