"""Phase 3: the frozen negation lexicon, triple normalization, the pair key."""

from __future__ import annotations

import subprocess
import sys

import pytest

from mnimi.conflict import lexicon, normalize
from mnimi.conflict.normalize import NormalizedTriple, normalize_triple


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
