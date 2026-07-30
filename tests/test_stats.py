"""Statistics, checked against values computable by hand or from the literature."""

from __future__ import annotations

import pytest
from evals.stats import (
    HARNESS_PARITY_FIELDS,
    analyse,
    discordance,
    exact_binomial_two_sided,
    harness_identity,
    holm,
    mcnemar_exact,
    mcnemar_power,
    minimum_detectable_gap,
    wilson,
)


def test_wilson_matches_the_formula_computed_by_hand():
    """20/100 at 95%: centre 0.211098, half-width 0.077732, so
    [0.133366, 0.288830]. Worked through by hand rather than read off another
    implementation, so this test is an independent check and not a mirror."""
    ci = wilson(20, 100)
    assert ci.point == pytest.approx(0.20)
    assert ci.low == pytest.approx(0.133366, abs=5e-5)
    assert ci.high == pytest.approx(0.288830, abs=5e-5)


def test_wilson_never_leaves_the_unit_interval():
    """The reason for using it: the normal approximation goes negative here."""
    ci = wilson(0, 20)
    assert ci.low == 0.0
    assert 0.0 < ci.high < 1.0

    ci = wilson(20, 20)
    assert ci.high == 1.0
    assert 0.0 < ci.low < 1.0


def test_wilson_interval_narrows_with_n():
    small = wilson(9, 20)
    large = wilson(45, 100)
    assert small.point == large.point
    assert (large.high - large.low) < (small.high - small.low)


def test_exact_binomial_is_exact_on_small_counts():
    # 0 of 3 fair trials: two-sided p = 2 * (1/8) = 0.25.
    assert exact_binomial_two_sided(0, 3) == pytest.approx(0.25)
    # 0 of 5: 2 * (1/32) = 0.0625 — still not significant at 0.05, which is why
    # a handful of discordant pairs can never reach significance.
    assert exact_binomial_two_sided(0, 5) == pytest.approx(0.0625)
    # 0 of 6: 2 * (1/64) = 0.03125 — the smallest discordant count that can.
    assert exact_binomial_two_sided(0, 6) == pytest.approx(0.03125)
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_mcnemar_reports_the_counts_alongside_the_p_value():
    result = mcnemar_exact(b=2, c=8, n_pairs=100)
    assert (result.b, result.c, result.discordant, result.n_pairs) == (2, 8, 10, 100)
    assert result.p_value == pytest.approx(0.109375, abs=1e-6)


def test_discordance_counts_both_directions():
    first = {"q1": True, "q2": True, "q3": False, "q4": False}
    second = {"q1": True, "q2": False, "q3": True, "q4": False}
    assert discordance(first, second) == (1, 1, 4)


def test_discordance_refuses_mismatched_question_sets():
    with pytest.raises(ValueError, match="same questions"):
        discordance({"q1": True}, {"q2": True})


def test_holm_is_step_down_and_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert adjusted["c"] == pytest.approx(0.06)  # max(0.03, 2 * 0.03)
    assert adjusted["b"] == pytest.approx(0.06)  # max(0.06, 1 * 0.04)
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]


def test_power_rises_with_sample_size_and_with_the_gap():
    assert mcnemar_power(500, 0.30, 0.10) > mcnemar_power(100, 0.30, 0.10)
    assert mcnemar_power(100, 0.30, 0.15) > mcnemar_power(100, 0.30, 0.05)
    assert mcnemar_power(100, 0.30, 0.0) < 0.06  # no effect: near alpha


def test_minimum_detectable_gap_is_the_smallest_powered_gap():
    mde = minimum_detectable_gap(100, 0.30)
    assert mde is not None
    assert mcnemar_power(100, 0.30, mde) >= 0.8
    assert mcnemar_power(100, 0.30, mde * 0.9) < 0.8


def test_minimum_detectable_gap_is_none_when_unreachable():
    """A tiny discordance rate caps the achievable gap: even winning every
    disagreement is not enough to reach 80% power."""
    assert minimum_detectable_gap(20, 0.02) is None


def _payload(system: str, **pin_overrides) -> dict:
    """A results.json-shaped payload with every harness field held constant."""
    pins = {field: f"shared-{field}" for field in HARNESS_PARITY_FIELDS}
    pins["system"] = system
    pins.update(pin_overrides)
    # Judge identity travels in its own block at schema /3.
    judge = {
        "judge_model": pins.pop("judge_model"),
        "judge_prompt_hash": pins.pop("judge_prompt_hash"),
        "judge_temperature": pins.pop("judge_temperature"),
        "judge_max_tokens": pins.pop("judge_max_tokens"),
    }
    return {"pins": pins, "judge": judge}


def test_pairing_refuses_across_arms_from_different_harness_configurations():
    """The n=20 table mixed answer_reserve 1024 with 800; the paired statistics
    it produced compared configurations, not systems. That must now be a hard
    error naming the field and both values, not a silent number."""
    arms = {"mnimi": {"q1": True}, "naive_rag": {"q1": False}}
    identities = {
        "mnimi": harness_identity(_payload("mnimi", reader_answer_reserve=800)),
        "naive_rag": harness_identity(_payload("naive_rag", reader_answer_reserve=1024)),
    }
    with pytest.raises(ValueError, match="reader_answer_reserve") as excinfo:
        analyse(arms, identities=identities)
    assert "800" in str(excinfo.value) and "1024" in str(excinfo.value)


def test_pairing_proceeds_across_arms_differing_only_in_retrieval_pins():
    """embedder/k/threshold differ across arms BY DESIGN — that difference is
    the experiment, so it must never trip the parity guard."""
    arms = {"mnimi": {"q1": True, "q2": False}, "naive_rag": {"q1": True, "q2": True}}
    identities = {
        "mnimi": harness_identity(
            _payload("mnimi", embedder_name="BAAI/bge-small-en-v1.5",
                     embedder_revision="5c38ec7c", k=10, dedup_cosine_threshold=0.95)
        ),
        "naive_rag": harness_identity(
            _payload("naive_rag", embedder_name="BAAI/bge-small-en-v1.5",
                     embedder_revision="5c38ec7c", k=10)
        ),
    }
    report = analyse(arms, identities=identities)
    assert report["primary"] is not None
    assert report["primary"]["result"].n_pairs == 2
