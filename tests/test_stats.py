"""Statistics, checked against values computable by hand or from the literature."""

from __future__ import annotations

import json
import math

import pytest
from evals import stats
from evals.stats import (
    HARNESS_PARITY_FIELDS,
    _format,
    analyse,
    assert_harness_parity,
    discordance,
    exact_binomial_two_sided,
    harness_identity,
    holm,
    mcnemar_exact,
    mcnemar_power,
    minimum_detectable_gap,
    parity_notes,
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


def test_pairing_refuses_across_reader_transport_builds():
    """Schema /5: the Ollama build is a harness pin. Two arms served by
    different builds are two configurations — the 0.32.5 -> 0.32.13 move
    changed 20/20 predictions — and the guard must name the field and both
    builds instead of pairing them under a bare sha mismatch."""
    identities = {
        "mnimi": harness_identity(_payload("mnimi", reader_transport_version="0.32.13")),
        "naive_rag": harness_identity(_payload("naive_rag", reader_transport_version="0.33.3")),
    }
    with pytest.raises(ValueError, match="reader_transport_version") as excinfo:
        assert_harness_parity(identities)
    assert "0.32.13" in str(excinfo.value) and "0.33.3" in str(excinfo.value)


def test_schema_4_artifacts_without_the_transport_field_still_pair():
    """The published 0.32.13 set predates the field on both sides; None == None
    is parity, not a mismatch, so those artifacts keep pairing."""
    identities = {
        "mnimi": harness_identity(_payload("mnimi", reader_transport_version=None)),
        "naive_rag": harness_identity(_payload("naive_rag", reader_transport_version=None)),
    }
    assert_harness_parity(identities)


def test_pairing_refuses_across_reader_transports():
    """Schema /6: the local 1.5B family and the gpt-4o API family are two
    configurations; a paired test across them would measure the reader."""
    identities = {
        "mnimi": harness_identity(_payload("mnimi", reader_transport="ollama")),
        "naive_rag": harness_identity(_payload("naive_rag", reader_transport="openai")),
    }
    with pytest.raises(ValueError, match="reader_transport") as excinfo:
        assert_harness_parity(identities)
    assert "ollama" in str(excinfo.value) and "openai" in str(excinfo.value)



def test_two_arms_of_one_system_pair_as_a_variant_pair():
    """Phase 1 protocol: a baseline and a variant of one pinned knob, run at
    one commit, are named by run dir and paired on their own; the primary and
    secondary tables skip a system that appears twice."""
    arms = {
        "mnimi@base": {"q1": True, "q2": False, "q3": False, "q4": True},
        "mnimi@variant": {"q1": True, "q2": True, "q3": True, "q4": False},
        "naive_rag": {"q1": True, "q2": True, "q3": False, "q4": False},
    }
    report = analyse(arms)
    (label, entry), = report["variants"].items()
    assert label == "mnimi@variant vs mnimi@base"
    assert (entry["result"].b, entry["result"].c) == (2, 1), "b counts the variant's wins"
    assert report["primary"] is None and report["secondary"] == {}
    text = _format(report)
    assert "variant pair" in text and "mnimi@variant vs mnimi@base" in text


def test_two_mnimi_arms_differing_in_render_unit_and_extractor_pair():
    """PHASE2: the render unit and the extractor pins are system-level, so the
    baseline arm (no extractor, turns) and the extraction arm pair as a
    variant pair — that is gate 4-iii's comparison."""
    arms = {
        "mnimi@p2base": {"q1": True, "q2": False, "q3": False},
        "mnimi@extract": {"q1": True, "q2": True, "q3": False},
    }
    identities = {
        "mnimi@p2base": harness_identity(
            _payload("mnimi", extractor_model="none", render_unit="turns",
                     render_unit_template_hash="t")
        ),
        "mnimi@extract": harness_identity(
            _payload("mnimi", extractor_model="Qwen/x@abc", render_unit="round+facts",
                     render_unit_template_hash="rf", extractor_prompt_hash="p")
        ),
    }
    report = analyse(arms, identities=identities)
    (label, entry), = report["variants"].items()
    assert label == "mnimi@extract vs mnimi@p2base"
    assert (entry["result"].b, entry["result"].c) == (1, 0)


def test_pairing_proceeds_across_a_schema_bump_and_a_harness_commit_and_reports_them():
    """Phase 6 (v2.9.0): D8 pairs an arm at schema /11 and a later commit against
    the published mnimi__500q_gpt4o (/10, f07c24d). Like evals.drift, the two
    fields are reported in the record, never refused — every other parity
    field still is."""
    arms = {"mnimi": {"q1": True, "q2": False}, "naive_rag": {"q1": True, "q2": True}}
    identities = {
        "mnimi": harness_identity(_payload(
            "mnimi", artifact_schema="mnimi-eval-artifact/11", harness_git_sha="f0a7de3")),
        "naive_rag": harness_identity(_payload(
            "naive_rag", artifact_schema="mnimi-eval-artifact/10", harness_git_sha="f07c24d")),
    }
    assert_harness_parity(identities)
    report = analyse(arms, identities=identities)
    assert report["primary"] is not None
    assert report["harness_notes"] == {
        "artifact_schema": {"mnimi": "mnimi-eval-artifact/11",
                            "naive_rag": "mnimi-eval-artifact/10"},
        "harness_git_sha": {"mnimi": "f0a7de3", "naive_rag": "f07c24d"},
    }
    assert "NOTE artifact_schema differs" in _format(report)


def test_parity_notes_are_empty_when_the_arms_share_commit_and_schema():
    identities = {
        "mnimi": harness_identity(_payload("mnimi")),
        "naive_rag": harness_identity(_payload("naive_rag")),
    }
    assert parity_notes(identities) == {}
    assert analyse({"mnimi": {"q1": True}, "naive_rag": {"q1": True}},
                   identities=identities)["harness_notes"] == {}


def test_a_reported_field_never_masks_a_refused_one():
    """A schema bump beside a reader change still refuses, on the reader."""
    identities = {
        "mnimi": harness_identity(_payload(
            "mnimi", artifact_schema="mnimi-eval-artifact/11", reader_model="gpt-4o-2024-08-06")),
        "naive_rag": harness_identity(_payload(
            "naive_rag", artifact_schema="mnimi-eval-artifact/10", reader_model="gpt-4o-mini")),
    }
    with pytest.raises(ValueError, match="reader_model"):
        assert_harness_parity(identities)

def _replay(directory, n, model, verdicts, judge_hash="h"):
    payload = {"judge": {"judge_model": model}, "judge_hash": judge_hash,
               "results": [{"question_id": q, "correct": c} for q, c in verdicts.items()]}
    (directory / f"judge_replay_{n}.json").write_text(json.dumps(payload), encoding="utf-8")


def _arm(directory, system, first, second, judge_hash="h"):
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({
        "pins": {"system": system}, "pins_hash": "p", "judge": {},
        "results": [{"question_id": q, "correct": c} for q, c in first.items()],
    }), encoding="utf-8")
    _replay(directory, 1, "gpt-4.1-2025-04-14", second, judge_hash)


def test_judge_selected_verdicts_come_from_that_judges_replay(tmp_path):
    _replay(tmp_path, 1, "gpt-4.1-2025-04-14", {"a": False, "b": True})
    verdicts = stats.load_correctness(tmp_path, judge_model="gpt-4.1-2025-04-14")
    assert verdicts == {"a": False, "b": True}


def test_judge_selected_pairs_refuse_ambiguous_or_mismatched_replays(tmp_path):
    _replay(tmp_path, 1, "gpt-4.1-2025-04-14", {"a": True})
    _replay(tmp_path, 2, "gpt-4.1-2025-04-14", {"a": False})
    with pytest.raises(ValueError, match="exactly one"):
        stats.load_correctness(tmp_path, judge_model="gpt-4.1-2025-04-14")
    with pytest.raises(ValueError, match="exactly one"):
        stats.load_correctness(tmp_path, judge_model="gpt-4o-2024-11-20")


def test_a_judge_selected_pair_is_saved_beside_never_over_the_pair_of_record(tmp_path):
    a, b = tmp_path / "naive_rag__x", tmp_path / "mnimi__x"
    _arm(a, "naive_rag", {"q1": False, "q2": True}, {"q1": False, "q2": False})
    _arm(b, "mnimi", {"q1": True, "q2": True}, {"q1": True, "q2": True})
    out = tmp_path / "analyses"
    assert stats.main([str(a), str(b), "--judge", "gpt-4.1-2025-04-14", "--out", str(out)]) == 0
    saved = json.loads((out / "naive_rag__x__vs__mnimi__x__judge-gpt-4.1-2025-04-14.json")
                       .read_text(encoding="utf-8"))
    assert saved["b_second_wins"] == 2 and saved["c_first_wins"] == 0
    assert saved["source"].startswith(
        "per-question verdicts of the judge replay under gpt-4.1-2025-04-14")
    assert not (out / "naive_rag__x__vs__mnimi__x.json").exists()


def test_a_judge_selected_pair_refuses_two_judges(tmp_path, capsys):
    a, b = tmp_path / "naive_rag__x", tmp_path / "mnimi__x"
    _arm(a, "naive_rag", {"q1": False}, {"q1": False}, judge_hash="h1")
    _arm(b, "mnimi", {"q1": True}, {"q1": True}, judge_hash="h2")
    out = str(tmp_path / "o")
    assert stats.main([str(a), str(b), "--judge", "gpt-4.1-2025-04-14", "--out", out]) == 2
    assert "judge_hash" in capsys.readouterr().err

def test_tost_on_a_fixed_vector():
    r = stats.tost(b=30, c=26, n=500, margin=0.05)
    assert r["delta"] == pytest.approx(0.008)
    assert r["se"] == pytest.approx(math.sqrt(56 - 16 / 500) / 500)
    assert r["p_tost"] == pytest.approx(max(r["p_lower"], r["p_upper"]))
    assert r["equivalent"] is (r["p_tost"] < 0.05)
    degenerate = stats.tost(b=0, c=0, n=10, margin=0.1)
    assert degenerate["se"] == 0.0 and degenerate["equivalent"] is True


def test_save_pair_records_family_and_holm(tmp_path):
    record = {"first": {"run_id": "a"}, "second": {"run_id": "b"}, "p_exact_mcnemar": 0.01}
    path = stats.save_pair(record, tmp_path, family="F2", p_holm=0.02)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["family"] == "F2" and saved["p_holm"] == 0.02


def test_a_family_of_pairs_is_holm_corrected_and_saved(tmp_path):
    a, b, c = tmp_path / "naive_rag__x", tmp_path / "mnimi__x", tmp_path / "mnimi__y"
    _arm(a, "naive_rag", {"q1": False, "q2": True, "q3": False},
         {"q1": False, "q2": True, "q3": False})
    _arm(b, "mnimi", {"q1": True, "q2": True, "q3": True}, {"q1": True, "q2": True, "q3": True})
    _arm(c, "mnimi", {"q1": True, "q2": True, "q3": False}, {"q1": True, "q2": True, "q3": False})
    out = tmp_path / "analyses"
    assert stats.main([str(a), str(b), str(c), "--family", "F9", "--out", str(out)]) == 0
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json"))]
    assert saved and all(s["family"] == "F9" for s in saved)
    assert all(s["p_holm"] >= s["p_exact_mcnemar"] for s in saved)


def _arm_by_category(directory, system, verdicts, categories):
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({
        "pins": {"system": system}, "pins_hash": "p", "judge": {},
        "results": [{"question_id": q, "correct": c, "category": categories[q]}
                    for q, c in verdicts.items()],
    }), encoding="utf-8")


def test_save_pair_takes_a_file_name(tmp_path):
    record = {"first": {"run_id": "a"}, "second": {"run_id": "b"}, "p_exact_mcnemar": 0.01}
    path = stats.save_pair(record, tmp_path, name="multi-session")
    assert path == tmp_path / "multi-session.json"


def test_by_category_saves_one_holm_corrected_pair_per_category(tmp_path):
    categories = {"q1": "x", "q2": "x", "q3": "y", "q4": "y", "q5": "y"}
    a, b = tmp_path / "naive_rag__x", tmp_path / "mnimi__x"
    _arm_by_category(a, "naive_rag",
                     {"q1": False, "q2": False, "q3": True, "q4": False, "q5": True}, categories)
    _arm_by_category(b, "mnimi",
                     {"q1": True, "q2": True, "q3": False, "q4": False, "q5": True}, categories)
    out = tmp_path / "f2"
    assert stats.main([str(a), str(b), "--by-category", "--family", "F2", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.json")) == ["x.json", "y.json"]
    x = json.loads((out / "x.json").read_text(encoding="utf-8"))
    y = json.loads((out / "y.json").read_text(encoding="utf-8"))
    # b = the second run's wins, counted inside the category only
    assert (x["category"], x["n_pairs"], x["b_second_wins"], x["c_first_wins"]) == ("x", 2, 2, 0)
    assert (y["category"], y["n_pairs"], y["b_second_wins"], y["c_first_wins"]) == ("y", 3, 0, 1)
    assert (x["first"]["run_id"], x["second"]["run_id"]) == ("naive_rag__x", "mnimi__x")
    assert (x["first"]["correct"], x["second"]["correct"]) == (0, 2)
    assert x["family"] == y["family"] == "F2"
    assert x["p_exact_mcnemar"] == 0.5 and x["p_holm"] == 1.0  # Holm over the two categories


def test_by_category_needs_exactly_two_run_dirs(tmp_path, capsys):
    a = tmp_path / "naive_rag__x"
    _arm_by_category(a, "naive_rag", {"q1": True}, {"q1": "x"})
    assert stats.main([str(a), "--by-category", "--out", str(tmp_path / "f2")]) == 2
    assert "exactly two" in capsys.readouterr().err
    assert not (tmp_path / "f2").exists()
