"""``--verify-drift`` / ``python -m evals.drift``: the N/n-changed instrument."""

from __future__ import annotations

import json

import pytest
from evals import artifacts, drift


def _row(qid, predicted, tokens=100, category="single-session-user"):
    return {
        "question_id": qid,
        "question": f"q {qid}",
        "answer": "a",
        "category": category,
        "predicted": predicted,
        "reader_prompt_tokens": tokens,
        "tokens_dropped": 0,
        "truncated": False,
        "is_abstention": False,
    }


def _write_run(directory, rows, pins=None, fingerprints=None):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    if pins is not None:
        artifacts.write_pins(directory, pins)
    if fingerprints is not None:
        artifacts.write_reader_resolved(
            directory, {"system_fingerprints": fingerprints, "requests": len(rows)}
        )
    return directory


PINS = {"system": "mnimi", "k": 10, "reader_model": "m"}


def test_first_divergence_is_a_utf8_byte_offset():
    assert drift.first_divergence("abc", "abc") is None
    assert drift.first_divergence("abc", "abd") == 2
    assert drift.first_divergence("abc", "abcd") == 3  # proper prefix
    assert drift.first_divergence("é1", "é2") == 2  # é is two bytes


def test_identical_runs_report_zero_changed(tmp_path):
    rows = [_row("a", "x"), _row("b", "y")]
    ref = _write_run(tmp_path / "ref", rows, PINS, {"fp_1": 2})
    fresh = _write_run(tmp_path / "fresh", rows, PINS, {"fp_1": 1, "fp_2": 1})
    report = drift.compare(ref, fresh)
    assert (report["n"], report["changed"], report["prompt_tokens_changed"]) == (2, 0, 0)
    assert report["pins_checked"] is True
    assert report["system_fingerprints"] == {
        "reference": {"fp_1": 2},
        "fresh": {"fp_1": 1, "fp_2": 1},
    }
    assert drift.statement(report) == "drift: 0/2 predictions changed (byte-identical rows)"


def test_changed_rows_are_named_with_their_first_byte(tmp_path):
    ref = _write_run(tmp_path / "ref", [_row("a", "same"), _row("b", "The answer is 4")], PINS)
    fresh = _write_run(
        tmp_path / "fresh", [_row("a", "same"), _row("b", "The answer is 5", tokens=101)], PINS
    )
    report = drift.verify(ref, fresh)
    assert report["changed"] == 1 and report["prompt_tokens_changed"] == 1
    (row,) = report["rows"]
    assert row["question_id"] == "b"
    assert row["first_divergence_byte"] == len("The answer is ")
    assert row["reader_prompt_tokens"] == [100, 101]
    written = json.loads((fresh / drift.DRIFT_FILE).read_text(encoding="utf-8"))
    assert written["changed"] == 1
    assert "1/2 predictions changed" in drift.format_report(report)


def test_different_pins_are_refused_as_not_a_drift_pair(tmp_path):
    ref = _write_run(tmp_path / "ref", [_row("a", "x")], PINS)
    fresh = _write_run(tmp_path / "fresh", [_row("a", "x")], {**PINS, "k": 20})
    with pytest.raises(drift.DriftPairError, match=r"pins differ on \['k'\]"):
        drift.compare(ref, fresh)


def test_different_question_sets_are_refused(tmp_path):
    ref = _write_run(tmp_path / "ref", [_row("a", "x")], PINS)
    fresh = _write_run(tmp_path / "fresh", [_row("b", "x")], PINS)
    with pytest.raises(drift.DriftPairError, match="different questions"):
        drift.compare(ref, fresh)


def test_a_bare_predictions_file_compares_without_pins(tmp_path):
    # Tier 1 shape: a published predictions.jsonl on its own still pairs; the
    # report says the pins went unchecked.
    ref = _write_run(tmp_path / "ref", [_row("a", "x")])
    fresh = _write_run(tmp_path / "fresh", [_row("a", "y")], PINS)
    report = drift.compare(ref / "predictions.jsonl", fresh)
    assert report["pins_checked"] is False and report["changed"] == 1
    assert "pins not checked" in drift.format_report(report)


def test_cli_prints_the_statement_and_writes_into_the_fresh_dir(tmp_path, capsys):
    ref = _write_run(tmp_path / "ref", [_row("a", "x")], PINS)
    fresh = _write_run(tmp_path / "fresh", [_row("a", "x")], PINS)
    assert drift.main([str(ref), str(fresh)]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("drift: 0/1 predictions changed")
    assert (fresh / drift.DRIFT_FILE).exists()

    assert drift.main([str(ref)]) == 2
    assert drift.main([str(ref), str(tmp_path / "absent")]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_published_restart_pair_is_zero_drift():
    # The local family's measured Tier 2 evidence, now checked by the instrument
    # instead of by hand: 100/100 byte-identical across a daemon restart.
    from pathlib import Path

    published = Path("results/published")
    report = drift.compare(
        published / "mnimi__100q", published / "mnimi__100q_restart_2026-09-10"
    )
    assert (report["n"], report["changed"], report["prompt_tokens_changed"]) == (100, 0, 0)
    # The pair straddles the /4 -> /5 schema bump: every shared pin is equal,
    # the hashes are not, and the instrument judges on the pins.
    assert report["pins_checked"] is True
    assert report["artifact_schema"] == {
        "reference": "mnimi-eval-artifact/4", "fresh": "mnimi-eval-artifact/5"
    }
    assert report["pins_hash"]["reference"] != report["pins_hash"]["fresh"]
    # ...and three commits: reported, not refused, and named in the text.
    assert report["harness_git_sha"]["reference"] != report["harness_git_sha"]["fresh"]
    assert "harness commit differs" in drift.format_report(report)


def test_a_schema_bump_that_only_adds_a_pin_still_pairs(tmp_path):
    ref = _write_run(tmp_path / "ref", [_row("a", "x")], {**PINS, "artifact_schema": "x/4"})
    fresh = _write_run(
        tmp_path / "fresh", [_row("a", "x")], {**PINS, "artifact_schema": "x/5", "added": 1}
    )
    assert drift.compare(ref, fresh)["changed"] == 0
    assert drift.pins_differences(
        {"a": 1, "artifact_schema": 1, "harness_git_sha": "x"},
        {"a": 2, "artifact_schema": 2, "harness_git_sha": "y"},
    ) == ["a"]


def test_the_two_spellings_of_no_extractor_pair_but_a_real_extractor_does_not():
    # /7 wrote the extractor pins as None (nothing to hash); /8 writes "none" on the
    # v1 arm. The published k10 arm vs the 2026-09-14 p2base arm is one configuration.
    seven = {"extractor_model": None, "extractor_prompt_hash": None, "k": 10}
    eight = {"extractor_model": "none", "extractor_prompt_hash": "none", "k": 10}
    assert drift.pins_differences(seven, eight) == []
    assert drift.pins_differences(eight, seven) == []
    assert drift.pins_differences(seven, {**eight, "extractor_model": "Qwen/x@abc"}) == [
        "extractor_model"
    ]
    assert drift.pins_differences({"render_unit": None}, {"render_unit": "none"}) == [
        "render_unit"
    ], "only the extractor pins have two spellings of absent"
