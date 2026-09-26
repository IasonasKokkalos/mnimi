"""Inter-judge agreement and the blind label sheet (Phase 7 D8). Committed-file readers only."""

from __future__ import annotations

import csv
import json

import pytest
from evals import agreement


def _row(qid, *, run_id="a", predicted=None, first=True, second=False, category="multi-session"):
    return {"run_id": run_id, "question_id": qid, "category": category, "is_abstention": False,
            "question": f"Q {qid}?", "answer": f"gold {qid}",
            "predicted": predicted or f"answer {qid}",
            "first": first, "second": second}


def test_kappa_matches_the_textbook_value_and_the_degenerate_cases():
    assert agreement.cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)
    assert agreement.cohen_kappa(10, 0, 0, 10) == 1.0
    assert agreement.cohen_kappa(10, 0, 0, 0) == 1.0


def test_the_sheet_is_blind_and_deterministic():
    disagree = [_row(f"q{i}") for i in range(80)]
    agree = [_row(f"a{i}", second=True) for i in range(40)]
    rows, key = agreement.draw_sheet(disagree, agree, n_disagree=50, n_control=10, seed=0)
    assert (rows, key) == agreement.draw_sheet(disagree, agree, n_disagree=50, n_control=10, seed=0)
    assert len(rows) == 60 and sum(k["kind"] == "control" for k in key.values()) == 10
    for row in rows:
        assert tuple(row) == agreement.SHEET_FIELDS and row["human"] == ""
        assert "first" not in row and "second" not in row
    assert agreement.draw_sheet(disagree, agree, n_disagree=50, n_control=10, seed=1)[0] != rows


def test_identical_responses_across_arms_are_labelled_once():
    same = [_row("q1", run_id="a", predicted="Luna"), _row("q1", run_id="b", predicted="Luna")]
    rows, key = agreement.draw_sheet(same, [], n_disagree=50, n_control=0, seed=0)
    assert len(rows) == 1 and key[rows[0]["row_id"]]["run_ids"] == ["a", "b"]


def test_labels_are_read_strictly_and_scored_per_judge(tmp_path):
    rows, key = agreement.draw_sheet([_row("q1"), _row("q2", first=False, second=True)],
                                     [_row("a1", second=True)], n_disagree=2, n_control=1, seed=0)
    sheet, _ = agreement.write_sheet(tmp_path, rows, key)
    with pytest.raises(FileExistsError):
        agreement.write_sheet(tmp_path, rows, key)
    with pytest.raises(ValueError, match="L0"):
        agreement.read_labels(sheet)
    with open(sheet, encoding="utf-8-sig", newline="") as fh:
        filled = list(csv.DictReader(fh))
    for r in filled:
        r["human"] = "yes"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=agreement.SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(filled)
    result = agreement.score(agreement.read_labels(sheet), key)
    assert result["by_kind"]["disagree"]["n"] == 2 and result["by_kind"]["control"]["n"] == 1
    assert result["by_kind"]["disagree"]["first"]["agrees_with_human"] == 1
    assert result["by_kind"]["disagree"]["second"]["agrees_with_human"] == 1
    assert result["by_kind"]["control"]["first"]["agrees_with_human"] == 1


def test_the_report_pairs_each_runs_first_verdicts_with_its_replay(tmp_path):
    cases = (("r1", [True, False], [True, True]), ("r2", [True, True], [True, True]))
    for name, first, second in cases:
        d = tmp_path / name
        d.mkdir()
        (d / "results.json").write_text(json.dumps({"judge": {"judge_model": "gpt-4o-2024-08-06"},
                                                    "results": []}), encoding="utf-8")
        (d / "predictions.jsonl").write_text("".join(
            json.dumps({"question_id": f"q{i}", "verdicts": [{"correct": v}]}) + "\n"
            for i, v in enumerate(first)), encoding="utf-8")
        (d / "judge_replay_1.json").write_text(json.dumps({
            "judge": {"judge_model": "gpt-4.1-2025-04-14"}, "judge_hash": "h",
            "results": [{"question_id": f"q{i}", "correct": v} for i, v in enumerate(second)]}),
            encoding="utf-8")
    rep = agreement.report([tmp_path / "r1", tmp_path / "r2"], "gpt-4.1-2025-04-14")
    assert rep["judge_first"] == "gpt-4o-2024-08-06" and rep["judge_second"] == "gpt-4.1-2025-04-14"
    assert rep["arms"]["r1"]["ny"] == 1 and rep["pooled"]["n"] == 4 and rep["pooled"]["agree"] == 3
