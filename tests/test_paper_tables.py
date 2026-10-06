"""The paper's tables (Phase 7 D9, PHASE8 Task 16): exported from committed files."""

from __future__ import annotations

import json
from pathlib import Path

from evals import paper_tables

ROOT = Path(__file__).resolve().parents[1]


def _summary(directory, correct, n=500, low=0.8, high=0.9):
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(json.dumps({
        "correct": correct, "n": n, "score": f"{correct}/{n}",
        "overall": {"accuracy": correct / n, "ci_low": low, "ci_high": high},
        "by_category": {c: {"correct": 1, "n": 2} for c in paper_tables.CATEGORIES},
    }), encoding="utf-8")


def _published_summary(directory: Path) -> dict:
    """An arm's committed counts: summary.json, or results.json's summary for the Phase 5 arms."""
    path = directory / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    overall = results["summary"]["overall"]
    return {"correct": overall["correct"], "n": overall["n"]}


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_rows_come_from_the_files_and_absent_inputs_render_a_dash(tmp_path, monkeypatch):
    published, analyses = tmp_path / "published", tmp_path / "analyses"
    monkeypatch.setattr(paper_tables, "ARMS_OF_RECORD",
                        (("a__x", "arm a", "floor"), ("b__x", "arm b", "the system")))
    monkeypatch.setattr(paper_tables, "PAIRS_OF_RECORD", (("a__x", "b__x", "primary"),))
    _summary(published / "a__x", 100)
    _summary(published / "b__x", 400, low=0.765, high=0.834)
    analyses.mkdir()
    (analyses / "a__x__vs__b__x.json").write_text(json.dumps({
        "b_second_wins": 310, "c_first_wins": 10, "discordant": 320, "p_exact_mcnemar": 1.1e-06,
    }), encoding="utf-8")
    md = paper_tables.render_markdown(paper_tables.load(published, analyses))
    assert "| b__x | the system | 400/500 | 80.0 % | [76.5, 83.4] | — |" in md
    assert "| a__x → b__x | primary | 310 | 10 | 320 | 1.1e-06 | — | — | — | — |" in md


def test_a_pair_holds_iff_the_sign_survives_the_second_judge(tmp_path, monkeypatch):
    published, analyses = tmp_path / "published", tmp_path / "analyses"
    monkeypatch.setattr(paper_tables, "ARMS_OF_RECORD", (("a__x", "arm a", "floor"),))
    monkeypatch.setattr(paper_tables, "PAIRS_OF_RECORD",
                        (("a__x", "b__x", "kept"), ("a__x", "c__x", "turned")))
    analyses.mkdir(parents=True)
    suffix = f"__judge-{paper_tables.SECOND_JUDGE}"
    for second, first_judge, second_judge in (("b__x", (9, 4), (8, 6)), ("c__x", (9, 4), (5, 7))):
        for name, (b, c) in ((f"a__x__vs__{second}", first_judge),
                             (f"a__x__vs__{second}{suffix}", second_judge)):
            (analyses / f"{name}.json").write_text(json.dumps({
                "b_second_wins": b, "c_first_wins": c, "discordant": b + c,
                "p_exact_mcnemar": 0.5,
            }), encoding="utf-8")
    md = paper_tables.render_markdown(paper_tables.load(published, analyses))
    assert "| a__x → b__x | kept | 9 | 4 | 13 | 0.5 | 8 | 6 | 0.5 | yes |" in md
    assert "| a__x → c__x | turned | 9 | 4 | 13 | 0.5 | 5 | 7 | 0.5 | no |" in md


def test_the_per_category_family_is_read_from_its_saved_pairs(tmp_path):
    published, analyses = tmp_path / "published", tmp_path / "analyses"
    (analyses / "f2").mkdir(parents=True)
    (analyses / "f2" / "multi-session.json").write_text(json.dumps({
        "first": {"run_id": "naive"}, "second": {"run_id": "system"}, "category": "multi-session",
        "b_second_wins": 20, "c_first_wins": 5, "discordant": 25, "n_pairs": 133,
        "p_exact_mcnemar": 0.004, "p_holm": 0.024, "family": "F2",
    }), encoding="utf-8")
    data = paper_tables.load(published, analyses)
    md = paper_tables.render_markdown(data)
    assert "| multi-session | 133 | 20 | 5 | 25 | 0.004 | 0.024 |" in md
    (entry,) = paper_tables.families_record(data)["families"]["F2"]
    assert entry["file"] == "f2/multi-session.json" and entry["p_holm"] == 0.024
    assert entry["category"] == "multi-session" and entry["judge"] is None


def test_latex_is_a_booktabs_tabular_with_the_special_characters_escaped(tmp_path, monkeypatch):
    published = tmp_path / "published"
    monkeypatch.setattr(paper_tables, "ARMS_OF_RECORD", (("b__x", "arm b", "R&D #1"),))
    monkeypatch.setattr(paper_tables, "PAIRS_OF_RECORD", ())
    _summary(published / "b__x", 400, low=0.765, high=0.834)
    tex = paper_tables.render_latex(paper_tables.load(published, tmp_path / "analyses"))
    assert r"b\_\_x & R\&D \#1 & 400/500 & 80.0 \% & [76.5, 83.4] & --- \\" in tex
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == tex.count(r"\begin{tabular}")


def test_the_committed_tables_match_their_sources():
    published, analyses = ROOT / "results" / "published", ROOT / "analyses"
    md = paper_tables.render_markdown(paper_tables.load(published, analyses))
    assert "| naive_rag__500q_gpt4o → mnimi__500q_gpt4o | primary | 75 | 26 | 101 | 1.1e-06 |" in md
    for run_id, _label, role in paper_tables.ARMS_OF_RECORD:
        s = _published_summary(published / run_id)
        assert f"| {run_id} | {role} | {s['correct']}/{s['n']} |" in md
    assert "429/500" in md and "422/500" in md
    mem0 = _manifest(published / "mem0__500q_gpt4o")
    config = mem0["arm"]["config"]
    assert (f"| mem0__500q_gpt4o | {config['competitor_version']} | {config['competitor_llm']} | "
            f"{config['chunk_unit']} |") in md  # T7
    omega = _manifest(published / "omega__500q_gpt4o")
    assert (f"| omega__500q_gpt4o | {omega['code']['commit'][:7]} | — | "
            f"{omega['arm']['config_sha256'][:12]} |") in md  # T8
    assert md == paper_tables.render_markdown(paper_tables.load(published, analyses))


def test_the_exported_files_are_the_render_of_the_committed_sources():
    """``results/paper/`` is never edited by hand: it equals a fresh export, byte for byte (LF)."""
    data = paper_tables.load(ROOT / "results" / "published", ROOT / "analyses")
    paper = ROOT / "results" / "paper"
    assert (paper / "tables.md").read_text(encoding="utf-8") == paper_tables.render_markdown(data)
    assert (paper / "tables.tex").read_text(encoding="utf-8") == paper_tables.render_latex(data)
    assert json.loads((paper / "families.json").read_text(encoding="utf-8")) == \
        paper_tables.families_record(data)
    assert set(paper_tables.families_record(data)["families"]) >= {"F1", "F2"}
