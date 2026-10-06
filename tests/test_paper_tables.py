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


def _pair_file(path, first, second, b, c, p, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "first": {"run_id": first}, "second": {"run_id": second}, "b_second_wins": b,
        "c_first_wins": c, "discordant": b + c, "n_pairs": 100, "p_exact_mcnemar": p, **extra,
    }), encoding="utf-8")


def test_the_per_category_family_is_read_from_its_saved_pairs(tmp_path):
    published, analyses = tmp_path / "published", tmp_path / "analyses"
    _pair_file(analyses / "f2" / "multi-session.json", "naive", "system", 20, 5, 0.004,
               category="multi-session", family="F2", p_holm=0.008)
    _pair_file(analyses / "f2" / "knowledge-update.json", "naive", "system", 7, 3, 0.34,
               category="knowledge-update", family="F2", p_holm=0.34)
    data = paper_tables.load(published, analyses)
    md = paper_tables.render_markdown(data)
    # the title names the pair and the number of tests the records themselves hold
    assert "T6b — the primary per category (F2): naive → system" in md
    assert "| category | n | b | c | discordant | p | p (Holm, 2 tests) |" in md
    assert "| multi-session | 100 | 20 | 5 | 25 | 0.004 | 0.008 |" in md
    entries = paper_tables.families_record(data)["families"]["F2"]
    assert [e["file"] for e in entries] == ["f2/knowledge-update.json", "f2/multi-session.json"]
    assert all(e["tests"] == 2 and e["judge"] is None for e in entries)


def test_a_family_is_holm_corrected_across_its_saved_pairs_per_judge(tmp_path):
    """A pair saved by its own invocation carries a one-test Holm p; the family's is recomputed."""
    published, analyses = tmp_path / "published", tmp_path / "analyses"
    judged = f"a__vs__b__judge-{paper_tables.SECOND_JUDGE}.json"
    _pair_file(analyses / "a__vs__b.json", "a", "b", 9, 1, 0.01, family="F9", p_holm=0.01)
    _pair_file(analyses / "a__vs__c.json", "a", "c", 8, 2, 0.04, family="F9", p_holm=0.04)
    _pair_file(analyses / judged, "a", "b", 9, 2, 0.03, family="F9", p_holm=0.03)
    _pair_file(analyses / "a__vs__d.json", "a", "d", 5, 5, 1.0)  # names no family: not listed
    (analyses / "note.json").write_text(json.dumps({"family": "F9", "p": 1}), encoding="utf-8")
    data = paper_tables.load(published, analyses)
    by_file = {e["file"]: e for e in paper_tables.families_record(data)["families"]["F9"]}
    assert set(by_file) == {"a__vs__b.json", "a__vs__c.json", judged}
    first, second, other = by_file["a__vs__b.json"], by_file["a__vs__c.json"], by_file[judged]
    assert (first["tests"], first["p_holm"], first["p_holm_saved"]) == (2, 0.02, 0.01)
    assert (second["tests"], second["p_holm"], second["p_holm_saved"]) == (2, 0.04, 0.04)
    assert (other["judge"], other["tests"], other["p_holm"]) == (paper_tables.SECOND_JUDGE, 1, 0.03)


def test_the_second_judges_score_is_counted_from_that_judges_replay(tmp_path, monkeypatch):
    published = tmp_path / "published"
    monkeypatch.setattr(paper_tables, "ARMS_OF_RECORD", (("b__x", "arm b", "the system"),))
    monkeypatch.setattr(paper_tables, "PAIRS_OF_RECORD", ())
    _summary(published / "b__x", 400, low=0.765, high=0.834)
    for n, model, verdicts in ((1, "gpt-4o-2024-08-06", (True, True, True)),
                               (2, paper_tables.SECOND_JUDGE, (True, False, True)),
                               (3, "gpt-4o-2024-08-06", (False, False, False))):
        (published / "b__x" / f"judge_replay_{n}.json").write_text(json.dumps({
            "judge": {"judge_model": model},
            "results": [{"question_id": str(i), "correct": c} for i, c in enumerate(verdicts)],
        }), encoding="utf-8")
    md = paper_tables.render_markdown(paper_tables.load(published, tmp_path / "analyses"))
    assert "| b__x | the system | 400/500 | 80.0 % | [76.5, 83.4] | 2/3 |" in md


def test_latex_is_a_booktabs_tabular_with_the_special_characters_escaped(tmp_path, monkeypatch):
    published = tmp_path / "published"
    monkeypatch.setattr(paper_tables, "ARMS_OF_RECORD",
                        (("b__x", "arm b", "R&D #1 {a}~$x^2<3>"),))
    monkeypatch.setattr(paper_tables, "PAIRS_OF_RECORD", ())
    _summary(published / "b__x", 400, low=0.765, high=0.834)
    tex = paper_tables.render_latex(paper_tables.load(published, tmp_path / "analyses"))
    role = (r"R\&D \#1 \{a\}\textasciitilde{}\$x\textasciicircum{}2"
            r"\textless{}3\textgreater{}")
    assert rf"b\_\_x & {role} & 400/500 & 80.0 \% & [76.5, 83.4] & --- \\" in tex
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


def test_the_per_category_pairs_sum_to_the_paper_systems_pair_of_record():
    analyses = ROOT / "analyses"
    name = f"{paper_tables.BAR}__vs__{paper_tables.PAPER_SYSTEM}.json"
    pair = json.loads((analyses / name).read_text(encoding="utf-8"))
    cells = [json.loads((analyses / "f2" / f"{c}.json").read_text(encoding="utf-8"))
             for c in paper_tables.CATEGORIES]
    assert sum(c["b_second_wins"] for c in cells) == pair["b_second_wins"]
    assert sum(c["c_first_wins"] for c in cells) == pair["c_first_wins"]
    assert sum(c["n_pairs"] for c in cells) == pair["n_pairs"] == 500
    md = paper_tables.render_markdown(paper_tables.load(ROOT / "results" / "published", analyses))
    assert (f"| {paper_tables.BAR} → {paper_tables.PAPER_SYSTEM} | the paper system vs the bar | "
            f"{pair['b_second_wins']} | {pair['c_first_wins']} | {pair['discordant']} |") in md


def test_the_accounting_rows_partition_the_benchmark():
    """T6a: an abstention row is counted once, in the abstention row, not in its category's."""
    path = ROOT / "analyses" / "judge_retest" / "buckets.json"
    buckets = json.loads(path.read_text(encoding="utf-8"))
    rows = [buckets["by_category"][c] for c in (*paper_tables.CATEGORIES, "abstention")]
    for key in ("n_stable", "n_unstable", *paper_tables.BUCKETS):
        assert sum(row[key] for row in rows) == buckets["overall"][key]
    md = paper_tables.render_markdown(
        paper_tables.load(ROOT / "results" / "published", ROOT / "analyses"))
    assert "not in their category's row" in md and "also count in their own category" not in md


def test_the_exported_files_are_the_render_of_the_committed_sources():
    """``results/paper/`` is never edited by hand: it equals a fresh export, byte for byte (LF)."""
    data = paper_tables.load(ROOT / "results" / "published", ROOT / "analyses")
    paper = ROOT / "results" / "paper"
    assert (paper / "tables.md").read_text(encoding="utf-8") == paper_tables.render_markdown(data)
    assert (paper / "tables.tex").read_text(encoding="utf-8") == paper_tables.render_latex(data)
    assert json.loads((paper / "families.json").read_text(encoding="utf-8")) == \
        paper_tables.families_record(data)
    assert set(paper_tables.families_record(data)["families"]) >= {"F1", "F2"}
