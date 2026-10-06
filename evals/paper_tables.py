"""The paper's tables (Phase 7 D9, PHASE8 Task 16): exported from committed files.

``python -m evals.paper_tables`` reads ``results/published/`` and ``analyses/`` and
writes ``results/paper/{tables.md,tables.tex,families.json}``. Every cell is a value a
committed file already holds: ``summary.json`` (``results.json``'s summary for the
arms published before that file existed), the pair records, the audit records, the
drift report, the agreement, retest, bucket and label records, the manifests. Nothing
is re-judged and no pair is recomputed; the one count taken from rows is an arm's score
under the second judge, read from that judge's replay file. An absent input renders a
dash, so the exporter runs on any subset of the files.

The three cells no committed record carries — a third-party system's wall-clock
behaviour, the deferred arm, the cited row — are the module constants below, each
naming its source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stats import holm

SECOND_JUDGE = "gpt-4.1-2025-04-14"
PAPER_SYSTEM = "mnimi__500q_gpt4o_p6time"
BAR = "naive_rag__500q_gpt4o"
DRIFT_ARM = "mnimi__500q_gpt4o_p6time_drift_2026-09-26"
DASH = "—"

ARMS_OF_RECORD = (  # (run_id, label for T2's columns, role for T1)
    ("no_memory__500q_gpt4o", "no_memory", "floor"),
    ("naive_rag__500q_gpt4o", "naive_rag", "strong K=V baseline"),
    ("mnimi__500q_gpt4o", "mnimi P5", "the Phase 5 verdict arm"),
    ("mnimi__500q_gpt4o_decay", "mnimi+decay", "SPEC's decay ablation"),
    ("mnimi__500q_gpt4o_p6time", "mnimi L1", "the shipped configuration (L1 alone)"),
    ("mnimi__500q_gpt4o_p6turns", "mnimi L3", "L3: turns, extractor on"),
    ("mnimi__500q_gpt4o_p6combo", "mnimi L1+L3", "the Phase 6 headline"),
    ("oracle__500q_gpt4o", "oracle", "evidence-availability bound"),
    ("mem0__500q_gpt4o", "Mem0 OSS", "third-party, LLM-routed writes"),
    ("omega__500q_gpt4o", "OMEGA retrieval", "third-party retrieval over verbatim rounds"),
)
PAIRS_OF_RECORD = (  # (first, second, kind); b = the second arm's wins, as analyses/ stores it
    ("naive_rag__500q_gpt4o", "mnimi__500q_gpt4o", "primary"),
    ("no_memory__500q_gpt4o", "mnimi__500q_gpt4o", "secondary"),
    ("oracle__500q_gpt4o", "mnimi__500q_gpt4o", "secondary"),
    ("mnimi__500q_gpt4o", "mnimi__500q_gpt4o_decay", "decay ablation"),
    ("mnimi__500q_gpt4o", "mnimi__500q_gpt4o_p6time", "Phase 6 L1"),
    ("mnimi__500q_gpt4o", "mnimi__500q_gpt4o_p6turns", "Phase 6 L3"),
    ("mnimi__500q_gpt4o", "mnimi__500q_gpt4o_p6combo", "Phase 6 L1+L3"),
    ("naive_rag__500q_gpt4o", "mnimi__500q_gpt4o_p6time", "the paper system vs the bar"),
    ("mnimi__500q_gpt4o_p6time", "mem0__500q_gpt4o", "competitor"),
    ("mnimi__500q_gpt4o_p6time", "omega__500q_gpt4o", "competitor"),
    ("naive_rag__500q_gpt4o", "mem0__500q_gpt4o", "competitor vs the bar"),
    ("naive_rag__500q_gpt4o", "omega__500q_gpt4o", "competitor vs the bar"),
)
CATEGORIES = ("single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "knowledge-update", "temporal-reasoning")
CITED = ({"system": "full_history", "reader": "GPT-4o + Chain-of-Note", "score": "64.0 %",
          "source": "LongMemEval (Wu et al., 2024), Fig. 3b, LongMemEval-S; cited, never run"},)
THIRD_PARTY = ("mem0__500q_gpt4o", "omega__500q_gpt4o")
# What each third-party system does with the wall clock: no manifest field records it, so
# the cell is the adapter's own disclosure (the module docstrings named here).
WALL_CLOCK = {
    "mem0__500q_gpt4o": "created_at stamped from the wall clock, not read by search; the "
                        "extraction prompt grounds relative dates on the machine date "
                        "(evals/systems/mem0_oss.py)",
    "omega__500q_gpt4o": "created_at, access times and decay read the wall clock; created_at "
                         "backdated to the session date (evals/systems/omega_retrieval.py)",
}
DEFERRED = ({"system": "agentmemory v4 @ 3aa3b83",
             "reason": "deferred: its smoke measured 20-28 min per history "
                       "(DECISIONS 2026-09-27); no n=500 arm"},)
BUCKETS = ("both_right", "oracle_right_system_wrong", "both_wrong", "system_right_oracle_wrong")
FAMILIES_SCHEMA = "mnimi-paper-families/1"


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _summary(directory: Path) -> dict | None:
    """``summary.json``, or ``results.json``'s summary for an arm published before it existed."""
    summary = _json(directory / "summary.json")
    if summary is None:
        results = _json(directory / "results.json")
        summary = (results or {}).get("summary")
    if summary is None:
        return None
    overall = summary.get("overall") or {}
    return {
        "correct": summary.get("correct", overall.get("correct")),
        "n": summary.get("n", overall.get("n")),
        "overall": overall,
        "by_category": summary.get("by_category") or {},
    }


def _second(directory: Path) -> dict | None:
    """The arm's score under the second judge, counted from that judge's one replay file."""
    found = []
    for path in sorted(directory.glob("judge_replay_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("judge") or {}).get("judge_model") == SECOND_JUDGE:
            found.append(payload)
    if len(found) != 1:
        return None
    rows = found[0]["results"]
    return {"correct": sum(1 for row in rows if row["correct"]), "n": len(rows)}


def _pair(analyses: Path, first: str, second: str) -> tuple[dict | None, dict | None]:
    stem = f"{first}__vs__{second}"
    return (_json(analyses / f"{stem}.json"),
            _json(analyses / f"{stem}__judge-{SECOND_JUDGE}.json"))


def load(published: Path, analyses: Path) -> dict:
    """Everything the tables show, as one plain dict read from the two committed trees."""
    published, analyses = Path(published), Path(analyses)
    arms = {}
    for run_id in [a[0] for a in ARMS_OF_RECORD] + [DRIFT_ARM]:
        directory = published / run_id
        arms[run_id] = {"summary": _summary(directory), "second": _second(directory),
                        "manifest": _json(directory / "manifest.json")}
    pairs = []
    for first, second, kind in PAIRS_OF_RECORD:
        record, record_second = _pair(analyses, first, second)
        pairs.append({"first": first, "second": second, "kind": kind,
                      "record": record, "record_second": record_second})
    drift = _json(analyses / f"{PAPER_SYSTEM}__drift__{DRIFT_ARM}.json")
    null_pair, null_pair_second = _pair(analyses, PAPER_SYSTEM, DRIFT_ARM)
    f2 = {}
    for category in CATEGORIES:
        record = _json(analyses / "f2" / f"{category}.json")
        if record is not None:
            f2[category] = record
    families: dict[str, list[dict]] = {}
    paths = [*analyses.glob("*.json"), *(analyses / "f2").glob("*.json")]
    for path in sorted(paths, key=lambda p: p.relative_to(analyses).as_posix()):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "family" not in record or "b_second_wins" not in record:
            continue  # not a pair record, or a pair that names no family
        judge = path.stem.split("__judge-", 1)[1] if "__judge-" in path.stem else None
        families.setdefault(record["family"], []).append({
            "file": path.relative_to(analyses).as_posix(),
            "first": record["first"]["run_id"], "second": record["second"]["run_id"],
            "category": record.get("category"), "judge": judge,
            "b_second_wins": record["b_second_wins"], "c_first_wins": record["c_first_wins"],
            "n_pairs": record["n_pairs"], "p_exact_mcnemar": record["p_exact_mcnemar"],
            "p_holm_saved": record.get("p_holm"),
        })
    # A pair saved by its own invocation carries a one-test Holm p, so the family's correction
    # is taken here, across every saved pair of one family under one judge.
    for entries in families.values():
        for judge in {e["judge"] for e in entries}:
            group = [e for e in entries if e["judge"] == judge]
            adjusted = holm({e["file"]: e["p_exact_mcnemar"] for e in group})
            for e in group:
                e["tests"], e["p_holm"] = len(group), adjusted[e["file"]]
    return {
        "arms": arms,
        "pairs": pairs,
        "audits": [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted((analyses / "audits").glob("*__tier1.json"))],
        "drift": drift, "null_pair": null_pair, "null_pair_second": null_pair_second,
        "agreement": _json(analyses / "judge_agreement" / "agreement.json"),
        "labels": _json(analyses / "judge_agreement" / "label_score.json"),
        "retest": _json(analyses / "judge_retest" / "retest.json"),
        "buckets": _json(analyses / "judge_retest" / "buckets.json"),
        "f2": f2,
        "families": families,
        "cited": CITED,
    }


def families_record(data: dict) -> dict:
    """``results/paper/families.json``: every saved pair that names a family.

    ``p_holm`` is Holm's correction across the ``tests`` pairs of that family under that
    judge; ``p_holm_saved`` is what the pair's own record holds.
    """
    return {"schema": FAMILIES_SCHEMA,
            "families": {name: data["families"][name] for name in sorted(data["families"])}}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(low: float, high: float) -> str:
    return f"[{_pct(low)}, {_pct(high)}]"


def _p(p: float | None) -> str:
    return DASH if p is None else f"{p:.2g}"


def _score(side: dict | None) -> str:
    return DASH if not side or side.get("correct") is None else f"{side['correct']}/{side['n']}"


def _bcp(record: dict | None) -> str:
    if record is None:
        return DASH
    return (f"{record['b_second_wins']} / {record['c_first_wins']} / "
            f"{_p(record['p_exact_mcnemar'])}")


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _order(run_id: str) -> tuple[int, str]:
    ids = [a[0] for a in ARMS_OF_RECORD]
    return (ids.index(run_id) if run_id in ids else len(ids), run_id)


def _t1(data: dict) -> dict:
    rows = []
    for run_id, _label, role in ARMS_OF_RECORD:
        arm = data["arms"][run_id]
        s = arm["summary"]
        if s is None or "accuracy" not in s["overall"]:
            rows.append([run_id, role, DASH, DASH, DASH, DASH])
            continue
        o = s["overall"]
        rows.append([run_id, role, _score(s), f"{_pct(o['accuracy'])} %",
                     _ci(o["ci_low"], o["ci_high"]), _score(arm["second"])])
    return {"title": "T1 — accuracy, n=500, reader = judge = gpt-4o-2024-08-06",
            "header": ["arm", "role", "correct / n", "%", "Wilson 95 %",
                       f"under {SECOND_JUDGE}"],
            "rows": rows}


def _t2(data: dict) -> dict:
    rows = []
    for category in CATEGORIES:
        cells, n = [], DASH
        for run_id, _label, _role in ARMS_OF_RECORD:
            s = data["arms"][run_id]["summary"]
            cell = (s or {}).get("by_category", {}).get(category)
            cells.append(DASH if cell is None else f"{cell['correct']}/{cell['n']}")
            if cell is not None and n == DASH:
                n = str(cell["n"])
        rows.append([category, n, *cells])
    return {"title": "T2 — per category (correct / n)",
            "header": ["category", "n", *[label for _run, label, _role in ARMS_OF_RECORD]],
            "rows": rows}


def _t3(data: dict) -> dict:
    rows = []
    for pair in data["pairs"]:
        r, r2 = pair["record"], pair["record_second"]
        row = [f"{pair['first']} → {pair['second']}", pair["kind"]]
        row += ([DASH] * 4 if r is None else
                [str(r["b_second_wins"]), str(r["c_first_wins"]), str(r["discordant"]),
                 _p(r["p_exact_mcnemar"])])
        row += ([DASH] * 3 if r2 is None else
                [str(r2["b_second_wins"]), str(r2["c_first_wins"]), _p(r2["p_exact_mcnemar"])])
        if r is None or r2 is None:
            row.append(DASH)
        else:
            same = (_sign(r["b_second_wins"] - r["c_first_wins"])
                    == _sign(r2["b_second_wins"] - r2["c_first_wins"]))
            row.append("yes" if same else "no")
        rows.append(row)
    return {"title": "T3 — the pairs of record (b = the second arm's wins; exact McNemar)",
            "header": ["comparison", "kind", "b", "c", "discordant", "p",
                       "b₂", "c₂", "p₂", "holds"],
            "note": f"b₂, c₂, p₂: the same pair under {SECOND_JUDGE}; a pair holds iff "
                    "sign(b − c) is the same under both judges (Phase 7 D7).",
            "rows": rows}


def _t4(data: dict) -> list[dict]:
    audits = []
    for a in sorted(data["audits"], key=lambda a: _order(a["run_id"])):
        audits.append([a["run_id"], _score(a["published"]), _score(a["recomputed"]),
                       f"{len(a['flips']['to_wrong'])} / {len(a['flips']['to_correct'])}",
                       str(a["cost"]["judge_calls"]), a["verdict"]])
    drift, null, null2 = data["drift"], data["null_pair"], data["null_pair_second"]
    tier2 = [[DASH] * 5] if drift is None else [[
        f"{PAPER_SYSTEM} → {DRIFT_ARM}", f"{drift['changed']}/{drift['n']}",
        f"{drift['prompt_tokens_changed']}/{drift['n']}", _bcp(null), _bcp(null2)]]
    agreement, judges = data["agreement"], []
    if agreement is not None:
        entries = sorted(agreement["arms"].items(), key=lambda kv: _order(kv[0]))
        for name, e in [*entries, ("pooled", agreement["pooled"])]:
            judges.append([name, f"{e['agree']}/{e['n']}", f"{_pct(e['rate'])} %",
                           _ci(e["rate_low"], e["rate_high"]), f"{e['kappa']:.3f}",
                           f"{e['yy']} / {e['yn']} / {e['ny']} / {e['nn']}"])
    retest, c2 = data["retest"], []
    if retest is not None:
        for name, e in sorted(retest["arms"].items(), key=lambda kv: _order(kv[0])):
            c2.append([name, f"{e['unstable']}/{e['n']}", _ci(e["rate_low"], e["rate_high"]),
                       ", ".join(str(s) for s in e["score_by_grading"]),
                       f"{e['flips_to_wrong']} / {e['flips_to_right']}"])
    labels, human = data["labels"], []
    if labels is not None:
        for kind, name in (("disagree", "the judges disagree"), ("control", "the judges agree")):
            e = labels["by_kind"][kind]
            row = [name, str(e["n"])]
            for judge in ("first", "second"):
                side = e[judge]
                row += [f"{side['agrees_with_human']}/{e['n']}",
                        _ci(side["rate_low"], side["rate_high"]) if "rate" in side else DASH]
            human.append(row)
    first_judge = (agreement or {}).get("judge_first", "the first judge")
    return [
        {"title": "T4a — the instrument, Tier 1: the cold audits",
         "header": ["arm", "published", "recomputed", "flips to wrong / to right",
                    "fresh judge calls", "verdict"],
         "rows": audits or [[DASH] * 6]},
        {"title": "T4b — the instrument, Tier 2: the drift pair of the shipped configuration",
         "header": ["reference → fresh", "texts changed", "prompt tokens changed",
                    "b / c / p (the null pair)", f"b / c / p under {SECOND_JUDGE}"],
         "rows": tier2},
        {"title": f"T4c — the instrument, the two judges ({first_judge} vs {SECOND_JUDGE})",
         "header": ["arm", "agree / n", "rate", "Wilson 95 %", "κ", "yy / yn / ny / nn"],
         "note": "yy = both yes, yn = the first judge yes and the second no, ny = the reverse, "
                 "nn = both no.",
         "rows": judges or [[DASH] * 6]},
        {"title": "T4d — the instrument, judge test-retest (C2): three cache-off replays "
                  "by the first judge",
         "header": ["arm", "judge-unstable rows / n", "Wilson 95 %",
                    "score by grading (first, then the replays)", "flips to wrong / to right"],
         "rows": c2 or [[DASH] * 5]},
        {"title": "T4e — the instrument, the human labels (60 rows, blind)",
         "header": ["rows where", "n", "human agrees with the first judge", "Wilson 95 %",
                    "human agrees with the second judge", "Wilson 95 %"],
         "rows": human or [[DASH] * 6]},
    ]


def _t5(data: dict) -> dict:
    return {"title": "T5 — cited, not run",
            "header": ["system", "reader", "score", "source"],
            "rows": [[c["system"], c["reader"], c["score"], c["source"]] for c in data["cited"]]}


def _t6(data: dict) -> list[dict]:
    buckets, rows = data["buckets"], []
    if buckets is not None:
        named = [(c, buckets["by_category"].get(c)) for c in (*CATEGORIES, "abstention")]
        for name, e in [*named, ("overall", buckets["overall"])]:
            if e is None:
                continue
            rows.append([name, str(e["n_stable"]), *[str(e[b]) for b in BUCKETS],
                         str(e["n_unstable"])])
    f2 = []
    for category in CATEGORIES:
        r = data["f2"].get(category)
        if r is not None:
            f2.append([category, str(r["n_pairs"]), str(r["b_second_wins"]),
                       str(r["c_first_wins"]), str(r["discordant"]),
                       _p(r["p_exact_mcnemar"]), _p(r.get("p_holm"))])
    sides = {(r["first"]["run_id"], r["second"]["run_id"]) for r in data["f2"].values()}
    pair = " → ".join(sides.pop()) if len(sides) == 1 else DASH
    system = (buckets or {}).get("system", PAPER_SYSTEM)
    oracle = (buckets or {}).get("oracle", "the oracle")
    return [
        {"title": f"T6a — the accounting (C1): {system} against {oracle}, judge-stable rows",
         "header": ["category", "n stable", "both right", "oracle right, system wrong",
                    "both wrong", "system right, oracle wrong", "judge-unstable"],
         "note": "The unanswerable (abstention) rows are counted in the abstention row and "
                 "not in their category's row, so the rows above overall partition the "
                 "benchmark and a category's n here is smaller than in T2.",
         "rows": rows or [[DASH] * 7]},
        {"title": f"T6b — the primary per category (F2): {pair}",
         "header": ["category", "n", "b", "c", "discordant", "p",
                    f"p (Holm, {len(data['f2'])} tests)"],
         "rows": f2 or [[DASH] * 7]},
    ]


def _t7(data: dict) -> dict:
    by_pair = {(p["first"], p["second"]): p["record"] for p in data["pairs"]}
    rows = []
    for run_id in THIRD_PARTY:
        arm = data["arms"].get(run_id) or {}
        config = ((arm.get("manifest") or {}).get("arm") or {}).get("config") or {}
        s = arm.get("summary")
        o = (s or {}).get("overall") or {}
        rows.append([
            run_id, config.get("competitor_version", DASH), config.get("competitor_llm", DASH),
            config.get("chunk_unit", DASH), config.get("competitor_embedder", DASH),
            WALL_CLOCK.get(run_id, DASH), _score(s),
            _ci(o["ci_low"], o["ci_high"]) if o else DASH,
            _bcp(by_pair.get((PAPER_SYSTEM, run_id))), _bcp(by_pair.get((BAR, run_id))),
        ])
    for d in DEFERRED:
        rows.append([d["system"], *[DASH] * 4, d["reason"], *[DASH] * 4])
    return {"title": "T7 — the third-party systems, through this harness under these pins",
            "header": ["arm", "version", "write-side LLM", "unit", "embedder", "wall clock",
                       "correct / n", "Wilson 95 %", f"b / c / p vs {PAPER_SYSTEM}",
                       f"b / c / p vs {BAR}"],
            "note": "b = the third-party arm's wins. Paired readings, never a rank. unit is the "
                    "manifest's chunk_unit, the units the system itself stores; in the OMEGA arm "
                    "those are naive_rag's rounds (its role in T1).",
            "rows": rows}


def _t8(data: dict) -> dict:
    rows = []
    for run_id in [a[0] for a in ARMS_OF_RECORD] + [DRIFT_ARM]:
        m = data["arms"][run_id]["manifest"]
        if m is None:
            rows.append([run_id, *[DASH] * 7])
            continue
        code, arm, judge = m.get("code") or {}, m.get("arm") or {}, m.get("judge") or {}
        sha = arm.get("config_sha256")
        hardware = (m.get("environment") or {}).get("hardware") or {}
        rows.append([
            run_id, str(code.get("commit", DASH))[:7], code.get("tag") or DASH,
            sha[:12] if sha else "pre-rule",
            ", ".join(sorted((m.get("reader") or {}).get("served_models") or {})) or DASH,
            judge.get("model", DASH), str(judge.get("replay_count", DASH)),
            hardware.get("gpu_model") or "API",
        ])
    return {"title": "T8 — provenance: one row per run the tables read",
            "header": ["run_id", "commit", "tag", "config sha256", "reader served models",
                       "judge", "replays", "machine"],
            "note": "tag is the exact tag on the run's commit (— when the commit carries none); "
                    "pre-rule = run before a committed config was required; UNKNOWN is what the "
                    "manifest itself records; the reader and the judge are served over the API, "
                    "the machine is where the contexts were built.",
            "rows": rows}


def tables(data: dict) -> list[dict]:
    """T1–T8 in order, each ``{title, header, rows[, note]}`` of plain strings."""
    return [_t1(data), _t2(data), _t3(data), *_t4(data), _t5(data), *_t6(data), _t7(data),
            _t8(data)]


def render_markdown(data: dict) -> str:
    out = ["# The paper's tables", "",
           "Exported by `python -m evals.paper_tables` from `results/published/` and `analyses/`; "
           "never edited by hand.", ""]
    for table in tables(data):
        out += [f"## {table['title']}", ""]
        out.append("| " + " | ".join(table["header"]) + " |")
        out.append("| " + " | ".join("---" for _ in table["header"]) + " |")
        out += ["| " + " | ".join(row) + " |" for row in table["rows"]]
        if table.get("note"):
            out += ["", table["note"]]
        out.append("")
    return "\n".join(out)


_LATEX = {
    "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
    "→": r"$\rightarrow$", "κ": r"$\kappa$", "₂": r"$_2$", "−": "$-$", "—": "---",
}


def _tex(cell: str) -> str:
    """One pass over the characters, so no replacement is itself escaped."""
    return "".join(_LATEX.get(ch, ch) for ch in cell)


def render_latex(data: dict) -> str:
    out = ["% The paper's tables: exported by `python -m evals.paper_tables`; "
           "never edited by hand.", ""]
    for table in tables(data):
        out.append(f"% {table['title']}")
        if table.get("note"):
            out.append(f"% {table['note']}")
        out.append(r"\begin{tabular}{" + "l" * len(table["header"]) + "}")
        out.append(r"\toprule")
        out.append(" & ".join(_tex(h) for h in table["header"]) + r" \\")
        out.append(r"\midrule")
        out += [" & ".join(_tex(cell) for cell in row) + r" \\" for row in table["rows"]]
        out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.paper_tables")
    parser.add_argument("--published", default="results/published")
    parser.add_argument("--analyses", default="analyses")
    parser.add_argument("--out", default="results/paper")
    args = parser.parse_args(argv)
    data = load(Path(args.published), Path(args.analyses))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "tables.md": render_markdown(data),
        "tables.tex": render_latex(data),
        "families.json": json.dumps(families_record(data), indent=2, sort_keys=True) + "\n",
    }
    for name, text in written.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    for table in tables(data):
        print(f"{table['title'].split(' — ')[0]}: {len(table['rows'])} rows")
    print(f"wrote {', '.join(str(out / name) for name in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
