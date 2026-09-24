"""The run-documentation rule (2026-09-22): manifest, registry, replays, retrieved ids, analyses.

Runs under the ``[dev]`` extra alone: fakes for the reader and the judge, no
network, no dataset.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evals import __main__ as evals_main
from evals import artifacts, manifest, publish, stats
from evals.runner import Prediction, Result, ingest_and_context
from evals.systems.full_history import FullHistorySystem
from evals.systems.mnimi import MnimiSystem
from evals.systems.naive_rag import NaiveRagSystem
from evals.systems.no_memory import NoMemorySystem

from mnimi.embeddings import HashingEmbedder

PINS = {
    "artifact_schema": artifacts.ARTIFACT_SCHEMA,
    "harness_git_sha": "abc123",
    "system": "mnimi",
    "k": 10,
    "reader_num_ctx": 128000,
    "reader_answer_reserve": 800,
    "reader_transport": "openai",
    "reader_model": "gpt-4o-2024-08-06",
    "reader_seed": 0,
    "reader_prompt_version": "mnimi-con-v1",
    "dataset_file": ".data/longmemeval_s_cleaned.json",
    "dataset_sha256": "d6f2",
    "limit": 2,
    "embedder_name": "BAAI/bge-small-en-v1.5",
    "embedder_revision": "5c38",
    "embed_template_hash": "e",
    "render_template_hash": "r",
    "dedup_cosine_threshold": 0.95,
    "dedup_scope": "session",
    "chunk_tokens": 0,
    "consolidate": False,
}
JUDGE = {
    "judge_model": "gpt-4o-2024-08-06", "judge_prompt_version": "longmemeval-paper-v3",
    "judge_prompt_hash": "j", "judge_temperature": 0, "judge_max_tokens": 10,
}
COST = {
    "reader_usd": 1.0, "judge_usd": 0.1, "tokens_in": 10, "tokens_out": 2,
    "wall_clock_s": 3.0, "ingest_s": 1.0, "llm_calls_at_write": 0,
}


def _manifest(**overrides):
    kwargs = dict(
        run_id="r1", purpose="a test", claim="none", rule_commit=None, pins=PINS,
        porcelain="", served_models={"gpt-4o-2024-08-06": 2}, served_fingerprints={"fp": 2},
        judge=JUDGE, replay_count=0, question_ids=["a", "b"],
        environment={"gpu_model": "G", "driver_version": "1", "cuda_version": "13"},
        cost=COST, outputs={},
    )
    kwargs.update(overrides)
    return manifest.build(**kwargs)


def test_a_full_manifest_is_complete_and_clean():
    m = manifest.finalize(_manifest(), provisional=[])
    assert manifest.check(m) == ([], [])
    assert m["status"] == "complete"
    assert m["code"]["clean_tree"] is True and m["code"]["commit"] == "abc123"
    assert m["code"]["mnimi_version"] == manifest.mnimi_version()
    assert m["dataset"]["n"] == 2
    assert m["dataset"]["question_ids_sha256"] == manifest.question_ids_sha256(["a", "b"])
    assert m["arm"]["config"]["token_budget"] == 128000
    assert m["arm"]["config"]["dedup"]["on"] is True
    assert m["reader"]["temperature"] == 0 and m["reader"]["max_tokens"] == 800
    assert m["environment"]["daemon_flags"]["daemon"].startswith("none (API family")


def test_missing_and_unknown_fields_are_reported_separately():
    m = _manifest(purpose=None, judge=None)
    m["cost"]["ingest_s"] = manifest.UNKNOWN
    missing, unknown = manifest.check(m)
    assert "purpose" in missing and "judge.model" in missing
    assert unknown == ["cost.ingest_s"]
    manifest.finalize(m, provisional=[])
    assert m["status"] == "incomplete"
    assert manifest.format_missing(m).startswith("INCOMPLETE manifest")


def test_a_dirty_tree_is_provisional_and_an_abort_is_aborted():
    dirty = _manifest(pins={**PINS, "harness_git_sha": "abc123-dirty"}, porcelain=" M x.py")
    manifest.finalize(dirty, provisional=["harness tree was dirty at run time"])
    assert dirty["status"] == "provisional" and dirty["code"]["clean_tree"] is False
    assert manifest.promotable(dirty)
    aborted = manifest.finalize(_manifest(), provisional=[], aborted=True)
    assert aborted["status"] == "aborted"
    assert manifest.promotable(None) == ["no manifest.json in the run directory"]


def test_the_registry_is_append_only_and_only_status_moves(tmp_path):
    m = manifest.finalize(_manifest(), provisional=[])
    manifest.index_append(manifest.index_row(m, "pending"), runs_dir=tmp_path)
    (row,) = manifest.index_rows(tmp_path)
    assert row["run_id"] == "r1" and row["score"] == "pending" and row["status"] == "complete"
    # A second append of the same run_id is an update, not a second row.
    manifest.index_append(manifest.index_row(m, "1/2"), runs_dir=tmp_path)
    (row,) = manifest.index_rows(tmp_path)
    assert row["score"] == "1/2"
    # The score, once set, never moves; the status may.
    manifest.index_update("r1", status="published", score="2/2", runs_dir=tmp_path)
    (row,) = manifest.index_rows(tmp_path)
    assert row["score"] == "1/2" and row["status"] == "published"
    with pytest.raises(ValueError):
        manifest.index_update("r1", status="deleted", runs_dir=tmp_path)
    with pytest.raises(KeyError):
        manifest.index_update("nope", status="aborted", runs_dir=tmp_path)
    other = manifest.finalize(_manifest(run_id="r2"), provisional=[])
    manifest.index_append(manifest.index_row(other, "0/2"), runs_dir=tmp_path)
    assert [r["run_id"] for r in manifest.index_rows(tmp_path)] == ["r1", "r2"]
    text = manifest.index_path(tmp_path).read_text(encoding="utf-8")
    assert text.startswith("# Run registry")


def test_merge_keeps_recorded_values_and_the_first_created_utc():
    first = _manifest(judge=None)
    first["created_utc"] = "2026-01-01T00:00:00Z"
    later = _manifest(purpose=None, served_models=None)
    merged = manifest.merge_into(first, later)
    assert merged["purpose"] == "a test"
    assert merged["reader"]["served_models"] == {"gpt-4o-2024-08-06": 2}
    assert merged["judge"]["model"] == "gpt-4o-2024-08-06"
    assert merged["created_utc"] == "2026-01-01T00:00:00Z"


def _rows(**extra):
    return [
        Prediction(question_id="a", category="c", is_abstention=False, question="q",
                   answer="x", predicted="x", **extra),
        Prediction(question_id="b", category="c", is_abstention=False, question="q",
                   answer="y", predicted="z", **extra),
    ]


def test_prediction_rows_survive_older_and_newer_schemas(tmp_path):
    artifacts.write_predictions(tmp_path, _rows(retrieved_ids=["k1"]))
    back = artifacts.read_predictions(tmp_path, Prediction)
    assert back[0].retrieved_ids == ["k1"] and back[0].verdicts == []
    # An older file without the two fields, and a newer one with a key this reader lacks.
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"question_id": "a", "category": "c", "is_abstention": False,
                               "question": "q", "answer": "x", "predicted": "x",
                               "future_field": 1}) + "\n", encoding="utf-8")
    (row,) = artifacts.read_predictions_file(old, Prediction)
    assert row.retrieved_ids == [] and row.verdicts == []


def test_verdicts_are_appended_and_the_first_never_overwritten(tmp_path):
    artifacts.write_predictions(tmp_path, _rows())
    results = [Result(question_id="a", category="c", is_abstention=False, correct=True,
                      answer="x", predicted="x"),
               Result(question_id="b", category="c", is_abstention=False, correct=False,
                      answer="y", predicted="z")]
    artifacts.annotate_verdicts(tmp_path, results, "j1", "2026-09-22T00:00:00Z")
    results[0].correct = False  # a replay that disagrees
    artifacts.annotate_verdicts(tmp_path, results, "j2", "2026-09-23T00:00:00Z")
    (a, b) = artifacts.read_predictions(tmp_path, Prediction)
    assert [v["correct"] for v in a.verdicts] == [True, False]
    assert [v["judge_hash"] for v in a.verdicts] == ["j1", "j2"]
    assert b.predicted == "z", "every other byte of the row is untouched"
    # stats reads the FIRST verdict.
    assert stats.load_correctness(tmp_path) == {"a": True, "b": False}
    assert artifacts.judge_replay_count(tmp_path) == 0
    artifacts.write_judge_replay(tmp_path, {"results": []})
    artifacts.write_judge_replay(tmp_path, {"results": []})
    assert artifacts.judge_replay_count(tmp_path) == 2
    assert (tmp_path / "judge_replay_2.json").exists()


def test_summary_json_carries_the_score_and_the_intervals(tmp_path):
    from evals.report import summary

    results = [Result(question_id="a", category="c", is_abstention=False, correct=True,
                      answer="x", predicted="x")]
    artifacts.write_summary(tmp_path, "r1", summary(results), results)
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["score"] == "1/1" and payload["overall"]["ci_method"] == "wilson"
    assert payload["by_category"]["c"]["correct"] == 1


def _question():
    from evals.dataset import Question, Session

    turns = [{"role": "user", "content": "My cat is Luna"},
             {"role": "assistant", "content": "Noted."}]
    return Question(
        question_id="q1", question_type="single-session-user", question="What is my cat called?",
        answer="Luna", question_date="2023/05/20 (Sat) 10:00",
        sessions=[Session(session_id="s1", date="2023/05/19 (Fri) 09:00", turns=turns),
                  Session(session_id="s2", date="2023/05/18 (Thu) 09:00",
                          turns=[{"role": "user", "content": "The weather is nice"},
                                 {"role": "assistant", "content": "Indeed."}])],
        answer_session_ids=["s1"],
    )


def test_every_arm_reports_what_it_handed_the_reader():
    q = _question()
    _ctx, ids, seconds = ingest_and_context(NoMemorySystem(), q)
    assert ids == [] and seconds >= 0
    _ctx, ids, _s = ingest_and_context(FullHistorySystem(), q)
    assert ids == ["s1", "s2"], "everything it was fed, by session id"
    for system in (MnimiSystem(embedder=HashingEmbedder()),
                   NaiveRagSystem(embedder=HashingEmbedder())):
        context, ids, _s = ingest_and_context(system, q)
        assert len(ids) == 2 and all("|" in i for i in ids), "round keys"
        assert "Luna" in context


def test_the_pair_record_is_computed_from_rows(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, verdicts in ((a, [True, False, False]), (b, [True, True, False])):
        d.mkdir()
        rows = [Result(question_id=f"q{i}", category="c", is_abstention=False, correct=v,
                       answer="x", predicted="x") for i, v in enumerate(verdicts)]
        artifacts.write_predictions(d, [Prediction(question_id=r.question_id, category="c",
                                                   is_abstention=False, question="q",
                                                   answer="x", predicted="x") for r in rows])
        artifacts.annotate_verdicts(d, rows, "j", "t")
    record = stats.pair_record(a, b, stats.load_correctness(a), stats.load_correctness(b))
    assert record["b_second_wins"] == 1 and record["c_first_wins"] == 0
    assert record["second_wins_ids"] == ["q1"]
    path = stats.save_pair(record, tmp_path / "analyses")
    assert path.name == "a__vs__b.json"


class _FakeOpenAI:
    """Serves the reader and the judge; the judge sees 'yes' for question a only."""

    def __init__(self):
        outer = self
        self.calls = 0

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                text = kwargs["messages"][-1]["content"]
                content = "yes" if "Luna" in text and "Question: What" not in text else "Luna."
                return SimpleNamespace(
                    id="c", model="gpt-4o-2024-08-06-served", system_fingerprint="fp_t",
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                )

        class _Models:
            def retrieve(self, model):
                return SimpleNamespace(id=model)

        self.chat = SimpleNamespace(completions=_Completions())
        self.models = _Models()


@pytest.fixture
def wired(monkeypatch, tmp_path):
    from evals import dataset as dataset_mod
    from evals import judge_cache
    from evals import runner as runner_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MNIMI_API_LEDGER", str(tmp_path / "ledger.jsonl"))
    client = _FakeOpenAI()
    monkeypatch.setattr(evals_main, "build_openai_reader_client", lambda: client)
    dataset_file = tmp_path / "fake_dataset.json"
    dataset_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(dataset_mod, "resolve_path", lambda *a, **k: dataset_file)
    monkeypatch.setattr(runner_mod, "load", lambda *a, **k: [_question()])

    class TmpCache(judge_cache.JudgeCache):
        def __init__(self, path=None, *, judge_fingerprint=""):
            super().__init__(str(tmp_path / "verdicts.json"), judge_fingerprint=judge_fingerprint)

    monkeypatch.setattr(judge_cache, "JudgeCache", TmpCache)
    return client


def _run(run_dir, *extra):
    return evals_main.main(
        ["--system", "no_memory", "--limit", "1", "--reader-transport", "openai",
         "--run-dir", str(run_dir), "--purpose", "test", *extra]
    )


def test_a_run_writes_its_manifest_registry_row_summary_and_verdicts(wired, tmp_path, capsys):
    run_dir = tmp_path / "runs" / "r"
    assert _run(run_dir, "--stage", "all") == 0, capsys.readouterr().err
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["manifest_schema"] == manifest.MANIFEST_SCHEMA
    assert m["purpose"] == "test" and m["claim"] == "none"
    assert m["reader"]["served_models"] == {"gpt-4o-2024-08-06-served": 1}
    assert m["judge"]["model"] == "gpt-4o-2024-08-06" and m["judge"]["replay_count"] == 0
    assert m["dataset"]["n"] == 1 and m["dataset"]["question_ids_sha256"]
    assert m["cost"]["tokens_in"] == 20 and m["cost"]["llm_calls_at_write"] == 0
    assert m["cost"]["ingest_s"] is not None and m["cost"]["wall_clock_s"] is not None
    assert m["missing_fields"] == [], m["missing_fields"]
    assert (run_dir / "summary.json").exists()
    (row,) = manifest.index_rows(run_dir.parent)
    assert row["run_id"] == "r" and row["n"] == "1" and row["score"] in ("0/1", "1/1")
    (pred,) = artifacts.read_predictions(run_dir, Prediction)
    assert len(pred.verdicts) == 1 and pred.retrieved_ids == []
    err = capsys.readouterr().err
    assert "git status --porcelain" in err


def test_an_existing_run_directory_is_never_overwritten(wired, tmp_path, capsys):
    run_dir = tmp_path / "runs" / "r"
    assert _run(run_dir, "--stage", "predict") == 0, capsys.readouterr().err
    before = (run_dir / "predictions.jsonl").read_bytes()
    assert _run(run_dir, "--stage", "predict") == 2
    assert "never overwritten" in capsys.readouterr().err
    assert (run_dir / "predictions.jsonl").read_bytes() == before
    assert _run(run_dir, "--stage", "predict", "--overwrite-run-dir") == 0
    # The predict-only row is pending until the judge lands.
    (row,) = manifest.index_rows(run_dir.parent)
    assert row["score"] == "pending"


def test_a_re_grade_is_a_replay_in_its_own_file(wired, tmp_path, capsys):
    run_dir = tmp_path / "runs" / "r"
    assert _run(run_dir, "--stage", "all") == 0, capsys.readouterr().err
    results_before = (run_dir / "results.json").read_bytes()
    assert _run(run_dir, "--stage", "judge") == 0, capsys.readouterr().err
    assert (run_dir / "results.json").read_bytes() == results_before
    assert (run_dir / "judge_replay_1.json").exists()
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["judge"]["replay_count"] == 1
    (pred,) = artifacts.read_predictions(run_dir, Prediction)
    assert len(pred.verdicts) == 2
    assert len(manifest.index_rows(run_dir.parent)) == 1, "one row per run"


def test_a_claim_needs_its_committed_rule(wired, tmp_path, capsys):
    run_dir = tmp_path / "runs" / "r"
    assert _run(run_dir, "--stage", "predict", "--claim", "C1") == 2
    assert "--rule-commit" in capsys.readouterr().err
    assert not run_dir.exists()


def test_an_aborted_run_is_registered(wired, tmp_path, capsys, monkeypatch):
    run_dir = tmp_path / "runs" / "r"

    def boom(*a, **k):
        raise RuntimeError("reader down")

    monkeypatch.setattr(evals_main, "_predict_openai", boom)
    with pytest.raises(RuntimeError):
        _run(run_dir, "--stage", "predict")
    (row,) = manifest.index_rows(run_dir.parent)
    assert row["status"] == "aborted" and row["run_id"] == "r"


def test_promotion_refuses_incomplete_and_copies_complete_runs(wired, tmp_path, capsys):
    run_dir = tmp_path / "runs" / "r"
    assert _run(run_dir, "--stage", "all") == 0, capsys.readouterr().err
    m = manifest.read_optional(run_dir)
    published = tmp_path / "published"
    if m["code"]["clean_tree"] is not True:
        with pytest.raises(ValueError, match="not clean"):
            publish.promote(run_dir, published)
        m["code"]["clean_tree"] = True
        m["provisional_reasons"] = []
        manifest.write(run_dir, m)
    target = publish.promote(run_dir, published)
    for name in ("pins.json", "predictions.jsonl", "results.json", "summary.json",
                 "manifest.json"):
        assert (target / name).exists(), name
    (row,) = manifest.index_rows(run_dir.parent)
    assert row["status"] == "published"
    with pytest.raises(FileExistsError):
        publish.promote(run_dir, published)
    m["purpose"] = None
    manifest.write(run_dir, m)
    with pytest.raises(ValueError, match="incomplete"):
        publish.promote(run_dir, published, name="again")


def test_backfill_recovers_what_the_artifacts_hold_and_writes_unknown_elsewhere(
    tmp_path, monkeypatch
):
    from evals import backfill

    # The version OF THE RUN comes from `git show <commit>:pyproject.toml`. A shallow CI
    # checkout (actions/checkout, depth 1) and a fork cannot resolve an old commit, so the
    # lookup is answered here instead of by this repository's history: the test is about
    # what backfill does with the answer, not about whether f07c24d is reachable.
    real_git = artifacts._git

    def fake_git(*args):
        if args[:1] == ("show",) and args[1].endswith(":pyproject.toml"):
            return "[project]\nname = \"mnimi\"\nversion = \"1.11.0\"\n"
        return real_git(*args)

    monkeypatch.setattr(artifacts, "_git", fake_git)

    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    results = [Result(question_id="a", category="c", is_abstention=False, correct=True,
                      answer="x", predicted="x", reader_prompt_tokens=5)]
    artifacts.write_predictions(run_dir, _rows()[:1])
    artifacts.write_results(
        run_dir, {**PINS, "harness_git_sha": "f07c24d1739afea4727783906621e526b91fdee2"},
        results, {"elapsed_s": 12.5, "environment": {"gpu_model": "G"}}, [],
        summary={}, judge=JUDGE,
    )
    m = backfill.backfill(
        run_dir, purpose="p", overrides={"cost.predict_wall_s": 100.0, "cost.ingest_s": "UNKNOWN"},
    )
    manifest.finalize(m, [])
    assert m["code"]["mnimi_version"] == "1.11.0", "the version at the run's commit, not today's"
    assert m["cost"]["judge_wall_s"] == 12.5 and m["cost"]["wall_clock_s"] == 112.5
    assert m["cost"]["reader_usd"] == manifest.UNKNOWN, "no reader_resolved.json"
    assert m["reader"]["served_models"] == {manifest.UNKNOWN: 1}
    assert m["environment"]["python"] == manifest.UNKNOWN
    assert m["dataset"]["question_ids_sha256"] == manifest.question_ids_sha256(["a"])
    assert m["missing_fields"] == [] and "cost.ingest_s" in m["unknown_fields"]
    assert m["status"] == "complete"
