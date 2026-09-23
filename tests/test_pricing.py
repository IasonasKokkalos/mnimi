"""The cost gate: prices, projection, ledger, budget, snapshot preflight (task 0.3).

No network. Every fake client answers ``models.retrieve``, the synchronous
completions call (with ``usage``), and the Batch API surface the gate has to
pass through. The ledger is always redirected to ``tmp_path`` via
``MNIMI_API_LEDGER`` so a test can never touch the real one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evals import __main__ as evals_main
from evals import pricing, runner
from evals.dataset import Question, Session
from evals.judge import Judge


# ---------------------------------------------------------------- fakes ---
class _HttpError(Exception):
    """Duck-typed stand-in for openai.APIStatusError: only status_code matters,
    so the test suite needs no openai install (CI runs the dev extra alone)."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class _FakeClient:
    def __init__(self, *, served=("gpt-4o-2024-08-06",), prompt_tokens=120,
                 completion_tokens=40, fingerprint="fp_test"):
        self.served = set(served)
        self.calls: list[dict] = []
        self.model_lookups: list[str] = []
        outer = self

        class _Models:
            def retrieve(self, model):
                outer.model_lookups.append(model)
                if model not in outer.served:
                    raise _HttpError(404, f"model {model} not found")
                return SimpleNamespace(id=model, created=1723000000)

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return SimpleNamespace(
                    id="chatcmpl-x",
                    system_fingerprint=fingerprint,
                    usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                          completion_tokens=completion_tokens),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="yes, 42"))],
                )

        self.models = _Models()
        self.chat = SimpleNamespace(completions=_Completions())


def _questions(n=2):
    return [
        Question(
            question_id=f"q{i}", question_type="single-session-user",
            question=f"question {i}?", answer=f"gold {i}",
            question_date="2023/05/20 (Sat) 10:00",
            sessions=[Session(session_id=f"s{i}", date="2023/05/19 (Fri) 09:00",
                              turns=[{"role": "user", "content": f"fact {i}"},
                                     {"role": "assistant", "content": "Noted."}])],
            answer_session_ids=[f"s{i}"],
        )
        for i in range(1, n + 1)
    ]


def _wire(monkeypatch, tmp_path, client, n=2):
    from evals import dataset as dataset_mod
    from evals import runner as runner_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(pricing.LEDGER_ENV, str(tmp_path / "ledger.jsonl"))

    def forbid(name):
        def _touched(*a, **k):
            raise AssertionError(f"Ollama touched via {name}")

        return _touched

    for name in ("ollama_preflight", "_force_model_load", "_live_ollama_version",
                 "ollama_context_length", "_ollama_get", "_ollama_post"):
        monkeypatch.setattr(evals_main, name, forbid(name))
    monkeypatch.setattr(evals_main, "build_openai_reader_client", lambda: client)
    dataset_file = tmp_path / "fake_dataset.json"
    dataset_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(dataset_mod, "resolve_path", lambda *a, **k: dataset_file)
    monkeypatch.setattr(runner_mod, "load", lambda *a, **k: _questions(n))


def _main(tmp_path, *extra, stage="predict"):
    run_dir = tmp_path / "r"
    argv = ["--system", "no_memory", "--limit", "2", "--stage", stage,
            "--reader-transport", "openai", "--run-dir", str(run_dir), *extra]
    return evals_main.main(argv), run_dir


def _ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------- pricing ---
def test_estimate_uses_the_dated_table_and_the_batch_discount():
    # gpt-4o-2024-08-06: $2.50 in / $10.00 out per million (official page, 2026-09-11).
    assert pricing.estimate_usd("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.50)
    assert pricing.estimate_usd("gpt-4o-2024-08-06", 0, 1_000_000) == pytest.approx(10.00)
    both = pricing.estimate_usd("gpt-4o-2024-08-06", 1_000_000, 1_000_000, batch=True)
    assert both == pytest.approx(6.25)
    assert pricing.PRICES_AS_OF == "2026-09-11"
    assert pricing.API_BUDGET_USD == 83.85  # 33.85 spent + the $50 top-up of 2026-09-22


def test_unpriced_model_cannot_be_estimated():
    with pytest.raises(pricing.UnpricedModelError):
        pricing.estimate_usd("gpt-4o-mini", 10, 10)


def _item(custom_id, content):
    body = {"model": "gpt-4o-2024-08-06", "max_tokens": 800,
            "messages": [{"role": "user", "content": content}]}
    return runner.BatchItem(custom_id=custom_id, body=body, category="c", is_abstention=False,
                            question="q", answer="a", truncated=False, tokens_dropped=0)


def test_projection_is_an_upper_bound_from_the_real_bodies():
    items = [
        _item("a", "x" * 4000),
        _item("b", "y" * 8000),
    ]
    p = pricing.project(model="gpt-4o-2024-08-06", items=items, batch=False,
                        judge_model="gpt-4o-2024-08-06", judge_calls=2)
    assert p.reader_prompt_tokens_est == 3000  # chars / 4, the harness's trim-gate convention
    assert p.reader_completion_tokens_max == 1600  # max_tokens per request, the ceiling
    assert p.reader_usd == pytest.approx(3000 * 2.5e-6 + 1600 * 10e-6)
    assert p.judge_usd == pytest.approx(2 * (600 * 2.5e-6 + 10 * 10e-6))
    assert p.total_usd == pytest.approx(p.reader_usd + p.judge_usd)
    assert "upper bound" in p.basis
    half = pricing.project(model="gpt-4o-2024-08-06", items=items, batch=True,
                           judge_model=None, judge_calls=0)
    assert half.reader_usd == pytest.approx(p.reader_usd / 2) and half.judge_usd == 0


# -------------------------------------------------------------- ledger ---
def test_ledger_appends_and_sums_actual_over_projected(tmp_path, monkeypatch):
    monkeypatch.setenv(pricing.LEDGER_ENV, str(tmp_path / "ledger.jsonl"))
    pricing.append({"ts": "t1", "projected_usd": 1.0, "actual_usd": 0.4})
    pricing.append({"ts": "t2", "projected_usd": 2.0})  # submitted, not yet collected
    assert pricing.spent_usd() == pytest.approx(2.4)
    # The resume line supersedes the submitted one: no double counting.
    pricing.append({"ts": "t3", "projected_usd": 2.0, "actual_usd": 0.9, "supersedes_ts": "t2"})
    assert pricing.spent_usd() == pytest.approx(1.3)
    assert len(pricing.entries()) == 3


def test_ledger_defaults_to_the_gitignored_cache_path(monkeypatch):
    monkeypatch.delenv(pricing.LEDGER_ENV, raising=False)
    assert str(pricing.ledger_path()).replace("\\", "/").endswith(".cache/api_ledger.jsonl")


# ---------------------------------------------------------------- gate ---
def test_projection_is_printed_and_the_run_is_recorded(tmp_path, monkeypatch, capsys):
    client = _FakeClient()
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    err = capsys.readouterr().err
    assert rc == 0, err
    assert "projected" in err and "upper bound" in err and "of $83.85" in err
    assert client.model_lookups == ["gpt-4o-2024-08-06"], "snapshot checked before any call"
    assert len(client.calls) == 2
    (entry,) = _ledger(tmp_path)
    assert entry["reader_model"] == "gpt-4o-2024-08-06" and entry["stage"] == "predict"
    assert entry["reader_prompt_tokens"] == 240 and entry["reader_completion_tokens"] == 80
    assert entry["actual_usd"] == pytest.approx(240 * 2.5e-6 + 80 * 10e-6)
    assert entry["projected_usd"] >= entry["actual_usd"]
    resolved = json.loads((run_dir / "reader_resolved.json").read_text(encoding="utf-8"))
    assert resolved["usage"] == {"prompt_tokens": 240, "completion_tokens": 80}
    assert resolved["actual_usd"] == pytest.approx(entry["actual_usd"])


def test_over_budget_is_refused_before_any_call(tmp_path, monkeypatch, capsys):
    client = _FakeClient()
    _wire(monkeypatch, tmp_path, client)
    pricing.append({"ts": "seed", "projected_usd": 83.84, "actual_usd": 83.84})

    rc, run_dir = _main(tmp_path)

    err = capsys.readouterr().err
    assert rc == 2
    assert "83.84" in err and "83.85" in err and "refus" in err.lower()
    assert client.calls == []
    assert not (run_dir / "predictions.jsonl").exists()
    assert len(_ledger(tmp_path)) == 1, "a refused run is not a spend"


def test_budget_override_is_explicit_and_loud(tmp_path, monkeypatch, capsys):
    client = _FakeClient()
    _wire(monkeypatch, tmp_path, client)
    pricing.append({"ts": "seed", "projected_usd": 49.99, "actual_usd": 49.99})

    rc, _ = _main(tmp_path, "--api-budget-usd", "100")

    err = capsys.readouterr().err
    assert rc == 0, err
    assert "BUDGET OVERRIDE" in err and "100.00" in err
    assert len(client.calls) == 2


def test_missing_snapshot_is_refused_before_any_call(tmp_path, monkeypatch, capsys):
    client = _FakeClient(served=())
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    err = capsys.readouterr().err
    assert rc == 2
    assert "gpt-4o-2024-08-06" in err and "not served" in err
    assert client.calls == [] and not run_dir.exists()
    assert _ledger(tmp_path) == []


def test_unpriced_reader_model_cannot_run(tmp_path, monkeypatch, capsys):
    client = _FakeClient(served=("gpt-4o-mini",))
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path, "--model", "gpt-4o-mini")

    err = capsys.readouterr().err
    assert rc == 2
    assert "gpt-4o-mini" in err and "project" in err
    assert client.calls == [] and not (run_dir / "predictions.jsonl").exists()


def test_allow_large_run_is_gone(tmp_path, monkeypatch):
    client = _FakeClient()
    _wire(monkeypatch, tmp_path, client)
    with pytest.raises(SystemExit):
        _main(tmp_path, "--allow-large-run")


def test_judge_stage_checks_its_snapshot_and_records_its_spend(tmp_path, monkeypatch, capsys):
    client = _FakeClient()
    _wire(monkeypatch, tmp_path, client)
    rc, run_dir = _main(tmp_path)
    assert rc == 0
    # The judge stage on the same run dir: verdicts come from the fake
    # completions client too ("yes, 42" -> correct), through the shared builder.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    judge_client = _FakeClient(prompt_tokens=500, completion_tokens=1)
    monkeypatch.setattr(evals_main, "build_openai_reader_client", lambda: judge_client)
    import evals.judge_cache as judge_cache_mod

    monkeypatch.setattr(judge_cache_mod, "JudgeCache", lambda **k: None)  # no cache in tests

    rc, _ = _main(tmp_path, stage="judge")

    err = capsys.readouterr().err
    assert rc == 0, err
    assert judge_client.model_lookups == ["gpt-4o-2024-08-06"]
    assert len(judge_client.calls) == 2
    entries = _ledger(tmp_path)
    assert len(entries) == 2 and entries[-1]["stage"] == "judge"
    assert entries[-1]["judge_calls"] == 2 and entries[-1]["judge_prompt_tokens"] == 1000
    assert entries[-1]["actual_usd"] == pytest.approx(2 * (500 * 2.5e-6 + 1 * 10e-6))


def test_judge_counts_usage_only_on_real_calls():
    class _Cache:
        def get(self, question_id, response):
            return True if question_id == "cached" else None

        def set(self, *a):
            pass

    client = _FakeClient(prompt_tokens=300, completion_tokens=2)
    judge = Judge("gpt-4o-2024-08-06", client=client, cache=_Cache())
    judge.is_correct(question_id="cached", question="q", answer="a", response="r",
                     question_type="single-session-user", abstention=False)
    judge.is_correct(question_id="live", question="q", answer="a", response="r",
                     question_type="single-session-user", abstention=False)
    assert judge.calls == 1 and judge.prompt_tokens == 300 and judge.completion_tokens == 2


def test_judge_tolerates_a_completion_without_usage():
    class _NoUsage:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])))

    judge = Judge("gpt-4o-2024-08-06", client=_NoUsage())
    assert judge.is_correct(question_id="x", question="q", answer="a", response="r",
                            question_type="single-session-user", abstention=False)
    assert judge.calls == 1 and judge.prompt_tokens == 0
