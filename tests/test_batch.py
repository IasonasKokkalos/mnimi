"""Batch API mode for the OpenAI reader transport (task 0.2).

No network. A fake client stands in for ``openai.OpenAI``: it records the
upload and the batch creation, walks a scripted status sequence on
``batches.retrieve``, serves output and error files built from the uploaded
requests (optionally out of order, optionally with failures), and answers the
synchronous fallback calls. Everything the real flow touches is exercised.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from evals import __main__ as evals_main
from evals import artifacts, batch, runner
from evals.dataset import Question, Session


# ---------------------------------------------------------------- fakes ---
class _FakeBatchClient:
    def __init__(self, statuses=("validating", "in_progress", "completed"),
                 fail_ids=(), reverse=False, fingerprint="fp_batch"):
        self.statuses = list(statuses)
        self.fail_ids = set(fail_ids)
        self.reverse = reverse
        self.fingerprint = fingerprint
        self.uploads: list[dict] = []
        self.created: list[dict] = []
        self.retrieves = 0
        self.sync_calls: list[dict] = []
        self._requests: list[dict] = []
        outer = self

        class _Files:
            def create(self, *, file, purpose):
                name, data = file
                outer.uploads.append({"name": name, "purpose": purpose, "bytes": data})
                outer._requests = [json.loads(line) for line in data.decode("utf-8").splitlines()]
                return SimpleNamespace(id=f"file-in-{len(outer.uploads)}")

            def content(self, file_id):
                if file_id == "file-out-1":
                    lines = outer._output_lines()
                elif file_id == "file-err-1":
                    lines = outer._error_lines()
                else:
                    raise AssertionError(f"unexpected file id {file_id}")
                payload = "\n".join(json.dumps(x) for x in lines)
                return SimpleNamespace(content=payload.encode("utf-8"))

        class _Batches:
            def create(self, **kwargs):
                outer.created.append(kwargs)
                return outer._batch("validating")

            def retrieve(self, batch_id):
                outer.retrieves += 1
                idx = min(outer.retrieves - 1, len(outer.statuses) - 1)
                return outer._batch(outer.statuses[idx], batch_id=batch_id)

        class _Completions:
            def create(self, **kwargs):
                outer.sync_calls.append(kwargs)
                return SimpleNamespace(
                    id="chatcmpl-sync",
                    system_fingerprint="fp_sync",
                    usage=SimpleNamespace(prompt_tokens=77, completion_tokens=3),
                    choices=[SimpleNamespace(message=SimpleNamespace(content="sync answer"))],
                )

        self.files = _Files()
        self.batches = _Batches()
        self.chat = SimpleNamespace(completions=_Completions())

    def _batch(self, status, batch_id="batch_1"):
        total = len(self._requests)
        ok = total - len(self.fail_ids)
        terminal = status in batch.TERMINAL
        return SimpleNamespace(
            id=batch_id,
            status=status,
            request_counts=SimpleNamespace(
                completed=ok if terminal else 0,
                failed=len(self.fail_ids) if terminal else 0,
                total=total,
            ),
            output_file_id="file-out-1" if status in batch.WITH_OUTPUT else None,
            error_file_id="file-err-1" if (terminal and self.fail_ids) else None,
            errors=SimpleNamespace(
                data=[SimpleNamespace(code="invalid_request", message="bad line", line=1)]
            ) if status == "failed" else None,
        )

    def _output_lines(self):
        lines = []
        for i, req in enumerate(self._requests):
            cid = req["custom_id"]
            if cid in self.fail_ids:
                continue
            lines.append({
                "id": f"batch_req_{i}", "custom_id": cid,
                "response": {"status_code": 200, "request_id": f"req_{i}", "body": {
                    "id": f"chatcmpl-{i}",
                    "system_fingerprint": self.fingerprint,
                    "usage": {"prompt_tokens": 100 + i, "completion_tokens": 5},
                    "choices": [
                        {"message": {"role": "assistant", "content": f"  answer for {cid}  "}}
                    ],
                }},
                "error": None,
            })
        return list(reversed(lines)) if self.reverse else lines

    def _error_lines(self):
        return [
            {"id": "batch_req_err", "custom_id": cid, "response": None,
             "error": {"code": "server_error", "message": "boom"}}
            for cid in sorted(self.fail_ids)
        ]


def _questions():
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
        for i in (1, 2)
    ]


def _wire(monkeypatch, tmp_path, client):
    from evals import dataset as dataset_mod
    from evals import runner as runner_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

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
    monkeypatch.setattr(runner_mod, "load", lambda *a, **k: _questions())


def _rows(run_dir):
    text = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines()]


def _main(tmp_path, *extra):
    run_dir = tmp_path / "r"
    argv = ["--system", "no_memory", "--limit", "2", "--stage", "predict",
            "--reader-transport", "openai", "--batch", "--batch-poll-seconds", "0",
            "--run-dir", str(run_dir), *extra]
    return evals_main.main(argv), run_dir


# ------------------------------------------------------ request building ---
def test_request_body_is_the_sync_request():
    reader = runner.OpenAIReader("gpt-4o-2024-08-06", num_ctx=128_000, client=object())
    body, truncated, dropped = reader.request_body("ctx", "q?", cache_bust="q1", question_date="d")
    assert body["model"] == "gpt-4o-2024-08-06"
    assert body["temperature"] == 0 and body["seed"] == runner.READER_SEED
    assert body["max_tokens"] == runner.READER_ANSWER_RESERVE
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"].startswith("q1\n\n")
    assert truncated is False and dropped == 0


def test_request_line_shape():
    item = runner.BatchItem(custom_id="q1", body={"model": "m"}, category="c",
                            is_abstention=False, question="q", answer="a",
                            truncated=False, tokens_dropped=0)
    assert batch.request_line(item) == {
        "custom_id": "q1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m"}
    }


def test_parse_completion_accepts_dict_and_object():
    body = {"choices": [{"message": {"content": " hi "}}], "usage": {"prompt_tokens": 9},
            "system_fingerprint": "fp_d"}
    assert runner.parse_completion(body) == ("hi", 9, "fp_d")
    obj = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" hi "))],
                          usage=SimpleNamespace(prompt_tokens=9), system_fingerprint="fp_o")
    assert runner.parse_completion(obj) == ("hi", 9, "fp_o")


# ------------------------------------------------------------ batch.py ---
def test_wait_walks_to_a_terminal_status_and_reports_changes():
    client = _FakeBatchClient(statuses=("validating", "in_progress", "in_progress", "completed"))
    client._requests = [{"custom_id": "q1"}]
    seen = []
    final = batch.wait(client, "batch_1", poll_seconds=0, progress=lambda b: seen.append(b.status))
    assert final.status == "completed"
    assert seen == ["validating", "in_progress", "completed"], "one line per status change"


def test_fetch_outputs_keys_by_custom_id_and_separates_errors():
    client = _FakeBatchClient(fail_ids={"q2"}, reverse=True)
    client._requests = [{"custom_id": "q1"}, {"custom_id": "q2"}, {"custom_id": "q3"}]
    outputs, errors = batch.fetch_outputs(client, client._batch("completed"))
    assert set(outputs) == {"q1", "q3"} and set(errors) == {"q2"}
    assert outputs["q3"]["usage"]["prompt_tokens"] == 102


# ------------------------------------------------------------ end to end ---
def test_batch_run_writes_the_same_artifacts_in_question_order(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient(reverse=True)
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    assert rc == 0, capsys.readouterr().err
    # Upload and creation, exactly as the guide specifies.
    (upload,) = client.uploads
    assert upload["purpose"] == "batch"
    (created,) = client.created
    assert created["endpoint"] == "/v1/chat/completions"
    assert created["completion_window"] == "24h"
    assert created["input_file_id"] == "file-in-1"
    # The uploaded lines are the run's own request file, byte for byte.
    assert (run_dir / "batch_requests.jsonl").read_bytes() == upload["bytes"]
    lines = [json.loads(x) for x in upload["bytes"].decode("utf-8").splitlines()]
    assert [x["custom_id"] for x in lines] == ["q1", "q2"]
    assert lines[0]["body"]["temperature"] == 0 and lines[0]["body"]["max_tokens"] == 800
    # Artifacts.
    pins = json.loads((run_dir / "pins.json").read_text(encoding="utf-8"))["pins"]
    assert pins["reader_transport"] == "openai"
    rows = _rows(run_dir)
    assert [r["question_id"] for r in rows] == ["q1", "q2"], "question order, not output order"
    assert rows[0]["predicted"] == "answer for q1" and rows[1]["reader_prompt_tokens"] == 101
    assert rows[0]["answer"] == "gold 1" and rows[0]["question"] == "question 1?"
    resolved = json.loads((run_dir / "reader_resolved.json").read_text(encoding="utf-8"))
    assert resolved["batch_id"] == "batch_1" and resolved["batch_status"] == "completed"
    assert resolved["request_counts"] == {"completed": 2, "failed": 0, "total": 2}
    assert resolved["system_fingerprints"] == {"fp_batch": 2}
    assert resolved["sync_fallbacks"] == []
    state = json.loads((run_dir / "batch_state.json").read_text(encoding="utf-8"))
    assert state["batch_id"] == "batch_1" and state["status"] == "done"
    assert client.sync_calls == []


def test_no_wait_submits_and_prints_the_resume_command(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient()
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path, "--batch-no-wait")

    assert rc == 0
    assert len(client.created) == 1 and client.retrieves == 0
    state = json.loads((run_dir / "batch_state.json").read_text(encoding="utf-8"))
    assert state["batch_id"] == "batch_1" and state["status"] == "validating"
    assert (run_dir / "pins.json").exists() and (run_dir / "batch_requests.jsonl").exists()
    assert not (run_dir / "predictions.jsonl").exists()
    err = capsys.readouterr().err
    assert "--batch" in err and str(run_dir) in err and "resume" in err.lower()


def test_resume_never_resubmits(tmp_path, monkeypatch, capsys):
    first = _FakeBatchClient()
    _wire(monkeypatch, tmp_path, first)
    rc, run_dir = _main(tmp_path, "--batch-no-wait")
    assert rc == 0

    second = _FakeBatchClient(statuses=("in_progress", "completed"))
    second._requests = first._requests  # the provider still holds the file
    monkeypatch.setattr(evals_main, "build_openai_reader_client", lambda: second)

    rc, run_dir = _main(tmp_path)

    assert rc == 0, capsys.readouterr().err
    assert second.uploads == [] and second.created == []
    rows = _rows(run_dir)
    assert [r["question_id"] for r in rows] == ["q1", "q2"]


def test_resume_with_different_pins_is_refused(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient()
    _wire(monkeypatch, tmp_path, client)
    rc, run_dir = _main(tmp_path, "--batch-no-wait")
    assert rc == 0
    state_path = run_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["pins_hash"] = "not-the-same"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    rc, run_dir = _main(tmp_path)

    assert rc == 2
    assert "pins" in capsys.readouterr().err
    assert not (run_dir / "predictions.jsonl").exists()


def test_failed_request_falls_back_to_one_sync_call(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient(fail_ids={"q2"})
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    assert rc == 0, capsys.readouterr().err
    (call,) = client.sync_calls
    assert call["messages"][0]["content"].startswith("q2\n\n"), "the failed item's own body"
    rows = _rows(run_dir)
    assert [r["predicted"] for r in rows] == ["answer for q1", "sync answer"]
    resolved = json.loads((run_dir / "reader_resolved.json").read_text(encoding="utf-8"))
    assert resolved["sync_fallbacks"] == ["q2"]
    assert resolved["system_fingerprints"] == {"fp_batch": 1, "fp_sync": 1}


def test_expired_batch_with_partial_output_uses_the_fallback(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient(statuses=("in_progress", "expired"), fail_ids={"q1"})
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    assert rc == 0, capsys.readouterr().err
    assert len(client.sync_calls) == 1
    resolved = json.loads((run_dir / "reader_resolved.json").read_text(encoding="utf-8"))
    assert resolved["batch_status"] == "expired" and resolved["sync_fallbacks"] == ["q1"]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_keeps_state_and_writes_no_predictions(
    tmp_path, monkeypatch, capsys, status
):
    client = _FakeBatchClient(statuses=("validating", status))
    _wire(monkeypatch, tmp_path, client)

    rc, run_dir = _main(tmp_path)

    assert rc == 2
    err = capsys.readouterr().err
    assert status in err
    if status == "failed":
        assert "invalid_request" in err
    assert not (run_dir / "predictions.jsonl").exists()
    state = json.loads((run_dir / "batch_state.json").read_text(encoding="utf-8"))
    assert state["status"] == status and state["batch_id"] == "batch_1"


def test_batch_on_the_ollama_transport_is_refused(tmp_path, monkeypatch, capsys):
    client = _FakeBatchClient()
    _wire(monkeypatch, tmp_path, client)

    rc = evals_main.main(["--system", "no_memory", "--limit", "2", "--stage", "predict",
                          "--batch", "--run-dir", str(tmp_path / "r")])

    assert rc == 2
    assert "--batch" in capsys.readouterr().err
    assert client.uploads == [] and not (tmp_path / "r").exists()


def test_batch_state_roundtrip(tmp_path):
    artifacts.write_batch_state(tmp_path, {"batch_id": "b", "status": "validating", "items": []})
    assert artifacts.read_batch_state_optional(tmp_path)["batch_id"] == "b"
    assert artifacts.read_batch_state_optional(tmp_path / "absent") is None
