"""Phase B: reproducibility header, dataset fingerprint, and the stage split.

No network, no Ollama, no OpenAI — the reader and judge are injected fakes, so
the whole staged pipeline is exercised offline.
"""

from __future__ import annotations

import json

from evals import artifacts
from evals.dataset import file_sha256
from evals.judge import judge_prompt_hash
from evals.judge_cache import JudgeCache
from evals.runner import Prediction, judge_predictions, reader_prompt_hash


def _pins(**overrides) -> dict:
    base = dict(
        dataset_file="x.json",
        dataset_sha256="abc",
        system="no_memory",
        limit=20,
        reader_model="qwen2.5:1.5b-instruct-q4_0",
        reader_digest="sha256:deadbeef",
        reader_num_ctx=32768,
        reader_seed=0,
        reader_top_k=1,
        sample_strategy="stratified-round-robin",
        sample_seed=0,
        reader_num_gpu=99,
        reader_num_thread=8,
        reader_num_batch=512,
        reader_prompt_version="plain-prose-v1",
        reader_prompt_hash="rp",
        judge_model="gpt-4o-2024-08-06",
        judge_prompt_version="longmemeval-paper-v1",
        judge_prompt_hash="jp",
    )
    base.update(overrides)
    return artifacts.build_pins(**base)


def test_fingerprint_is_stable_across_processes():
    # Not Python's salted hash(): a persisted cache key must survive a restart.
    assert artifacts.fingerprint("hello") == artifacts.fingerprint("hello")
    assert artifacts.fingerprint("hello") != artifacts.fingerprint("hello ")


def test_pins_hash_changes_when_any_pin_changes():
    baseline = artifacts.pins_hash(_pins())
    assert artifacts.pins_hash(_pins()) == baseline
    assert artifacts.pins_hash(_pins(dataset_sha256="different")) != baseline
    assert artifacts.pins_hash(_pins(reader_num_ctx=4096)) != baseline
    assert artifacts.pins_hash(_pins(judge_model="gpt-4o")) != baseline
    # Decode config determines the output text, so it must move the pins hash.
    assert artifacts.pins_hash(_pins(reader_seed=1)) != baseline
    assert artifacts.pins_hash(_pins(reader_top_k=40)) != baseline


def test_reader_sends_pinned_decode_options():
    """temperature=0 alone is not greedy; top_k and seed must reach Ollama."""
    from evals.runner import Reader

    captured = {}

    class FakeOllama:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "ok"}, "prompt_eval_count": 10}

    Reader("m", num_ctx=32768, client=FakeOllama()).answer("ctx", "q?")

    opts = captured["options"]
    assert opts["temperature"] == 0
    assert opts["top_k"] == 1
    assert opts["seed"] == 0
    assert opts["num_ctx"] == 32768
    # num_batch is the load-bearing pin: leaving it to Ollama produced the drift
    # once misattributed to CUDA atomics. It must be sent on every call.
    assert opts["num_batch"] == 512
    assert opts["num_gpu"] == 99
    assert opts["num_thread"] == 8


def test_prompt_hashes_track_the_real_prompt_text():
    # The locked judge templates and the reader prompt must each be pinned.
    assert len(judge_prompt_hash()) == 64
    assert len(reader_prompt_hash()) == 64
    assert judge_prompt_hash() != reader_prompt_hash()


ENV_FIELDS = (
    "gpu_model", "driver_version", "cuda_version", "ollama_version",
    "offloaded_layers", "offloaded_layers_source",
    "ollama_env_client", "ollama_env_daemon", "ollama_env_mismatch",
    "flash_attention_reported", "kv_cache_type", "model_blob_path",
    "runner_cmd", "serve_log_path", "model_load_log",
)

# A realistic slice of an Ollama server log, including the daemon's own config
# line, the resolved flash-attention state, and the runner command. Forward
# slashes keep the fixture free of backslash-escaping ambiguity.
MODELS_DIR = "D:/ollama-models"
FAKE_SERVE_LOG = (
    'time=2026-07-26T10:11:58Z level=INFO source=routes.go:2054 msg="server config" '
    'env="map[HTTP_PROXY: OLLAMA_DEBUG:false OLLAMA_FLASH_ATTENTION:true '
    f'OLLAMA_KV_CACHE_TYPE:q8_0 OLLAMA_MODELS:{MODELS_DIR} OLLAMA_NUM_PARALLEL:1]"\n'
    'time=2026-07-26T10:12:00Z level=INFO msg="starting llama server" '
    f'cmd="ollama runner --model {MODELS_DIR}/blobs/sha256-635e70c8 '
    '--ctx-size 32768 --batch-size 512"\n'
    "load_tensors: loading model tensors, this can take a while...\n"
    "llama_context: flash_attn    = auto\n"
    "llama_context: Flash Attention enabled\n"
    "llama_kv_cache: type_k = f16, type_v = f16\n"
    "load_tensors: offloaded 29/29 layers to GPU\n"
)


def test_environment_capture_never_enters_pins_hash(tmp_path):
    """CHANGE 4: env fields are diagnostic. If they hashed, every machine would
    look like a different configuration and no two runs could be compared."""
    log = tmp_path / "serve.log"
    log.write_text(FAKE_SERVE_LOG, encoding="utf-8")
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1",  # unreachable on purpose
        serve_log=str(log),
        save_log_to=tmp_path / "model_load.log",
    )
    # Shape is stable even with nothing reachable — absence is recorded, not faked.
    for key in ENV_FIELDS:
        assert key in env, f"missing diagnostic field {key}"

    pins = _pins()
    baseline = artifacts.pins_hash(pins)
    # Every env key — including the ones added for the drift investigation —
    # must be absent from pins, and a populated capture must not move the hash.
    assert not (set(env) & set(pins)), "environment leaked into pins"
    assert artifacts.pins_hash(_pins()) == baseline
    # Populated, not merely present: this is the case that would catch a leak.
    assert env["ollama_env_daemon"], "daemon env should parse from the log"
    assert artifacts.pins_hash({**pins}) == baseline


def test_environment_capture_reads_resolved_load_state(tmp_path):
    """Resolved values, not requested ones, and the daemon's own OLLAMA_* view."""
    log = tmp_path / "serve.log"
    log.write_text(FAKE_SERVE_LOG, encoding="utf-8")
    saved = tmp_path / "model_load.log"
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log), save_log_to=saved
    )

    # flash_attn was "auto"; the resolution is what gets recorded.
    assert env["flash_attention_reported"] == "Flash Attention enabled"
    assert "f16" in env["kv_cache_type"]
    assert "load log" in env["kv_cache_type"], "source of the value must be stated"
    assert "sha256-635e70c8" in env["model_blob_path"]
    assert "--ctx-size 32768" in env["runner_cmd"]
    assert env["offloaded_layers"] == "29/29"
    assert env["offloaded_layers_source"] == "server_log"

    # Daemon-side OLLAMA_* parsed from the server's own config line.
    daemon = env["ollama_env_daemon"]
    assert daemon["OLLAMA_FLASH_ATTENTION"] == "true"
    assert daemon["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert "ollama-models" in daemon["OLLAMA_MODELS"]
    assert all(k.startswith("OLLAMA_") for k in daemon)

    # Verbatim load log is saved beside the run, with a pointer recorded.
    assert saved.exists() and "offloaded 29/29" in saved.read_text(encoding="utf-8")
    assert env["model_load_log"] == "see model_load.log"


def test_client_and_daemon_env_mismatch_is_surfaced(tmp_path, monkeypatch):
    """The C:-vs-D: OLLAMA_MODELS split cost a run; a disagreement must be loud."""
    log = tmp_path / "serve.log"
    log.write_text(FAKE_SERVE_LOG, encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODELS", "C:/somewhere/else")
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert env["ollama_env_client"]["OLLAMA_MODELS"] == "C:/somewhere/else"
    assert "OLLAMA_MODELS" in env["ollama_env_mismatch"]

    # Agreement reports "none", not an empty value that reads as missing data.
    monkeypatch.setenv("OLLAMA_MODELS", MODELS_DIR)
    env2 = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert env2["ollama_env_mismatch"] == "none"


def test_windows_path_escaping_is_not_a_mismatch(tmp_path, monkeypatch):
    """Regression: the daemon logs backslashes doubled.

    Comparing raw strings reported a mismatch on every Windows path, which
    would make the mismatch field noise in exactly the case it exists for.
    """
    log = tmp_path / "serve.log"
    log.write_text(
        'msg="server config" env="map[OLLAMA_MODELS:D:\\\\ollama-models '
        'OLLAMA_DEBUG:false]"\n',
        encoding="utf-8",
    )
    # Client holds the same path with single separators, as Windows reports it.
    monkeypatch.setenv("OLLAMA_MODELS", "D:\\ollama-models")
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert env["ollama_env_daemon"]["OLLAMA_MODELS"] == "D:\\ollama-models", (
        "daemon value should be unescaped when recorded"
    )
    assert env["ollama_env_mismatch"] == "none", "same path must not read as a mismatch"

    # A genuinely different drive still registers.
    monkeypatch.setenv("OLLAMA_MODELS", "C:\\Users\\me\\.ollama\\models")
    env2 = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert "OLLAMA_MODELS" in env2["ollama_env_mismatch"]


def test_sampling_pins_are_part_of_the_hash():
    """A file-order slice and a stratified slice are different benchmarks."""
    baseline = artifacts.pins_hash(_pins())
    assert artifacts.pins_hash(_pins(sample_strategy="file-order")) != baseline
    assert artifacts.pins_hash(_pins(sample_seed=7)) != baseline


def test_stratified_sampling_spreads_across_categories():
    """CHANGE 5: file order is category-clustered; a naive slice is one category."""
    from evals.dataset import Question, sample_stratified

    def q(i, cat):
        return Question(question_id=f"q{i}", question_type=cat, question="?",
                        answer="a", question_date="", sessions=[], answer_session_ids=[])

    # Clustered exactly like the real file: all of one category, then the next.
    questions = ([q(i, "single-session-user") for i in range(50)]
                 + [q(i + 50, "temporal-reasoning") for i in range(50)]
                 + [q(i + 100, "knowledge-update") for i in range(50)]
                 + [q(i + 150, "multi-session") for i in range(50)])

    picked = sample_stratified(questions, limit=20, seed=0)
    assert len(picked) == 20
    cats = {p.category for p in picked}
    assert len(cats) == 4, f"expected all 4 categories, got {cats}"
    # Same seed -> same slice; different seed -> different slice.
    assert [p.question_id for p in sample_stratified(questions, 20, 0)] == \
           [p.question_id for p in picked]
    assert [p.question_id for p in sample_stratified(questions, 20, 1)] != \
           [p.question_id for p in picked]


def test_truncation_caveat_labels_a_truncated_row():
    """CHANGE 6: a truncated full_history row is not a ceiling."""
    from evals.report import format_table, truncation_caveat
    from evals.runner import Result

    def r(truncated, dropped, fed):
        return Result(question_id="q", category="multi-session", is_abstention=False,
                      correct=True, answer="a", predicted="p",
                      reader_prompt_tokens=fed, truncated=truncated,
                      tokens_dropped=dropped)

    assert truncation_caveat([r(False, 0, 100)]) is None
    caveat = truncation_caveat([r(True, 90000, 27000)])
    assert "NOT full history" in caveat
    table = format_table("full_history", [r(True, 90000, 27000)])
    assert "*" in table and "NOT full history" in table
    assert "abstention questions in slice:" in table


def test_dataset_sha256_detects_a_changed_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"question_id": "a"}]', encoding="utf-8")
    before = file_sha256(path)
    assert before == file_sha256(path)
    path.write_text('[{"question_id": "b"}]', encoding="utf-8")
    assert file_sha256(path) != before


def _predictions() -> list[Prediction]:
    return [
        Prediction(
            question_id="q1",
            category="single-session-user",
            is_abstention=False,
            question="where do I live?",
            answer="Athens",
            predicted="You live in Athens.",
            reader_prompt_tokens=1234,
            truncated=False,
            tokens_dropped=0,
        ),
        Prediction(
            question_id="q2",
            category="temporal-reasoning",
            is_abstention=False,
            question="how many days?",
            answer="18",
            predicted="19 days.",
            reader_prompt_tokens=31000,
            truncated=True,
            tokens_dropped=84000,
        ),
    ]


def test_predictions_roundtrip_through_disk(tmp_path):
    artifacts.write_predictions(tmp_path, _predictions())
    restored = artifacts.read_predictions(tmp_path, Prediction)
    assert restored == _predictions()


def test_judge_stage_runs_from_disk_without_dataset_or_reader(tmp_path):
    """The point of the split: grading needs only predictions.jsonl."""
    artifacts.write_predictions(tmp_path, _predictions())
    restored = artifacts.read_predictions(tmp_path, Prediction)

    class FakeJudgeClient:
        """Minimal OpenAI-shaped double; says yes to everything."""

        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": "yes"})()})()]},
            )()

    client = FakeJudgeClient()
    results = judge_predictions(restored, judge_model="gpt-4o-2024-08-06", judge_client=client)

    assert [r.correct for r in results] == [True, True]
    assert client.calls == 2
    # Reader-side accounting survives the disk roundtrip into the graded rows.
    assert results[1].truncated is True
    assert results[1].tokens_dropped == 84000


def test_verdict_cache_avoids_the_second_call(tmp_path):
    cache = JudgeCache(str(tmp_path / "verdicts.json"))

    class CountingClient:
        def __init__(self):
            self.calls = 0
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, **kwargs):
            self.calls += 1
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": "yes"})()})()]},
            )()

    preds = _predictions()
    client = CountingClient()
    judge_predictions(preds, judge_model="m", judge_cache=cache, judge_client=client)
    assert client.calls == 2

    # A second grading pass over identical predictions must cost nothing.
    warm = JudgeCache(str(tmp_path / "verdicts.json"))
    judge_predictions(preds, judge_model="m", judge_cache=warm, judge_client=client)
    assert client.calls == 2
    assert warm.hits == 2


def test_results_artifact_carries_header_and_provisionality(tmp_path):
    pins = _pins()
    artifacts.write_results(
        tmp_path,
        pins,
        [],
        {"stage": "all", "n": 0},
        ["reader prompt is plain-prose-v1"],
    )
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["pins"]["dataset_sha256"] == "abc"
    assert payload["pins_hash"] == artifacts.pins_hash(pins)
    assert payload["provisional"] == ["reader prompt is plain-prose-v1"]
