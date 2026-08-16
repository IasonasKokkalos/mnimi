"""Phase B: reproducibility header, dataset fingerprint, and the stage split.

No network, no Ollama, no OpenAI — the reader and judge are injected fakes, so
the whole staged pipeline is exercised offline.
"""

from __future__ import annotations

import json

import pytest
from evals import __main__ as evals_main
from evals import artifacts, runner
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
        reader_flash_attention=1,
        reader_cache_ram=0,
        reader_prompt_version="mnimi-con-v1",
        reader_prompt_hash="rp",
        render_template_hash="rt",
        # Real values, so the test moves with the trim gate rather than a copy.
        **runner.reader_trim_pins(),
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
    # Decode config determines the output text, so it must move the pins hash.
    assert artifacts.pins_hash(_pins(reader_seed=1)) != baseline
    assert artifacts.pins_hash(_pins(reader_top_k=40)) != baseline


def test_dedup_threshold_moves_the_pins_hash():
    """Found by measurement, not review: two mnimi runs whose only difference
    was this threshold differed by 19/20 predictions and 20 accuracy points,
    and carried the same pins_hash because there was no slot for it."""
    at_095 = _pins(system="mnimi", dedup_cosine_threshold=0.95)
    at_085 = _pins(system="mnimi", dedup_cosine_threshold=0.85)

    assert at_095["dedup_cosine_threshold"] == 0.95
    assert artifacts.pins_hash(at_095) != artifacts.pins_hash(at_085)


def test_retrieval_pins_move_the_pins_hash():
    baseline = artifacts.pins_hash(_pins())
    assert artifacts.pins_hash(_pins(embedder_name="BAAI/bge-small-en-v1.5")) != baseline
    assert artifacts.pins_hash(_pins(embedder_dim=384)) != baseline
    assert artifacts.pins_hash(_pins(embedder_revision="5c38ec7c405e")) != baseline
    assert artifacts.pins_hash(_pins(k=10)) != baseline


def test_schema_declares_revision_for_retrieval_arms_only():
    """A bare model name is mutable and can move every vector without moving
    any header field; the HF commit is the immutable identity."""
    assert _pins()["artifact_schema"] == "mnimi-eval-artifact/4"
    assert _pins()["embedder_revision"] is None, "no_memory retrieves nothing"
    retrieving = _pins(embedder_name="BAAI/bge-small-en-v1.5",
                       embedder_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
    assert retrieving["embedder_revision"].startswith("5c38ec7c")


def test_judge_identity_never_moves_pins_hash(tmp_path):
    """Schema /3: pins describe the predict stage only. Judge identity lives in
    the results.json judge block with its own hash, refreshed at judge time, so
    a judge replay updates grading provenance without touching predict
    provenance."""
    pins = _pins()
    assert not any(key.startswith("judge_") for key in pins), (
        "judge fields inside pins would let a re-grade rewrite predict provenance"
    )

    judge_a = {"judge_model": "gpt-4o-2024-08-06", "judge_prompt_hash": "aaa",
               "judge_temperature": 0, "judge_max_tokens": 10}
    judge_b = {**judge_a, "judge_max_tokens": 500}
    artifacts.write_results(tmp_path, pins, [], {"stage": "judge", "n": 0}, [], judge=judge_a)
    first = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    artifacts.write_results(tmp_path, pins, [], {"stage": "judge", "n": 0}, [], judge=judge_b)
    second = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    assert first["pins_hash"] == second["pins_hash"], "judge change must not move pins_hash"
    assert first["judge_hash"] != second["judge_hash"], "judge change must move judge_hash"
    assert second["judge"]["judge_max_tokens"] == 500


def test_reader_prompt_and_trim_gate_move_the_pins_hash():
    """Phase D swaps the reader prompt; the swap has to be loud in the header."""
    baseline = artifacts.pins_hash(_pins())
    assert artifacts.pins_hash(_pins(reader_prompt_version="plain-prose-v2")) != baseline
    assert artifacts.pins_hash(_pins(reader_prompt_hash="different")) != baseline
    # num_ctx alone does not decide what the reader sees: the trim budget is
    # num_ctx - answer_reserve - scaffold, estimated at chars_per_token.
    assert artifacts.pins_hash(_pins(reader_answer_reserve=1024)) != baseline
    assert artifacts.pins_hash(_pins(reader_scaffold_tokens=512)) != baseline
    assert artifacts.pins_hash(_pins(reader_chars_per_token=5)) != baseline


def test_template_split_hashes_move_the_pins_hash():
    """The embed/render split: two distinct hashes, both loud in the header.
    A render edit changes reader context and no vectors; an embed edit changes
    every vector — the header must be able to tell the two apart."""
    baseline = artifacts.pins_hash(_pins())
    assert artifacts.pins_hash(_pins(render_template_hash="different")) != baseline
    assert artifacts.pins_hash(_pins(embed_template_hash="e1")) != baseline
    # Non-embedding arms leave the embed hash unset; the render hash is
    # harness-wide and always present.
    assert _pins()["embed_template_hash"] is None
    assert _pins()["render_template_hash"] == "rt"


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
    "flash_attention_reported", "prompt_cache_reported", "kv_cache_type",
    "model_blob_path",
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


# VERBATIM from a real daemon. The two prompt-cache states announce themselves
# in different sentences, and the disabled one emits NO `cache state` line at
# all — so a fixture that invents a "limits: 0.000 MiB" line would test the
# wrong model of reality and pass against broken code. It did, once.
CACHE_DISABLED = (
    "srv    load_model: prompt cache is disabled - use `--cache-ram N` to enable it\n"
)
CACHE_ACTIVE = (
    "srv    load_model: use `--cache-ram 0` to disable the prompt cache\n"
    "cache state: 0 prompts, 0.000 MiB (limits: 8192.000 MiB, 32768 tokens, 299467 est)\n"
)


def test_prompt_cache_state_is_captured_as_resolved(tmp_path):
    """A live prompt cache makes a prediction depend on the request before it,
    so the daemon's own printed state is recorded — not the env var that asked."""
    log = tmp_path / "serve.log"

    log.write_text(FAKE_SERVE_LOG + CACHE_ACTIVE, encoding="utf-8")
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert "ACTIVE" in env["prompt_cache_reported"]
    assert "8192.000 MiB" in env["prompt_cache_reported"]

    # The state the harness requires must be confirmable, not merely inferred
    # from the absence of the active line.
    log.write_text(FAKE_SERVE_LOG + CACHE_DISABLED, encoding="utf-8")
    env = artifacts.capture_environment(
        ollama_host="http://127.0.0.1:1", serve_log=str(log)
    )
    assert env["prompt_cache_reported"] == "prompt cache disabled"


class TestPreflightReaderEnv:
    """The pins say what the configuration is; preflight says the daemon agrees.

    Every case here was observed for real during the drift investigation, so
    these are regression tests for a specific wrong number, not hypotheticals.
    """

    GOOD = "llama_context: flash_attn    = enabled\n" + CACHE_DISABLED

    def test_accepts_a_correctly_launched_daemon(self):
        runner.preflight_reader_env(self.GOOD)

    def test_absent_cache_evidence_is_not_a_pass(self):
        """The disabled daemon emits no `cache state` line, so keying off that
        line alone would silently pass a log that proves nothing."""
        with pytest.raises(runner.ReaderEnvError, match="cannot confirm"):
            runner.preflight_reader_env("llama_context: flash_attn    = enabled\n")

    def test_rejects_auto_because_that_means_the_env_var_was_unset(self):
        """`auto` is the tray app's signature: it resolves per host GPU, so the
        run is only reproducible on one machine."""
        with pytest.raises(runner.ReaderEnvError, match="auto"):
            runner.preflight_reader_env("llama_context: flash_attn    = auto\n")

    def test_rejects_the_opposite_flash_attention_value(self):
        with pytest.raises(runner.ReaderEnvError, match="flash_attn=disabled"):
            runner.preflight_reader_env("llama_context: flash_attn    = disabled\n")

    def test_rejects_a_live_prompt_cache(self):
        log = "llama_context: flash_attn    = enabled\n" + CACHE_ACTIVE
        with pytest.raises(runner.ReaderEnvError, match="ACTIVE"):
            runner.preflight_reader_env(log)

    def test_rejects_a_log_with_no_resolution_at_all(self):
        """Absence of evidence is not evidence the daemon is right."""
        with pytest.raises(runner.ReaderEnvError, match="cannot confirm"):
            runner.preflight_reader_env("")


# A correctly launched daemon's load block — OLLAMA_FLASH_ATTENTION resolved to
# `enabled`, prompt cache off. Distinct from FAKE_SERVE_LOG, which carries the
# tray app's unresolved `auto`.
PINNED_LOAD_BLOCK = (
    'time=2026-07-30T09:00:00Z level=INFO source=routes.go:2054 msg="server config" '
    'env="map[OLLAMA_FLASH_ATTENTION:1 OLLAMA_NUM_PARALLEL:1]"\n'
    'time=2026-07-30T09:00:02Z level=INFO msg="starting llama server" '
    f'cmd="ollama runner --model {MODELS_DIR}/blobs/sha256-635e70c8 '
    '--ctx-size 32768 --batch-size 512"\n'
    "load_tensors: loading model tensors, this can take a while...\n"
    "llama_context: flash_attn    = enabled\n"
    "load_tensors: offloaded 29/29 layers to GPU\n"
) + CACHE_DISABLED


# Condensed verbatim from a real 0.32.13 daemon launched with the pinned env
# (OLLAMA_FLASH_ATTENTION=1, LLAMA_ARG_CACHE_RAM=0), 2026-08-16. The spawn line
# is hyphenated and names llama-server.exe directly — no "ollama runner".
OLLAMA_032_13_LOAD_BLOCK = (
    'time=2026-08-16T15:29:06.842+03:00 level=INFO source=llama_server.go:431 '
    'msg="starting llama-server" cmd="C:\\\\...\\\\lib\\\\ollama\\\\llama-server.exe '
    "--model D:\\\\ollama-models\\\\blobs\\\\sha256-a3b7d8df --port 5025 "
    '--flash-attn on -b 512 -ub 512 -ngl 99 -t 8"\n'
    "load_tensors: loading model tensors, this can take a while... (load_mode = none)\n"
    "load_tensors: offloaded 29/29 layers to GPU\n"
    "llama_context: flash_attn            = enabled\n"
    "srv    load_model: prompt cache is disabled - use `--cache-ram N` to enable it\n"
    'time=2026-08-16T15:29:15.885+03:00 level=INFO source=llama_server.go:1360 '
    'msg="llama-server started in 9.04 seconds"\n'
)


def _request_traffic(nbytes: int) -> str:
    """Serve-log noise carrying no load block — what a warm model writes.

    This is the whole mechanism of the bug: the daemon keeps appending request
    lines for the length of a run, and never restates the settings it resolved
    when the runner spawned.
    """
    line = (
        "time=2026-07-30T09:14:22Z level=INFO source=server.go:100 "
        'msg="request" method=POST path=/api/chat status=200\n'
    )
    return line * (nbytes // len(line) + 1)


class TestModelLoadLogSelection:
    """Which bytes of the serve log preflight is shown.

    `preflight_reader_env` is only as good as the text handed to it, and
    choosing that text is a separate job with its own failure mode — one that
    cost three arms of the n=100 sitting.
    """

    def test_finds_the_load_block_however_far_from_the_end_it_sits(
        self, tmp_path, monkeypatch
    ):
        """Regression, measured 2026-07-30 during the n=100 run.

        The resolved `flash_attn` line is emitted once per runner spawn and is
        never restated while the model stays warm, but request logging keeps
        appending. A fixed-size tail therefore stops covering the load block
        once a run is long enough: the log passed ~1.2 MB, the line sat at byte
        16,862, and preflight refused three arms whose daemon was verifiably in
        the pinned configuration.
        """
        log = tmp_path / "serve.log"
        log.write_text(
            PINNED_LOAD_BLOCK + _request_traffic(1_200_000), encoding="utf-8"
        )
        monkeypatch.setenv("OLLAMA_SERVE_LOG", str(log))

        # The two conditions that together produced the failure.
        assert log.stat().st_size > 1_000_000
        assert log.read_text(encoding="utf-8").index("flash_attn") < 2_000

        runner.preflight_reader_env(evals_main._recent_model_load_log())

    def test_a_stale_good_block_does_not_excuse_the_current_bad_one(
        self, tmp_path, monkeypatch
    ):
        """Why the fix is not "read the whole log".

        `preflight_reader_env` takes the FIRST resolution it finds. Handed the
        entire file, it would grade the oldest runner in it — so a daemon
        restarted without OLLAMA_FLASH_ATTENTION would be waved through on the
        strength of a correct block written hours earlier. That turns a false
        refusal into a false pass, which is the one outcome a preflight must
        never return.
        """
        log = tmp_path / "serve.log"
        log.write_text(
            PINNED_LOAD_BLOCK + _request_traffic(50_000) + FAKE_SERVE_LOG,
            encoding="utf-8",
        )
        monkeypatch.setenv("OLLAMA_SERVE_LOG", str(log))

        with pytest.raises(runner.ReaderEnvError, match="auto"):
            runner.preflight_reader_env(evals_main._recent_model_load_log())

    def test_a_log_with_no_runner_start_marker_is_refused(
        self, tmp_path, monkeypatch
    ):
        """Fail closed. Without a runner-start marker there is no way to say
        which daemon wrote the settings below it, and unattributable evidence
        is not evidence."""
        log = tmp_path / "serve.log"
        log.write_text(
            "llama_context: flash_attn    = enabled\n" + CACHE_DISABLED,
            encoding="utf-8",
        )
        monkeypatch.setenv("OLLAMA_SERVE_LOG", str(log))

        with pytest.raises(runner.ReaderEnvError, match="cannot confirm"):
            runner.preflight_reader_env(evals_main._recent_model_load_log())

    def test_a_missing_log_is_refused_rather_than_passed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMA_SERVE_LOG", str(tmp_path / "absent.log"))

        with pytest.raises(runner.ReaderEnvError, match="cannot confirm"):
            runner.preflight_reader_env(evals_main._recent_model_load_log())

    def test_finds_the_load_block_in_the_0_32_13_log_format(self, tmp_path, monkeypatch):
        """Regression, measured 2026-08-16: Ollama 0.32.13 renamed the spawn line.

        0.32.5 wrote ``msg="starting llama server" cmd="ollama runner ..."``;
        0.32.13 writes ``msg="starting llama-server" cmd=".../llama-server.exe ..."``
        (hyphen, direct binary). A marker anchored on the old spelling returns ""
        on the new format and preflight refuses a daemon that is verifiably in
        the pinned configuration — a false refusal, same class as the 400KB tail.
        """
        log = tmp_path / "serve.log"
        log.write_text(OLLAMA_032_13_LOAD_BLOCK, encoding="utf-8")
        monkeypatch.setenv("OLLAMA_SERVE_LOG", str(log))

        runner.preflight_reader_env(evals_main._recent_model_load_log())

    def test_warmup_probe_timeout_covers_a_model_load(self, monkeypatch):
        """Regression, measured 2026-08-16 on Ollama 0.32.13: the daemon ABORTS
        a model load when the requesting client disconnects ("client connection
        closed before llama-server finished loading, aborting load"), and the
        warmup probe's 5s transport timeout disconnected mid-load (measured
        9.04s warm). The probe must ask for a timeout that a load fits inside —
        otherwise the load-bearing warmup can never complete on 0.32.13."""
        seen = {}

        def fake_post(path, payload, timeout=5):
            seen["timeout"] = timeout
            return {}

        monkeypatch.setattr(evals_main, "_ollama_post", fake_post)
        evals_main._force_model_load("m", num_ctx=32768, num_gpu=99)

        assert seen["timeout"] >= 60, (
            f"warmup probe asked for a {seen.get('timeout')}s timeout; a model "
            "load takes ~9s warm and much longer cold"
        )


# The reference CoN template, VERBATIM from src/generation/run_generation.py
# line 55 (github.com/xiaowu0162/LongMemEval) — the byte authority Figure 13
# typesets. One uninterrupted literal, same locking discipline as the judge
# templates in test_judge.py.
REFERENCE_CON_TEMPLATE = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"  # noqa: E501

ABSTENTION_SENTENCE = (
    " If the chat history does not contain enough information to answer the "
    "question, say that you do not know rather than guessing."
)


def test_reader_template_is_the_reference_plus_exactly_two_deviations():
    """mnimi-con-v1 = the reference CoN bytes + the abstention sentence + the
    cache_bust prefix. Anything else appearing in this template is a third,
    un-ruled deviation and must fail here."""
    expected = "${cache_bust}\n\n" + (
        REFERENCE_CON_TEMPLATE
        .replace("to get the answer.", "to get the answer." + ABSTENTION_SENTENCE, 1)
        .replace("{}", "${context}", 1)
        .replace("{}", "${question_date}", 1)
        .replace("{}", "${question}", 1)
    )
    assert runner.READER_TEMPLATE == expected


def test_cache_bust_prefix_makes_the_prompt_unique_per_question():
    """Zero-length shared prefix is the whole point: it forces every prefill to
    start at n_past=0 instead of resuming at the previous question's offset."""
    from string import Template

    a = Template(runner.READER_TEMPLATE).substitute(
        cache_bust="q_alpha", context="c", question_date="d", question="q")
    b = Template(runner.READER_TEMPLATE).substitute(
        cache_bust="q_beta", context="c", question_date="d", question="q")
    assert a != b
    assert a.startswith("q_alpha"), "must lead the message, or the prefix is shared"
    assert b.startswith("q_beta")
    # The hash pins the unsubstituted request shape, not a rendered instance —
    # otherwise every question would look like a different configuration. It
    # covers the roles and the (absent) system slot as recorded values, so
    # restructuring the message stack is as loud as editing the template.
    shape = runner.reader_request_shape()
    assert shape["system_message"] is None
    assert shape["roles"] == ["user"]
    assert "${cache_bust}" in shape["template"], "slot presence is in the hashed text"
    assert runner.reader_prompt_hash() == artifacts.fingerprint(
        artifacts.canonical(shape)
    )


def test_reader_sends_one_user_message_and_consumes_question_date():
    """The reference sends a single user message; question_date reaches the
    prompt through the Current Date line and the context lands in History
    Chats — no harness-invented system message rides along."""
    from evals.runner import Reader

    captured = {}

    class FakeOllama:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "ok"}, "prompt_eval_count": 10}

    Reader("m", num_ctx=32768, client=FakeOllama()).answer(
        "user: I moved to Athens", "where do I live?",
        cache_bust="qid_7", question_date="2023/06/01 (Thu) 09:00",
    )
    assert [m["role"] for m in captured["messages"]] == ["user"]
    prompt = captured["messages"][0]["content"]
    assert prompt.startswith("qid_7")
    assert "History Chats:\n\nuser: I moved to Athens" in prompt
    assert "Current Date: 2023/06/01 (Thu) 09:00" in prompt
    assert prompt.endswith("Answer (step by step):")


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


class TestTier1Audit:
    """A published predictions.jsonl must be gradeable on its own.

    Tier 1's claim is that anyone can recompute the published score from the
    published predictions with no dataset, no reader and no run directory. Each
    test here removes one of those props and asserts the audit still works.
    """

    def test_prediction_rows_carry_question_and_gold_answer(self, tmp_path):
        """Without these two fields inline, Tier 1 needs the dataset and dies."""
        artifacts.write_predictions(tmp_path, _predictions())
        rows = [
            json.loads(line)
            for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            assert row["question"], "question text must travel with the prediction"
            assert row["answer"], "gold answer must travel with the prediction"

    def test_predictions_load_from_an_explicit_path(self, tmp_path):
        """The auditor points at a file, not at a run directory layout."""
        artifacts.write_predictions(tmp_path, _predictions())
        moved = tmp_path / "somewhere_else.jsonl"
        (tmp_path / "predictions.jsonl").rename(moved)
        assert artifacts.read_predictions_file(moved, Prediction) == _predictions()

    def test_pins_fall_back_to_results_then_to_empty(self, tmp_path):
        """A bare predictions file still grades; it just loses provenance."""
        assert artifacts.read_pins_optional(tmp_path) == {}

        artifacts.write_results(tmp_path, _pins(), [], {"stage": "all", "n": 0}, [])
        assert artifacts.read_pins_optional(tmp_path)["dataset_sha256"] == "abc"

        artifacts.write_pins(tmp_path, _pins(dataset_sha256="from-pins-json"))
        assert artifacts.read_pins_optional(tmp_path)["dataset_sha256"] == "from-pins-json"

    def test_published_score_is_read_back_for_comparison(self, tmp_path):
        assert artifacts.read_published_score(tmp_path) is None

        graded = [
            runner.Result(
                question_id="q1",
                category="single-session-user",
                is_abstention=False,
                correct=True,
                answer="a",
                predicted="a",
            ),
            runner.Result(
                question_id="q2",
                category="temporal-reasoning",
                is_abstention=False,
                correct=False,
                answer="18",
                predicted="19",
            ),
        ]
        artifacts.write_results(tmp_path, _pins(), graded, {"stage": "all", "n": 2}, [])
        assert artifacts.read_published_score(tmp_path) == (1, 2)


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
