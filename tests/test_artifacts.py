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
        reader_num_gpu=0,
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
    # Load-time settings pinned too: GPU offload and an unpinned thread count
    # were each measured to make identical inputs produce different answers.
    assert opts["num_gpu"] == 0
    assert opts["num_thread"] == 8
    assert opts["num_batch"] == 512


def test_prompt_hashes_track_the_real_prompt_text():
    # The locked judge templates and the reader prompt must each be pinned.
    assert len(judge_prompt_hash()) == 64
    assert len(reader_prompt_hash()) == 64
    assert judge_prompt_hash() != reader_prompt_hash()


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
