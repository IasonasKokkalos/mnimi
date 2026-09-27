"""The competitor contract (PHASE8 D9): the one renderer, the system's own ids, no leak.

No third-party package is imported here: the adapters are exercised through fake
modules injected into ``sys.modules``, so the test path stays inside the CI rule.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from evals import artifacts
from evals.systems import competitor

TS_A = "2023/05/20 (Sat) 09:00"
TS_B = "2023/05/21 (Sun) 10:00"


def test_import_evals_systems_loads_no_competitor_package():
    import evals.systems  # noqa: F401

    for name in ("agentmemory", "mem0", "sentence_transformers", "qdrant_client", "omega"):
        assert name not in sys.modules


def test_render_hits_orders_by_session_date_and_uses_the_one_renderer():
    from mnimi.memory import render_turns

    hits = [competitor.Hit("b", "second", TS_B), competitor.Hit("a", "first", TS_A)]
    assert competitor.render_hits(hits, "text") == render_turns(
        [{"role": "memory", "content": "first", "ts": TS_A},
         {"role": "memory", "content": "second", "ts": TS_B}], fmt="text")


def test_render_hits_keeps_recall_order_within_one_session():
    hits = [competitor.Hit("z", "ranked first", TS_A), competitor.Hit("a", "ranked second", TS_A)]
    rendered = competitor.render_hits(hits, "text")
    assert rendered.index("ranked first") < rendered.index("ranked second"), "never re-sorted by id"


def test_epoch_parses_the_dataset_format_as_utc_and_refuses_nothing_else():
    assert competitor.epoch(TS_A) == 1684573200.0
    assert competitor.epoch("not a date") is None
    assert competitor.epoch(None) is None


class _FakeNode:
    def __init__(self, id_, content, session_id):
        self.id, self.content = id_, content
        self.provenance = types.SimpleNamespace(session_id=session_id)


class _FakeStore:
    instances: list = []

    def __init__(self, path=":memory:", **kwargs):
        self.path, self.kwargs, self.sessions, self.closed = path, kwargs, [], False
        _FakeStore.instances.append(self)

    def ingest_conversation(self, messages, session_id="", reference_date=None, **_):
        self.sessions.append((session_id, reference_date, messages))

    def recall(self, query, limit=10, **_):
        out = []
        for sid, _ref, msgs in self.sessions:
            for i, m in enumerate(msgs):
                out.append(types.SimpleNamespace(node=_FakeNode(f"{sid}#{i}", m["content"], sid),
                                                 score=1.0))
        return out[:limit]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_agentmemory(monkeypatch):
    mod = types.ModuleType("agentmemory")
    mod.MemoryStore = _FakeStore
    mod.__version__ = "4.0.0-fake"
    monkeypatch.setitem(sys.modules, "agentmemory", mod)
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    _FakeStore.instances.clear()
    return mod


def _config(**overrides):
    block = {"package": "agentmemory @ fake", "store": {"auto_graph": True},
             "pythonhashseed": 42,
             "context": {"budget_tokens": 1_000_000, "candidates": 50, "tokenizer": "chars/4"}}
    block.update(overrides)
    return block


def test_agentmemory_adapter_renders_through_the_one_renderer_and_returns_its_ids(
    fake_agentmemory,
):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    system = AgentMemorySystem(_config())
    system.reset()
    system.add([{"role": "user", "content": "I live in Boston.", "ts": TS_A},
                {"role": "assistant", "content": "Noted.", "ts": TS_A}])
    system.add([{"role": "user", "content": "I adopted a cat.", "ts": TS_B},
                {"role": "assistant", "content": "Lovely.", "ts": TS_B}])
    ctx = system.get_context("where do I live")
    assert "memory: I live in Boston." in ctx and f"[Session date: {TS_A}]" in ctx
    assert f"[Session date: {TS_B}]" in ctx
    assert system.retrieved_ids() == ["s000#0", "s000#1", "s001#0", "s001#1"]
    store = _FakeStore.instances[-1]
    (sid_a, ref_a, msgs_a), (sid_b, ref_b, _msgs_b) = store.sessions
    assert (sid_a, sid_b) == ("s000", "s001"), "one unique id per session"
    assert ref_a == 1684573200.0 and ref_b == 1684573200.0 + 86400 + 3600, "the session's own date"
    assert all(set(m) == {"role", "content", "timestamp"} for m in msgs_a), "no harness invention"
    assert all(m["timestamp"] == 1684573200.0 for m in msgs_a)
    assert store.kwargs == {"auto_graph": True, "embedder": None}


def test_agentmemory_reset_closes_the_store_and_starts_a_fresh_one(fake_agentmemory):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    system = AgentMemorySystem(_config())
    system.reset()
    first = _FakeStore.instances[-1]
    system.add([{"role": "user", "content": "x", "ts": TS_A}])
    system.reset()
    assert first.closed and _FakeStore.instances[-1] is not first
    assert _FakeStore.instances[-1].sessions == []


def test_agentmemory_pins_name_the_system_and_its_configuration(fake_agentmemory):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    pins = AgentMemorySystem(_config()).retrieval_pins()
    assert pins["competitor_name"] == "agentmemory"
    assert pins["competitor_version"] == "agentmemory @ fake"
    assert pins["competitor_llm"] == "none", "declared, not implied by absence"
    assert pins["render_unit"] == "turns" and "k" not in pins, "a budget, not a unit count"
    assert pins["competitor_context_budget_tokens"] == 1_000_000
    assert pins["competitor_config_hash"] == competitor.config_hash(_config())
    assert len(pins["competitor_config_hash"]) == 64


def test_agentmemory_refuses_a_process_without_the_configured_hash_seed(
    fake_agentmemory, monkeypatch
):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        AgentMemorySystem(_config())


def test_a_competitor_arm_needs_its_config_file():
    from evals.__main__ import build_system

    with pytest.raises(SystemExit, match="--competitor-config"):
        build_system("agentmemory")


def _pins_kwargs(**extra):
    base = dict(
        dataset_file="d.json", dataset_sha256="s", system="agentmemory", limit=2,
        sample_strategy="stratified", sample_seed=0, reader_transport="openai",
        reader_model="gpt-4o-2024-08-06", reader_digest=None, reader_num_ctx=128000,
        reader_seed=0, reader_top_k=None, reader_num_gpu=None, reader_num_thread=None,
        reader_num_batch=None, reader_flash_attention=None, reader_cache_ram=None,
        reader_transport_version="openai", reader_prompt_version="mnimi-con-v1",
        reader_prompt_hash="p", reader_answer_reserve=800, reader_scaffold_tokens=0,
        reader_chars_per_token=4, render_template_hash="r",
    )
    base.update(extra)
    return base


def test_build_pins_carries_the_competitor_keys_and_omits_them_when_absent():
    with_keys = artifacts.build_pins(**_pins_kwargs(
        competitor_name="agentmemory", competitor_version="v", competitor_embedder="e",
        competitor_llm="none", competitor_config_hash="h", competitor_context_budget_tokens=5364))
    assert with_keys["competitor_name"] == "agentmemory" and with_keys["competitor_llm"] == "none"
    assert with_keys["competitor_context_budget_tokens"] == 5364
    without = artifacts.build_pins(**_pins_kwargs(system="mnimi"))
    assert not any(k.startswith("competitor_") for k in without)


def test_published_pins_still_hash_to_their_recorded_pins_hash():
    root = Path(__file__).resolve().parents[1] / "results" / "published"
    for arm in ("mnimi__500q_gpt4o_p6time", "naive_rag__500q_gpt4o", "oracle__500q_gpt4o"):
        payload = json.loads((root / arm / "pins.json").read_text(encoding="utf-8"))
        assert artifacts.pins_hash(payload["pins"]) == payload["pins_hash"], arm


def _hits(n, text_len=40):
    return [competitor.Hit(f"id{i}", f"m{i:02d} " + "x" * (text_len - 4), TS_A if i % 2 else TS_B)
            for i in range(n)]


def _count(text):
    return len(text)  # characters: exact and dependency-free for the test


def test_fit_to_budget_takes_the_longest_recall_order_prefix_that_fits():
    hits = _hits(10)
    for m in range(1, 11):
        budget = _count(competitor.render_hits(hits[:m], "text"))
        fitted = competitor.fit_to_budget(hits, "text", budget, _count)
        assert fitted == hits[:m], m
        assert competitor.fit_to_budget(hits, "text", budget - 1, _count) == hits[:max(m - 1, 1)]


def test_fit_to_budget_keeps_the_top_hit_even_when_it_alone_is_over():
    hits = _hits(3, text_len=500)
    assert competitor.fit_to_budget(hits, "text", 10, _count) == hits[:1]
    assert competitor.fit_to_budget([], "text", 10, _count) == []


def test_make_counter_names_its_tokenizer_and_refuses_an_unknown_one():
    assert competitor.make_counter("chars/4")("abcdefgh") == 2
    with pytest.raises(ValueError, match="token counter"):
        competitor.make_counter("words")


def test_a_competitor_needs_a_context_budget(fake_agentmemory):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    block = _config()
    del block["context"]
    with pytest.raises(ValueError, match="budget_tokens"):
        AgentMemorySystem(block)


def test_the_adapter_fills_the_budget_in_recall_order(fake_agentmemory):
    from evals.systems.agentmemory_v4 import AgentMemorySystem

    system = AgentMemorySystem(_config(context={"budget_tokens": 40, "candidates": 50,
                                                "tokenizer": "chars/4"}))
    system.reset()
    system.add([{"role": "user", "content": "a" * 40, "ts": TS_A},
                {"role": "user", "content": "b" * 40, "ts": TS_A},
                {"role": "user", "content": "c" * 40, "ts": TS_A}])
    ctx = system.get_context("q")
    assert system.retrieved_ids() == ["s000#0", "s000#1"], "two hits render to 34, three to 47"
    assert "a" * 40 in ctx and "b" * 40 in ctx and "c" * 40 not in ctx


def test_the_manifest_describes_a_competitor_arm_without_missing_fields():
    from evals import manifest

    pins = {"system": "agentmemory", "competitor_name": "agentmemory",
            "competitor_embedder": "sentence-transformers/all-mpnet-base-v2@e8c3b32e",
            "competitor_context_budget_tokens": 5364, "render_template_hash": "r",
            "embed_template_hash": None}
    config = manifest.arm_config(pins)
    assert config["top_k"] == "context budget 5364 tokens (units vary per question)"
    assert config["embedder"] == "sentence-transformers/all-mpnet-base-v2@e8c3b32e"
    assert config["chunk_unit"] == "the system's own memories"
    assert config["dedup"] == {"on": "the system's own write policy"}
    assert config["competitor_context_budget_tokens"] == 5364
