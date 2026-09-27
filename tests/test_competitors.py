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

# -- Mem0 OSS (PHASE8 Task 14) --------------------------------------------------------------


class _FakeMem0Client:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeMem0:
    instances: list = []

    def __init__(self, config):
        self.config, self.adds, self.searches, self.closed = config, [], [], False
        self.vector_store = types.SimpleNamespace(client=_FakeMem0Client())
        _FakeMem0.instances.append(self)

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def add(self, messages, *, user_id=None, metadata=None, infer=True, **extra):
        assert not extra, f"unexpected add() keywords {sorted(extra)}"
        self.adds.append({"messages": messages, "user_id": user_id, "metadata": metadata,
                          "infer": infer})

    def search(self, query, *, top_k=20, filters=None, threshold=0.1, **extra):
        assert not extra, f"unexpected search() keywords {sorted(extra)}"
        self.searches.append({"query": query, "top_k": top_k, "filters": filters,
                              "threshold": threshold})
        rows = []
        for a, add in enumerate(self.adds):
            for m, msg in enumerate(add["messages"]):
                rows.append({"id": f"mem{a}-{m}", "memory": msg["content"], "score": 0.9,
                             "metadata": add["metadata"]})
        return {"results": rows[:top_k]}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_mem0(monkeypatch, tmp_path):
    mod = types.ModuleType("mem0")
    mod.Memory = _FakeMem0
    mod.__version__ = "2.2.1-fake"
    monkeypatch.setitem(sys.modules, "mem0", mod)
    monkeypatch.delenv("MEM0_TELEMETRY", raising=False)
    _FakeMem0.instances.clear()
    return mod


def _mem0_config(**overrides):
    block = {"package": "mem0ai==fake",
             "llm": {"provider": "fake", "config": {"model": "gpt-4o-mini-2024-07-18",
                                                    "temperature": 0}},
             "embedder": {"provider": "fake", "repo": "BAAI/bge-small-en-v1.5",
                          "revision": "5c38", "embedding_dims": 384},
             "vector_store": {"provider": "qdrant",
                              "config": {"collection_name": "eval", "embedding_model_dims": 384}},
             "version": "v1.1", "add": {"infer": True}, "search": {"threshold": 0.1},
             "context": {"budget_tokens": 1_000_000, "candidates": 1000, "tokenizer": "chars/4"}}
    block.update(overrides)
    return block


def test_mem0_adapter_hands_the_session_date_over_as_metadata_only(fake_mem0):
    import os

    from evals.systems.mem0_oss import Mem0System

    system = Mem0System(_mem0_config())
    assert os.environ["MEM0_TELEMETRY"] == "False", "telemetry off before mem0 is imported"
    system.reset()
    system.add([{"role": "user", "content": "I live in Boston.", "ts": TS_A},
                {"role": "assistant", "content": "Noted.", "ts": TS_A}])
    memory = _FakeMem0.instances[-1]
    (add,) = memory.adds
    assert add["messages"] == [{"role": "user", "content": "I live in Boston."},
                               {"role": "assistant", "content": "Noted."}], "no harness invention"
    assert add["metadata"] == {"session_date": TS_A} and add["infer"] is True
    assert add["user_id"] == "eval"
    config = memory.config
    assert config["llm"] == {"provider": "fake", "config": {"model": "gpt-4o-mini-2024-07-18",
                                                            "temperature": 0}}
    assert config["vector_store"]["config"]["path"].endswith("qdrant")
    assert config["history_db_path"].endswith("history.db") and config["version"] == "v1.1"


def test_mem0_adapter_fills_the_budget_from_its_own_search_order(fake_mem0):
    from evals.systems.mem0_oss import Mem0System

    system = Mem0System(_mem0_config())
    system.reset()
    system.add([{"role": "user", "content": "I adopted a cat.", "ts": TS_B}])
    system.add([{"role": "user", "content": "I live in Boston.", "ts": TS_A}])
    ctx = system.get_context("where do I live")
    (search,) = _FakeMem0.instances[-1].searches
    assert search == {"query": "where do I live", "top_k": 1000, "filters": {"user_id": "eval"},
                      "threshold": 0.1}
    assert system.retrieved_ids() == ["mem0-0", "mem1-0"], "search order"
    assert ctx.index(f"[Session date: {TS_A}]") < ctx.index(f"[Session date: {TS_B}]")
    assert "memory: I live in Boston." in ctx and "memory: I adopted a cat." in ctx


def test_mem0_reset_closes_the_store_and_its_vector_client(fake_mem0):
    from evals.systems.mem0_oss import Mem0System

    system = Mem0System(_mem0_config())
    system.reset()
    first = _FakeMem0.instances[-1]
    system.reset()
    assert first.closed and first.vector_store.client.closed
    assert _FakeMem0.instances[-1] is not first
    assert first.config["vector_store"]["config"]["path"] != \
        _FakeMem0.instances[-1].config["vector_store"]["config"]["path"], "a fresh store"


def test_mem0_pins_name_its_llm_and_embedder(fake_mem0):
    from evals.systems.mem0_oss import Mem0System

    pins = Mem0System(_mem0_config()).retrieval_pins()
    assert pins["competitor_name"] == "mem0" and pins["competitor_version"] == "mem0ai==fake"
    assert pins["competitor_llm"] == "fake:gpt-4o-mini-2024-07-18"
    assert pins["competitor_embedder"] == "BAAI/bge-small-en-v1.5@5c38"
    assert pins["competitor_context_budget_tokens"] == 1_000_000


def test_mem0_usage_is_recorded_and_taken_per_question(fake_mem0):
    from evals.systems.mem0_oss import Mem0System

    system = Mem0System(_mem0_config())
    system._record_usage("gpt-4o-mini-2024-07-18", 100, 10, cached=64)
    system._record_usage("gpt-4o-mini-2024-07-18", 50, 5)
    assert system.take_llm_usage() == {
        "gpt-4o-mini-2024-07-18": {"calls": 2, "prompt": 150, "completion": 15, "cached": 64}}
    assert system.take_llm_usage() == {}


# -- parallel contexts (--workers) ------------------------------------------------------------


def _questions(n):
    from evals.dataset import Question, Session

    return [Question(question_id=f"q{i}", question_type="multi-session", question=f"what {i}?",
                     answer="a", question_date=TS_B,
                     sessions=[Session(session_id=f"s{j}", date=TS_A,
                                       turns=[{"role": "user", "content": "x"}])
                               for j in range(i % 3 + 1)],
                     answer_session_ids=[])
            for i in range(n)]


def test_contexts_in_parallel_equal_the_sequential_ones():
    from evals import runner

    from _workers_fake import make_counting_system

    questions = _questions(7)
    sequential = runner.contexts_for(questions, system=make_counting_system())
    parallel = runner.contexts_for(questions, factory=make_counting_system, workers=3)
    assert [(c.context, c.retrieved_ids, c.llm_usage) for c in parallel] == \
        [(c.context, c.retrieved_ids, c.llm_usage) for c in sequential]
    assert [c.context for c in sequential][:3] == ["what 0?|1", "what 1?|2", "what 2?|3"]


def test_predict_stats_sums_the_llm_usage_it_is_given():
    from evals import runner

    stats = runner.PredictStats()
    stats.record_llm_usage({"m": {"calls": 1, "prompt": 5, "completion": 1, "cached": 2}})
    stats.record_llm_usage({"m": {"calls": 2, "prompt": 10, "completion": 2}})
    assert stats.as_resolved("openai", "gpt-4o-2024-08-06")["competitor_llm"] == {
        "m": {"calls": 3, "prompt": 15, "completion": 3, "cached": 2}}
