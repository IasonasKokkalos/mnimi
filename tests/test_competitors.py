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

# -- OMEGA's retrieval over per-round verbatim storage (PHASE8 Task 15) ----------------------


def test_render_hits_renders_a_hit_that_carries_turns_as_those_turns():
    from mnimi.memory import render_turns

    turns = [{"role": "user", "content": "I moved."}, {"role": "assistant", "content": "Nice."}]
    hit = competitor.Hit("r0", "embed text", TS_A, turns)
    assert competitor.render_hits([hit], "text") == render_turns(
        [{**t, "ts": TS_A} for t in turns], fmt="text")


class _FakeOmegaResult:
    def __init__(self, id_, content, metadata):
        self.id, self.content, self.metadata = id_, content, metadata


class _FakeOmegaConn:
    def __init__(self, events):
        self.events, self.updates = events, []

    def execute(self, sql, params=()):
        assert sql == "UPDATE memories SET created_at = ? WHERE node_id = ?", sql
        self.updates.append(params)
        self.events.append("update")
        return self

    def commit(self):
        self.events.append("commit")


class _FakeOmegaStore:
    instances: list = []

    def __init__(self, db_path=None, decompose_queries=True):
        self.db_path, self.decompose_queries = db_path, decompose_queries
        self.records, self.queries, self.closed = [], [], False
        self.skip_inference, self.events = [], []
        self._conn = _FakeOmegaConn(self.events)
        _FakeOmegaStore.instances.append(self)

    def store(self, content, session_id=None, metadata=None, skip_inference=False, **extra):
        assert not extra, f"unexpected store() keywords {sorted(extra)}"
        node_id = f"mem-{len(self.records)}"
        self.records.append((node_id, content, session_id, dict(metadata or {})))
        self.skip_inference.append(skip_inference)
        self.events.append("store")
        return node_id

    def query(self, query_text, limit=10, **extra):
        self.queries.append((query_text, limit, extra))
        self.events.append("query")
        rows = [_FakeOmegaResult(nid, content, {**meta, "session_id": sid})
                for nid, content, sid, meta in reversed(self.records)]
        return rows[:limit]

    def _invalidate_query_cache(self):
        self.events.append("invalidate")

    def close(self):
        self.closed = True


@pytest.fixture
def fake_omega(monkeypatch):
    omega = types.ModuleType("omega")
    store_mod = types.ModuleType("omega.sqlite_store")
    store_mod.SQLiteStore = _FakeOmegaStore
    embedding = types.ModuleType("omega.embedding")
    embedding.get_embedding_model_info = lambda: {"model_name": "bge-small-en-v1.5",
                                                  "backend": "onnx"}
    embedding.is_embedding_degraded = lambda: False
    omega.sqlite_store, omega.embedding, omega.__version__ = store_mod, embedding, "1.5.17-fake"
    for name, mod in (("omega", omega), ("omega.sqlite_store", store_mod),
                      ("omega.embedding", embedding)):
        monkeypatch.setitem(sys.modules, name, mod)
    for var in ("OMEGA_HOME", "OMEGA_ONNX_MODEL_DIR", "OMEGA_RERANKER_AUTODOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    _FakeOmegaStore.instances.clear()
    return omega


def _omega_config(**overrides):
    block = {"package": "omega-memory==fake", "store": {"decompose_queries": True},
             "embedder": None, "reranker": None,
             "context": {"budget_tokens": 1_000_000, "candidates": 100, "tokenizer": "chars/4"}}
    block.update(overrides)
    return block


def test_omega_stores_naive_rags_rounds_and_renders_their_turns(fake_omega):
    import os

    from evals.systems.omega_retrieval import OmegaSystem

    from mnimi.memory import _messages_to_rounds

    system = OmegaSystem(_omega_config())
    assert os.environ["OMEGA_HOME"].startswith(system._scratch.name), "no side file in ~/.omega"
    system.reset()
    session_a = [{"role": "user", "content": "I live in Boston.", "ts": TS_A},
                 {"role": "assistant", "content": "Noted.", "ts": TS_A}]
    session_b = [{"role": "user", "content": "I adopted a cat.", "ts": TS_B}]
    system.add(session_a)
    system.add(session_b)
    store = _FakeOmegaStore.instances[-1]
    expected = [r.content for r in _messages_to_rounds(session_a) + _messages_to_rounds(session_b)]
    assert [content for _id, content, _sid, _meta in store.records] == expected, \
        "naive_rag's frozen embed text, one record per round"
    assert [sid for _id, _c, sid, _m in store.records] == ["s000", "s001"]
    # the session date through OMEGA's own event-time field, not its wall-clock created_at
    assert store.records[0][3] == {"session_date": TS_A, "round": "r0",
                                   "referenced_date": "2023-05-20T09:00:00+00:00"}
    ctx = system.get_context("where do I live")
    assert store.queries == [("where do I live", 100, {})]
    assert system.retrieved_ids() == ["mem-1", "mem-0"], "OMEGA's own order and ids"
    assert "user: I live in Boston." in ctx and "assistant: Noted." in ctx
    assert ctx.index(f"[Session date: {TS_A}]") < ctx.index(f"[Session date: {TS_B}]")


def test_omega_reset_closes_the_store_and_opens_a_fresh_file(fake_omega):
    from evals.systems.omega_retrieval import OmegaSystem

    system = OmegaSystem(_omega_config())
    system.reset()
    first = _FakeOmegaStore.instances[-1]
    system.reset()
    assert first.closed and _FakeOmegaStore.instances[-1].db_path != first.db_path


def test_omega_refuses_a_degraded_or_other_embedder(fake_omega, monkeypatch):
    from evals.systems import omega_retrieval

    monkeypatch.setattr(omega_retrieval, "_model_dir", lambda spec, root: str(root))
    monkeypatch.setattr(fake_omega.embedding, "is_embedding_degraded", lambda: True)
    spec = {"repo": "BAAI/bge-small-en-v1.5", "revision": "5c38", "omega_name": "bge-small-en-v1.5"}
    with pytest.raises(RuntimeError, match="embedder"):
        omega_retrieval.OmegaSystem(_omega_config(embedder=spec))


def test_omega_refuses_a_candidate_pool_over_the_vec0_k_ceiling(fake_omega):
    from evals.systems.omega_retrieval import OmegaSystem

    context = {"budget_tokens": 5364, "candidates": 1000, "tokenizer": "chars/4"}
    with pytest.raises(ValueError, match="4096"):
        OmegaSystem(_omega_config(context=context))


def test_omega_turns_off_its_llm_query_expansion(fake_omega, monkeypatch):
    import os

    from evals.systems.omega_retrieval import OmegaSystem

    monkeypatch.setenv("OMEGA_QUERY_EXPANSION", "1")
    OmegaSystem(_omega_config())
    assert os.environ["OMEGA_QUERY_EXPANSION"] == "0", "competitor_llm is pinned 'none'"


def test_omega_lays_out_pinned_files_and_refuses_other_bytes(tmp_path, monkeypatch):
    import hashlib

    from evals.systems import omega_retrieval

    snap = tmp_path / "snap"
    (snap / "onnx").mkdir(parents=True)
    (snap / "onnx" / "model.onnx").write_bytes(b"pinned bytes")
    calls = []
    monkeypatch.setattr(omega_retrieval, "_snapshot",
                        lambda repo, revision, patterns: calls.append(patterns) or snap)
    digest = hashlib.sha256(b"pinned bytes").hexdigest()
    spec = {"repo": "org/model", "revision": "abc",
            "files": [["onnx/model.onnx", "model.onnx", digest]]}
    target = omega_retrieval._lay_out(spec, tmp_path / "out")
    assert (target / "model.onnx").read_bytes() == b"pinned bytes"
    assert calls == [["onnx/model.onnx"]], "only the named files are fetched"
    omega_retrieval._lay_out(spec, tmp_path / "out")
    assert len(calls) == 1, "files already in place and matching are not fetched again"
    bad = {**spec, "files": [["onnx/model.onnx", "model.onnx", "0" * 64]]}
    with pytest.raises(RuntimeError, match="does not match"):
        omega_retrieval._lay_out(bad, tmp_path / "other")


def test_omega_follows_its_authors_type_independent_ingest(fake_omega):
    from datetime import datetime, timedelta, timezone

    from evals.systems.competitor import epoch
    from evals.systems.omega_retrieval import OmegaSystem

    question_date = "2023/05/30 (Tue) 10:00"
    system = OmegaSystem(_omega_config(ingest={"skip_inference": True,
                                               "backdate_created_at": True}))
    system.reset()
    system.add([{"role": "user", "content": "I live in Boston.", "ts": TS_A}])
    system.add([{"role": "user", "content": "I adopted a cat.", "ts": TS_B}])
    store = _FakeOmegaStore.instances[-1]
    assert store.skip_inference == [True, True], "the authors' script stores with skip_inference"
    system.set_question_date(question_date)
    system.get_context("where do I live")
    assert store.events[-4:] == ["update", "commit", "invalidate", "query"], \
        "backdated once, after every store and before the query"
    stamps = {nid: datetime.fromisoformat(stamp) for stamp, nid in store._conn.updates}
    assert stamps["mem-1"] - stamps["mem-0"] == timedelta(seconds=epoch(TS_B) - epoch(TS_A))
    age = datetime.now(timezone.utc) - stamps["mem-0"]
    expected = timedelta(seconds=epoch(question_date) - epoch(TS_A))
    assert abs(age - expected) < timedelta(minutes=1), "the question's date reads as now"


def test_omega_ingest_defaults_are_the_librarys(fake_omega):
    from evals.systems.omega_retrieval import OmegaSystem

    system = OmegaSystem(_omega_config())
    system.reset()
    system.add([{"role": "user", "content": "I live in Boston.", "ts": TS_A}])
    system.set_question_date("2023/05/30 (Tue) 10:00")
    system.get_context("where do I live")
    store = _FakeOmegaStore.instances[-1]
    assert store.skip_inference == [False] and store._conn.updates == []


def test_omega_pins_name_the_system(fake_omega):
    from evals.systems.omega_retrieval import OmegaSystem

    pins = OmegaSystem(_omega_config()).retrieval_pins()
    assert pins["competitor_name"] == "omega" and pins["competitor_llm"] == "none"
    assert pins["competitor_version"] == "omega-memory==fake"
    assert pins["competitor_context_budget_tokens"] == 1_000_000
