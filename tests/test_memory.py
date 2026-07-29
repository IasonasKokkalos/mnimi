"""Memory facade behavior: config threading, dedup, and the add() contract."""

from __future__ import annotations

import pytest

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder


def test_config_defaults_match_spec():
    config = MemoryConfig()
    assert config.dedup_cosine_threshold == 0.95
    assert config.top_k == 10


def test_config_is_threaded_into_memory(tmp_path):
    config = MemoryConfig(dedup_cosine_threshold=0.9, top_k=3)
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder(), config)
    assert memory.config is config


def test_memory_defaults_to_spec_config(tmp_path):
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder())
    assert memory.config.dedup_cosine_threshold == 0.95
    assert memory.config.top_k == 10


def _message(content: str, role: str = "user", ts: str = "2023-05-20") -> dict:
    return {"role": role, "content": content, "ts": ts}


def test_exact_duplicate_after_normalization_is_stored_once(tmp_path):
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder())
    memory.add([_message("I adopted a golden retriever puppy last spring.")], user_id="u1")
    memory.add([_message("i adopted a GOLDEN retriever puppy, last spring")], user_id="u1")

    assert memory.store.count("u1") == 1


def test_cosine_near_duplicate_is_stored_once(tmp_path):
    # The pair below measures cosine 0.923 under HashingEmbedder, so the
    # threshold is stated explicitly rather than inherited from the default:
    # a test that depends on the default's value fails when the default is
    # legitimately retuned, which says nothing about the behaviour under test.
    memory = Memory(
        str(tmp_path / "mem.db"), HashingEmbedder(), MemoryConfig(dedup_cosine_threshold=0.90)
    )
    memory.add(
        [_message("every saturday morning I hike the coastal trail with my dog before work")],
        user_id="u1",
    )
    memory.add(
        [_message("every saturday morning I hike the coastal trail with my dog before breakfast")],
        user_id="u1",
    )

    assert memory.store.count("u1") == 1


def test_distinct_facts_are_stored_twice(tmp_path):
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder())
    memory.add([_message("my sister works as a marine biologist in Crete")], user_id="u1")
    memory.add([_message("the conference talk covered sqlite virtual tables")], user_id="u1")

    assert memory.store.count("u1") == 2


def _memory(tmp_path) -> Memory:
    return Memory(str(tmp_path / "mem.db"), HashingEmbedder())


def test_add_rejects_bare_strings(tmp_path):
    memory = _memory(tmp_path)
    with pytest.raises(TypeError):
        memory.add("I live in Athens", user_id="u1")


def test_add_rejects_lists_of_strings(tmp_path):
    memory = _memory(tmp_path)
    with pytest.raises(TypeError):
        memory.add(["I live in Athens"], user_id="u1")


def test_user_assistant_round_becomes_one_record(tmp_path):
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("what should I cook for the dinner party on friday?"),
            _message("a mushroom risotto pairs well with your pinot", role="assistant"),
        ],
        user_id="u1",
    )
    assert memory.store.count("u1") == 1


def test_each_round_gets_its_own_record(tmp_path):
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("my sister works as a marine biologist in Crete"),
            _message("that sounds like fascinating field work", role="assistant"),
            _message("the conference talk covered sqlite virtual tables"),
            _message("virtual tables are a powerful extension point", role="assistant"),
        ],
        user_id="u1",
    )
    assert memory.store.count("u1") == 2


def test_unpaired_turn_gets_its_own_record(tmp_path):
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("my sister works as a marine biologist in Crete"),
            _message("the conference talk covered sqlite virtual tables"),
            _message("virtual tables are a powerful extension point", role="assistant"),
        ],
        user_id="u1",
    )
    # First user turn has no assistant reply: it stands alone. The next two pair.
    assert memory.store.count("u1") == 2


def test_session_date_is_folded_into_content_and_reader_visible(tmp_path):
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("what should I cook for the dinner party?", ts="2023-05-20"),
            _message("a mushroom risotto pairs well", role="assistant", ts="2023-05-20"),
        ],
        user_id="u1",
    )
    context = memory.get_context("dinner plans", "u1")
    assert "[Session date: 2023-05-20]" in context


def test_role_is_metadata_not_embedded_in_content(tmp_path):
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("what should I cook for the dinner party?"),
            _message("a mushroom risotto pairs well", role="assistant"),
        ],
        user_id="u1",
    )
    (content,) = memory.store.contents("u1")
    assert "user:" not in content and "assistant:" not in content

    [record] = memory.recall("dinner", "u1")
    assert record.source == "user+assistant"


def test_round_record_carries_the_session_ts(tmp_path):
    memory = _memory(tmp_path)
    memory.add([_message("I moved to Thessaloniki", ts="2023-07-01")], user_id="u1")
    [record] = memory.recall("where do I live", "u1")
    assert record.created_at == "2023-07-01"


def test_add_requires_ts_on_messages(tmp_path):
    memory = _memory(tmp_path)
    with pytest.raises(ValueError, match="ts"):
        memory.add([{"role": "user", "content": "undated turn"}], user_id="u1")


def test_recall_reads_top_k_from_config(tmp_path):
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder(), MemoryConfig(top_k=2))
    facts = [
        "my sister works as a marine biologist in Crete",
        "the conference talk covered sqlite virtual tables",
        "a recipe for baking sourdough bread with fresh yeast",
        "the golden retriever puppy chased a ball across the park",
    ]
    for fact in facts:
        memory.add([_message(fact)], user_id="u1")
    assert memory.store.count("u1") == 4

    assert len(memory.recall("tell me about my life", "u1")) == 2


def test_dedup_threshold_is_read_from_config_not_hardcoded(tmp_path):
    near_pair = [
        "every saturday morning I hike the coastal trail with my dog before work",
        "every saturday morning I hike the coastal trail with my dog before breakfast",
    ]

    # near_pair measures cosine 0.923, so 0.95 and 0.90 bracket it. Only the
    # config value can decide the outcome — nothing here relies on the default.
    strict = Memory(
        str(tmp_path / "strict.db"),
        HashingEmbedder(),
        MemoryConfig(dedup_cosine_threshold=0.95),
    )
    for text in near_pair:
        strict.add([_message(text)], user_id="u1")
    assert strict.store.count("u1") == 2, "0.95 is above the pair's 0.923: keep both"

    permissive = Memory(
        str(tmp_path / "permissive.db"),
        HashingEmbedder(),
        MemoryConfig(dedup_cosine_threshold=0.90),
    )
    for text in near_pair:
        permissive.add([_message(text)], user_id="u1")
    assert permissive.store.count("u1") == 1, "0.90 is below the pair's 0.923: merge"
