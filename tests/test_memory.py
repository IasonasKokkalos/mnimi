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


class TestEmbedTextFreeze:
    """THE SAFETY PROPERTY of the embed/render split, locked byte-for-byte.

    Every expected string below was captured by running the fixture through
    the PRE-SPLIT code (v1.2.1, before `turns`/`render_records` existed), not
    written by hand. If any of these assertions fails, the embed text has
    moved: every vector moves with it, the dedup key moves with it, and the
    0.95 threshold's selection evidence — which cannot be regenerated, since
    further threshold selection against LongMemEval is prohibited — is void.
    That is a hard stop, not a formatting nit.
    """

    FIXTURE = [
        {"role": "user", "content": "I moved to Athens last spring.",
         "ts": "2023/05/20 (Sat) 02:21"},
        {"role": "assistant", "content": "Noted - Athens is lovely in spring.",
         "ts": "2023/05/20 (Sat) 02:21"},
        {"role": "user",
         "content": "Two things:\n1. I adopted a dog.\n2. Named him Ari.",
         "ts": "2023-07-01T09:30:00"},
        {"role": "assistant", "content": "Congratulations!\nAri is a great name.",
         "ts": "2023-07-01T09:30:00"},
        {"role": "user", "content": "Remind me about the dentist.", "ts": "2023-08-15"},
        {"role": "assistant", "content": "Your appointment is on Monday.",
         "ts": "2023-08-15"},
        {"role": "user", "content": "No timestamp on this one."},
        {"role": "assistant", "content": "Understood, no date recorded."},
        {"role": "user", "content": "Budget: $1,200.50 -- (approx.) for the trip?!",
         "ts": "2024/01/02 (Tue) 18:05"},
        {"role": "assistant", "content": "Got it: $1,200.50, noted.",
         "ts": "2024/01/02 (Tue) 18:05"},
    ]

    # Captured from pre-change code. Do not regenerate these from current code:
    # a lock recomputed from the thing it locks locks nothing.
    EXPECTED = [
        ("[Session date: 2023/05/20] I moved to Athens last spring.\n"
         "Noted - Athens is lovely in spring.",
         "user+assistant", "2023/05/20 (Sat) 02:21"),
        ("[Session date: 2023-07-01] Two things:\n1. I adopted a dog.\n"
         "2. Named him Ari.\nCongratulations!\nAri is a great name.",
         "user+assistant", "2023-07-01T09:30:00"),
        ("[Session date: 2023-08-15] Remind me about the dentist.\n"
         "Your appointment is on Monday.",
         "user+assistant", "2023-08-15"),
        ("No timestamp on this one.\nUnderstood, no date recorded.",
         "user+assistant", None),
        ("[Session date: 2024/01/02] Budget: $1,200.50 -- (approx.) for the trip?!\n"
         "Got it: $1,200.50, noted.",
         "user+assistant", "2024/01/02 (Tue) 18:05"),
    ]

    SOLO_FIXTURE = [
        {"role": "assistant", "content": "Welcome back! How can I help?",
         "ts": "2023/03/03 (Fri) 11:11"},
        {"role": "user", "content": "First question about tickets.",
         "ts": "2023/03/03 (Fri) 11:11"},
        {"role": "user", "content": "Actually, second question about hotels.",
         "ts": "2023/03/03 (Fri) 11:11"},
        {"role": "user", "content": "   ", "ts": "2023/03/03 (Fri) 11:11"},
    ]

    SOLO_EXPECTED = [
        ("[Session date: 2023/03/03] Welcome back! How can I help?",
         "assistant", "2023/03/03 (Fri) 11:11"),
        ("[Session date: 2023/03/03] First question about tickets.",
         "user", "2023/03/03 (Fri) 11:11"),
        ("[Session date: 2023/03/03] Actually, second question about hotels.",
         "user", "2023/03/03 (Fri) 11:11"),
    ]

    def test_embed_text_is_byte_identical_to_pre_split_capture(self):
        from mnimi.memory import _messages_to_rounds

        rounds = _messages_to_rounds(self.FIXTURE)
        assert [(r.content, r.roles, r.ts) for r in rounds] == self.EXPECTED

    def test_solo_turn_embed_text_is_byte_identical_to_pre_split_capture(self):
        from mnimi.memory import _messages_to_rounds

        rounds = _messages_to_rounds(self.SOLO_FIXTURE)
        assert [(r.content, r.roles, r.ts) for r in rounds] == self.SOLO_EXPECTED

    def test_dedup_key_is_computed_from_the_frozen_embed_text(self):
        """The exact-dup key derives from `content` — frozen text in, frozen
        key out. Guards against the key quietly moving to the render text."""
        from mnimi.memory import _messages_to_rounds, _normalize

        rounds = _messages_to_rounds(self.FIXTURE)
        assert [_normalize(r.content) for r in rounds[:2]] == [
            "session date 20230520 i moved to athens last spring "
            "noted athens is lovely in spring",
            "session date 20230701 two things 1 i adopted a dog 2 named him ari "
            "congratulations ari is a great name",
        ]


def test_stored_record_carries_verbatim_turns_for_rendering(tmp_path):
    """`content` stays the frozen embed text; `turns` carries what rendering
    needs. The two travel together but never mix."""
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("what should I cook for the dinner party?"),
            _message("a mushroom risotto pairs well", role="assistant"),
        ],
        user_id="u1",
    )
    [record] = memory.recall("dinner", "u1")
    assert record.turns == [
        {"role": "user", "content": "what should I cook for the dinner party?"},
        {"role": "assistant", "content": "a mushroom risotto pairs well"},
    ]
    assert "user:" not in record.content, "embed text must stay role-free"


def test_get_context_renders_roles_and_full_timestamp_not_embed_text(tmp_path):
    """The reader sees speaker labels and the verbatim timestamp; the embed
    text (date-only fold, no roles) never reaches the reader anymore."""
    memory = _memory(tmp_path)
    memory.add(
        [
            _message("what should I cook?", ts="2023/05/20 (Sat) 02:21"),
            _message("a mushroom risotto", role="assistant", ts="2023/05/20 (Sat) 02:21"),
        ],
        user_id="u1",
    )
    context = memory.get_context("dinner", "u1")
    assert context == (
        "[Session date: 2023/05/20 (Sat) 02:21]\n"
        "user: what should I cook?\n"
        "assistant: a mushroom risotto"
    )


def test_embed_and_render_template_hashes_are_pinned_and_distinct():
    from mnimi.memory import (
        EMBED_TEMPLATE,
        RENDER_TEMPLATE,
        embed_template_hash,
        render_template_hash,
    )

    assert len(embed_template_hash()) == 64
    assert len(render_template_hash()) == 64
    assert embed_template_hash() != render_template_hash()
    # The hashed constants are the ones the code paths actually run — locked
    # here so the hash can never describe a parallel claim.
    assert EMBED_TEMPLATE == "[Session date: ${date}] ${text}"
    assert RENDER_TEMPLATE == "[Session date: ${ts}]\n${role}: ${content}"


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
