"""Memory facade behavior: config threading, dedup, and the add() contract."""

from __future__ import annotations

import json

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

    [hit] = memory.recall("dinner", "u1")
    assert hit.record.source == "user+assistant"


def test_round_record_carries_the_session_ts(tmp_path):
    memory = _memory(tmp_path)
    memory.add([_message("I moved to Thessaloniki", ts="2023-07-01")], user_id="u1")
    [hit] = memory.recall("where do I live", "u1")
    assert hit.record.created_at == "2023-07-01"


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
    [hit] = memory.recall("dinner", "u1")
    record = hit.record
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


# -- the JSON render format (gpt-4o era presentation pair, 2026-09-12) ---------


def _pair_of_sessions():
    return [
        {"role": "user", "content": "what should I cook?", "ts": "2023/05/20 (Sat) 02:21"},
        {"role": "assistant", "content": "a mushroom risotto", "ts": "2023/05/20 (Sat) 02:21"},
        {"role": "user", "content": "I moved to Lyon — “enfin”", "ts": "2023/06/01 (Thu) 09:00"},
    ]


def test_json_render_format_frames_the_same_blocks_as_text():
    import json

    from mnimi.memory import render_turns

    text = render_turns(_pair_of_sessions())
    rendered = render_turns(_pair_of_sessions(), fmt="json")
    blocks = json.loads(rendered)
    assert [b["session_date"] for b in blocks] == [
        "2023/05/20 (Sat) 02:21",
        "2023/06/01 (Thu) 09:00",
    ]
    assert [len(b["turns"]) for b in blocks] == [2, 1]
    assert blocks[0]["turns"][1] == {"role": "assistant", "content": "a mushroom risotto"}
    # Same information, only the framing differs: every date, role and content
    # the text format carries is in the JSON, and nothing else is.
    assert set(blocks[0]) == {"session_date", "turns"}
    assert set(blocks[0]["turns"][0]) == {"role", "content"}
    assert text.count("[Session date:") == len(blocks)
    # Non-ASCII survives unescaped (ensure_ascii=False) so the reader sees the
    # same characters the text format shows it.
    assert "“enfin”" in rendered and "\\u201c" not in rendered


def test_text_render_format_is_the_default_and_unchanged():
    from mnimi.memory import RENDER_FORMATS, render_turns

    assert RENDER_FORMATS == ("text", "json")
    assert render_turns(_pair_of_sessions()) == render_turns(_pair_of_sessions(), fmt="text")
    with pytest.raises(ValueError):
        render_turns(_pair_of_sessions(), fmt="yaml")


def test_render_template_hash_is_per_format_and_the_default_did_not_move():
    from mnimi.memory import RENDER_JSON_TEMPLATE, render_template_hash

    # The value the local family's published artifacts carry. The JSON format
    # existing must not move it — the whole point of hashing per format.
    assert render_template_hash().startswith("9c03ddae5c33")
    assert render_template_hash("text") == render_template_hash()
    assert render_template_hash("json") != render_template_hash("text")
    assert len(render_template_hash("json")) == 64
    assert RENDER_JSON_TEMPLATE == (
        '[{"session_date": "${ts}", "turns": [{"role": "${role}", "content": "${content}"}]}]'
    )
    with pytest.raises(ValueError):
        render_template_hash("yaml")


def test_get_context_reads_render_format_from_config(tmp_path):
    import json

    from mnimi import Memory, MemoryConfig
    from mnimi.embeddings import HashingEmbedder

    messages = [
        {"role": "user", "content": "what should I cook?", "ts": "2023/05/20 (Sat) 02:21"},
        {"role": "assistant", "content": "a mushroom risotto", "ts": "2023/05/20 (Sat) 02:21"},
    ]
    as_json = Memory(
        str(tmp_path / "j.db"), HashingEmbedder(),
        MemoryConfig(render_format="json", render_unit="turns"),
    )
    as_json.add(messages, user_id="u")
    blocks = json.loads(as_json.get_context("cook", "u"))
    assert blocks[0]["session_date"] == "2023/05/20 (Sat) 02:21"
    assert [t["role"] for t in blocks[0]["turns"]] == ["user", "assistant"]

    # The default unit since v1.9.0 (round+facts) frames each round as an item that
    # carries its facts (none without an extractor) and its turns.
    by_default = Memory(
        str(tmp_path / "d.db"), HashingEmbedder(), MemoryConfig(render_format="json")
    )
    by_default.add(messages, user_id="u")
    items = json.loads(by_default.get_context("cook", "u"))[0]["items"]
    assert items[0]["facts"] == []
    assert [t["role"] for t in items[0]["turns"]] == ["user", "assistant"]

    as_text = Memory(str(tmp_path / "t.db"), HashingEmbedder(), MemoryConfig())
    as_text.add(messages, user_id="u")
    assert as_text.get_context("cook", "u").startswith("[Session date: 2023/05/20")


# -- dedup scope (R3, 2026-09-12) ------------------------------------------------


def test_dedup_scope_session_keeps_cross_session_near_duplicates(tmp_path):
    # Under HashingEmbedder with the date fold in the embed text: same-date
    # cosine 0.973, cross-date 0.865 (measured), so 0.85 brackets both.
    a = "every saturday morning I hike the coastal trail with my dog before work"
    b = a + " again"

    store_scope = Memory(
        str(tmp_path / "s.db"), HashingEmbedder(),
        MemoryConfig(dedup_cosine_threshold=0.85, dedup_scope="store"),
    )
    store_scope.add([_message(a, ts="2023-05-20")], user_id="u")
    store_scope.add([_message(b, ts="2023-06-01")], user_id="u")
    assert store_scope.store.count("u") == 1, "store scope: cross-session near-duplicate dropped"

    session_scope = Memory(
        str(tmp_path / "t.db"), HashingEmbedder(),
        MemoryConfig(dedup_cosine_threshold=0.85, dedup_scope="session"),
    )
    session_scope.add([_message(a, ts="2023-05-20")], user_id="u")
    session_scope.add([_message(b, ts="2023-06-01")], user_id="u")
    assert session_scope.store.count("u") == 2, "session scope: another session is a repeat, kept"
    session_scope.add([_message(b + " really", ts="2023-06-01")], user_id="u")
    assert session_scope.store.count("u") == 2, "same session: still a duplicate, dropped"

    with pytest.raises(ValueError, match="dedup_scope"):
        Memory(str(tmp_path / "x.db"), HashingEmbedder(), MemoryConfig(dedup_scope="user"))


# -- query instruction (R5, 2026-09-12) -----------------------------------------


def test_query_instruction_changes_the_query_vector_not_the_stored_ones(tmp_path):
    from mnimi.embeddings import BGE_QUERY_INSTRUCTION

    assert BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "
    # The library default is the BGE instruction since R5; the bare question
    # is the explicit "" (the published local-family configuration).
    plain = Memory(str(tmp_path / "p.db"), HashingEmbedder(), MemoryConfig(query_instruction=""))
    prefixed = Memory(
        str(tmp_path / "q.db"), HashingEmbedder(),
        MemoryConfig(query_instruction=BGE_QUERY_INSTRUCTION),
    )
    for m in (plain, prefixed):
        m.add([_message("I adopted a cat named Miso")], user_id="u")
    assert plain.store.contents("u") == prefixed.store.contents("u")
    assert plain._query_embedding("cat") != prefixed._query_embedding("cat")
    assert prefixed._query_embedding("cat") == plain._query_embedding(BGE_QUERY_INSTRUCTION + "cat")


# -- chunking (R4, 2026-09-12) -------------------------------------------------


def test_chunked_rounds_embed_pieces_and_render_once(tmp_path):
    long_turn = " ".join(f"fact{i}" for i in range(60))
    m = Memory(
        str(tmp_path / "c.db"), HashingEmbedder(),
        MemoryConfig(chunk_tokens=20, chunk_overlap=5),
    )
    m.add([_message(long_turn), _message("ok", role="assistant")], user_id="u")
    assert m.store.count("u") == 4, "one record per window of the 60-word round"
    hits = m.store.search(m._query_embedding("fact1"), "u", k=10)
    assert {r.round_key for r, _ in hits} == {hits[0][0].round_key} and hits[0][0].round_key
    assert all(r.turns == hits[0][0].turns for r, _ in hits)
    context = m.get_context("fact59", "u")
    assert context.count("fact0 ") == 1, "the round renders once, whole"
    assert "fact59" in context and context.count("[Session date:") == 1

    whole = Memory(str(tmp_path / "w.db"), HashingEmbedder(), MemoryConfig())
    whole.add([_message("a short round")], user_id="u")
    (record,) = [r for r, _ in whole.store.search(whole._query_embedding("short"), "u", k=1)]
    assert record.round_key is None, "chunking off: v1's shape, no key"
    with pytest.raises(ValueError):
        Memory(
            str(tmp_path / "x.db"), HashingEmbedder(), MemoryConfig(chunk_tokens=8, chunk_overlap=8)
        )


def test_recall_counts_rounds_not_windows_when_chunking(tmp_path):
    long_turn = " ".join(f"fact{i}" for i in range(60))
    m = Memory(
        str(tmp_path / "k.db"), HashingEmbedder(),
        MemoryConfig(chunk_tokens=20, chunk_overlap=5, top_k=2),
    )
    m.add([_message(long_turn), _message("ok", role="assistant")], user_id="u")
    m.add([_message("something else entirely, about gardening", ts="2023-05-21")], user_id="u")
    recalled = m.recall("fact3 fact4 fact5", "u")
    assert len(recalled) == 2 and len({hit.record.round_key for hit in recalled}) == 2
    assert m.get_context("fact3", "u").count("[Session date:") == 2, "two rounds, both rendered"


# -- the extraction era (PHASE2 Task 6) ---------------------------------------------------


def test_without_an_extractor_the_write_path_is_v1_byte_for_byte(tmp_path):
    m = Memory(str(tmp_path / "v1.db"), HashingEmbedder())
    m.add([_message("I adopted a cat named Miso"), _message("lovely", role="assistant")],
          user_id="u")
    ((record, _cos),) = m.store.search(m._query_embedding("cat"), "u", k=1)
    assert record.kind == "round" and record.round_key is None and record.raw is None
    assert record.content == "[Session date: 2023-05-20] I adopted a cat named Miso\nlovely"
    assert m.store.count("u") == 1 and m.extractor is None
    assert m.extraction_stats["rounds_sent"] == 0


def test_extractor_adds_fact_records_on_the_rounds_key(tmp_path):
    from mnimi.extract.fake import RuleExtractor

    m = Memory(str(tmp_path / "x.db"), HashingEmbedder(), extractor=RuleExtractor())
    m.add([_message("Yesterday I adopted a cat named Miso. Any tips?", ts="2023-05-20"),
           _message("Congratulations!", role="assistant", ts="2023-05-20")], user_id="u")
    rows = {r.kind: r for r, _ in m.store.search(m._query_embedding("cat"), "u", k=5)}
    assert set(rows) == {"round", "fact"}
    assert rows["round"].round_key == rows["fact"].round_key is not None
    assert rows["fact"].raw == "user: Yesterday I adopted a cat named Miso."
    assert rows["fact"].fact == "The user adopted a cat named Miso."
    assert rows["fact"].valid_time == "2023-05-19" and rows["fact"].time_mention == "Yesterday"
    assert rows["fact"].content == f"{rows['fact'].raw}\n{rows['fact'].fact}"
    assert rows["round"].content.startswith("[Session date: 2023-05-20] ")
    assert rows["fact"].turns == rows["round"].turns, "a fact carries its round's turns"
    assert len(m.recall("cat", "u")) == 1, "k counts rounds: the fact and its round are one hit"
    assert m.extraction_stats["rounds_sent"] == 1 and m.extraction_stats["facts"] == 1


def test_dedup_screens_are_per_kind_and_per_session(tmp_path):
    from mnimi.extract.fake import RuleExtractor

    # A one-sentence round: under HashingEmbedder its fact text is near its
    # round text. Per-kind screens keep both records; a single pool would have
    # dropped the fact as a duplicate of its own round.
    m = Memory(str(tmp_path / "d.db"), HashingEmbedder(),
               MemoryConfig(dedup_cosine_threshold=0.5), extractor=RuleExtractor())
    m.add([_message("I moved to Athens last year.", ts="2023-05-20")], user_id="u")
    assert m.store.count("u", kind="round") == 1 and m.store.count("u", kind="fact") == 1
    # The same fact stated again in another session is a repeat, kept (session scope).
    m.add([_message("I moved to Athens last year.", ts="2023-06-01")], user_id="u")
    assert m.store.count("u", kind="fact") == 2
    # Restated in the same session it is an exact duplicate, dropped.
    m.add([_message("I moved to Athens last year.", ts="2023-06-01")], user_id="u")
    assert m.store.count("u", kind="fact") == 2
    # Store scope keys facts store-wide: the cross-session repeat is dropped too.
    s = Memory(str(tmp_path / "s.db"), HashingEmbedder(),
               MemoryConfig(dedup_cosine_threshold=0.5, dedup_scope="store"),
               extractor=RuleExtractor())
    s.add([_message("I moved to Athens last year.", ts="2023-05-20")], user_id="u")
    s.add([_message("I moved to Athens last year.", ts="2023-06-01")], user_id="u")
    assert s.store.count("u", kind="fact") == 1


def _extracted_memory(db_path, **config):
    from mnimi.extract.fake import RuleExtractor

    m = Memory(str(db_path), HashingEmbedder(), MemoryConfig(**config),
               extractor=RuleExtractor())
    m.add([_message("I run every morning. Yesterday I adopted a cat named Miso. Any tips?",
                    ts="2023/05/20 (Sat) 09:00"),
           _message("Lovely!", role="assistant", ts="2023/05/20 (Sat) 09:00")], user_id="u")
    m.add([_message("what is the weather like", ts="2023/05/21 (Sun) 09:00"),
           _message("Grey.", role="assistant", ts="2023/05/21 (Sun) 09:00")], user_id="u")
    return m


def test_render_units_frame_the_same_rounds(tmp_path):
    turns = _extracted_memory(tmp_path / "a.db", render_unit="turns").get_context("cat", "u")
    round_facts = _extracted_memory(tmp_path / "b.db", render_unit="round+facts").get_context(
        "cat", "u"
    )
    facts = _extracted_memory(tmp_path / "c.db", render_unit="facts").get_context("cat", "u")
    header = "[Session date: 2023/05/20 (Sat) 09:00]"
    user_line = "user: I run every morning. Yesterday I adopted a cat named Miso. Any tips?"
    assert turns.startswith(header + "\n" + user_line + "\nassistant: Lovely!")
    assert "facts:" not in turns
    # round+facts: the same block with the round's facts (all of them, dated) first.
    assert round_facts.startswith(
        header + "\nfacts:\n- The user run every morning.\n"
        "- The user adopted a cat named Miso. (2023-05-19)\n" + user_line
    )
    assert turns.replace(header + "\n", "") in round_facts
    # facts: fact/source lines only; the round without facts falls back to its turns.
    assert facts.startswith(
        header + "\nfact: The user run every morning.\nsource: user: I run every morning.\n"
        "fact: The user adopted a cat named Miso. (2023-05-19)\n"
        "source: user: Yesterday I adopted a cat named Miso."
    )
    assert user_line not in facts
    assert "[Session date: 2023/05/21 (Sun) 09:00]\nuser: what is the weather like" in facts
    json_text = _extracted_memory(tmp_path / "d.db", render_unit="round+facts",
                                  render_format="json").get_context("cat", "u")
    payload = json.loads(json_text)
    assert payload[0]["items"][0]["facts"][1]["valid_time"] == "2023-05-19"
    assert payload[0]["items"][0]["turns"][1]["role"] == "assistant"


def test_render_unit_and_fact_template_hashes_are_pinned_and_distinct():
    from mnimi.memory import (
        FACT_EMBED_TEMPLATE,
        RENDER_UNITS,
        embed_template_hash,
        fact_embed_template_hash,
        render_template_hash,
        render_unit_template_hash,
    )

    assert FACT_EMBED_TEMPLATE == "${raw}\n${fact}"
    assert RENDER_UNITS == ("turns", "round+facts", "facts")
    assert len(embed_template_hash()) == 64
    assert render_template_hash("text").startswith("9c03ddae5c33"), "the format hash did not move"
    assert len({render_unit_template_hash(u) for u in RENDER_UNITS}) == 3
    assert fact_embed_template_hash() != embed_template_hash()
    assert len(fact_embed_template_hash()) == 64
    with pytest.raises(ValueError):
        render_unit_template_hash("paragraphs")
    with pytest.raises(ValueError):
        Memory(":memory:", HashingEmbedder(), MemoryConfig(render_unit="paragraphs"))


# -- Phase 3 Task 2: the screens in the write path -------------------------------------


def test_config_has_the_phase3_fields_with_spec_defaults():
    config = MemoryConfig()
    assert config.dedup_entropy_gate == 2.0 and config.conflict_resolution is True
    assert MemoryConfig(conflict_resolution=False, dedup_entropy_gate=1.5).dedup_entropy_gate == 1.5


def _fact(content, raw, subject=None, predicate=None, obj=None, when=None, salience=1.0):
    from mnimi.extract.protocol import ExtractedFact

    return ExtractedFact(content=content, raw=raw, when=when, subject=subject,
                         predicate=predicate, object=obj, salience=salience)


def _scripted_memory(db_path, script, **config):
    from mnimi.extract.fake import ScriptedExtractor

    # 0.5: under HashingEmbedder the fact pairs below sit well above it, so
    # the k=1 probe fires and the screens — not the threshold — decide.
    return Memory(str(db_path), HashingEmbedder(),
                  MemoryConfig(dedup_cosine_threshold=0.5, **config),
                  extractor=ScriptedExtractor(script))


_VALUE_SCRIPT = {
    "Quick note: I live in Boston now.": [
        _fact("The user lives in Boston.", "user: I live in Boston now.", "user", "lives in",
              "Boston")],
    "Update from me: I live in Seattle now.": [
        _fact("The user lives in Seattle.", "user: I live in Seattle now.", "user", "lives in",
              "Seattle")],
}


def test_value_substitution_keeps_both_facts_within_a_session(tmp_path):
    m = _scripted_memory(tmp_path / "v.db", _VALUE_SCRIPT)
    for text in _VALUE_SCRIPT:
        m.add([_message(text, ts="2023-05-20")], user_id="u")
    assert m.store.count("u", kind="fact") == 2
    assert m.conflict_stats["pairs_screened"] == 1 and m.conflict_stats["kept_value"] == 1
    facts = [r for r, _cos in m.store.search(m._query_embedding("live"), "u", k=5, kind="fact")]
    assert len(facts) == 2 and {f.pair_key for f in facts} == {"user|lives in"}
    assert {f.object for f in facts} == {"Boston", "Seattle"}


def test_conflict_resolution_off_is_the_v19_write_path(tmp_path):
    off = _scripted_memory(tmp_path / "off.db", _VALUE_SCRIPT, conflict_resolution=False)
    for text in _VALUE_SCRIPT:
        off.add([_message(text, ts="2023-05-20")], user_id="u")
    assert off.store.count("u", kind="fact") == 1, "v1.9 drops the second fact as a near-duplicate"
    assert off.conflict_stats["pairs_screened"] == 0


def test_negation_keeps_both_facts_within_a_session(tmp_path):
    script = {
        "By the way, I like jazz.": [
            _fact("The user likes jazz.", "user: I like jazz.", "user", "likes", "jazz")],
        "Actually, I dislike jazz.": [
            _fact("The user dislikes jazz.", "user: I dislike jazz.", "user", "dislikes", "jazz")],
    }
    m = _scripted_memory(tmp_path / "n.db", script)
    for text in script:
        m.add([_message(text, ts="2023-05-20")], user_id="u")
    assert m.store.count("u", kind="fact") == 2 and m.conflict_stats["kept_negation"] == 1


def test_low_entropy_facts_are_not_auto_merged(tmp_path):
    # Two-token facts (1.0 bit) whose exact keys differ ("is") and whose
    # rounds carry a first-person cue, so both reach the cosine probe.
    script = {
        "My cat is Miso.": [_fact("Cat Miso.", "user: My cat is Miso.")],
        "My cat, Miso!": [_fact("Cat Miso!", "user: My cat, Miso!")],
    }
    m = _scripted_memory(tmp_path / "e.db", script)
    for text in script:
        m.add([_message(text, ts="2023-05-20")], user_id="u")
    assert m.conflict_stats["kept_low_entropy"] == 1 and m.store.count("u", kind="fact") == 2
    gated = _scripted_memory(tmp_path / "g.db", script, dedup_entropy_gate=0.0)
    for text in script:
        gated.add([_message(text, ts="2023-05-20")], user_id="u")
    assert gated.store.count("u", kind="fact") == 1, "the gate reads config, not a constant"


def test_screens_never_touch_round_records(tmp_path):
    # Two near-identical rounds in one session with no extractor: the v1.9
    # round screen drops the second, screens on or off.
    for flag in (True, False):
        m = Memory(str(tmp_path / f"r{flag}.db"), HashingEmbedder(),
                   MemoryConfig(dedup_cosine_threshold=0.5, conflict_resolution=flag))
        m.add([_message("I live in Boston now, by the way.", ts="2023-05-20")], user_id="u")
        m.add([_message("I live in Seattle now, by the way.", ts="2023-05-20")], user_id="u")
        assert m.store.count("u", kind="round") == 1 and m.conflict_stats["pairs_screened"] == 0


def test_fact_records_carry_their_pair_key(tmp_path):
    script = {
        "I moved to Seattle.": [
            _fact("The user moved to Seattle.", "user: I moved to Seattle.", "user", "moved to",
                  "Seattle")],
        "I feel happy about it all.": [_fact("The user feels happy.", "user: I feel happy.")],
    }
    m = _scripted_memory(tmp_path / "k.db", script)
    m.add([_message("I moved to Seattle.", ts="2023-05-20")], user_id="u")
    m.add([_message("I feel happy about it all.", ts="2023-05-21")], user_id="u")
    by_key = {}
    for hit in m.recall("Seattle happy", "u"):
        assert hit.record.pair_key is None or hit.record.kind == "fact", "a round has no pair key"
        for fact in m.store.facts_of("u", hit.record.round_key):
            by_key[fact.fact] = fact.pair_key
    assert by_key == {"The user moved to Seattle.": "user|lives in", "The user feels happy.": None}


# -- Phase 3 Task 3: supersede inside add(), consolidate() ----------------------------------


def _two_sessions(m, script, ts_a="2023/01/10 (Tue) 09:00", ts_b="2023/06/10 (Sat) 09:00"):
    (text_a, text_b) = script
    m.add([_message(text_a, ts=ts_a)], user_id="u")
    m.add([_message(text_b, ts=ts_b)], user_id="u")


def _active(m, pair_key):
    return m.store.active_facts_by_pair("u", pair_key)


def test_value_change_across_sessions_supersedes_the_older_fact(tmp_path, caplog):
    import logging

    m = _scripted_memory(tmp_path / "s.db", _VALUE_SCRIPT)
    with caplog.at_level(logging.INFO, logger="mnimi"):
        _two_sessions(m, list(_VALUE_SCRIPT))
    (active,) = _active(m, "user|lives in")
    assert active.object == "Seattle" and active.supersedes is not None
    loser = {r.id: r for r in m.store.facts_with_pair_key("u")}[active.supersedes]
    assert loser.object == "Boston" and loser.salience == 0.0
    assert f"superseded {loser.id}: functional user|lives in: boston -> seattle" in caplog.text
    assert m.conflict_stats["conflicts_functional"] == 1 and m.conflict_stats["superseded"] == 1


def test_an_out_of_order_dated_fact_loses_to_the_later_valid_time(tmp_path):
    script = {
        "I moved to Seattle last month.": [
            _fact("The user moved to Seattle.", "user: I moved to Seattle last month.", "user",
                  "moved to", "Seattle", when="last month")],
        "Back in 2019 I lived in Boston.": [
            _fact("The user lived in Boston in 2019.", "user: Back in 2019 I lived in Boston.",
                  "user", "lived in", "Boston", when="in 2019")],
    }
    m = _scripted_memory(tmp_path / "d.db", script)
    _two_sessions(m, list(script))  # Seattle first (January), Boston later (June)
    (active,) = _active(m, "user|lives in")
    assert active.object == "Seattle" and active.valid_time == "2022-12"


def test_an_incoming_fact_can_lose_and_is_stored_inactive(tmp_path):
    m = _scripted_memory(tmp_path / "l.db", _VALUE_SCRIPT)
    _two_sessions(m, list(_VALUE_SCRIPT), ts_a="2023/06/10 (Sat) 09:00",
                  ts_b="2023/01/10 (Tue) 09:00")
    (active,) = _active(m, "user|lives in")
    assert active.object == "Boston", "the January Seattle fact arrived second and lost"
    assert m.store.count("u", kind="fact") == 2


def test_a_superseded_loser_is_never_a_candidate_again(tmp_path):
    script = {**_VALUE_SCRIPT, "Third time: I live in Denver now.": [
        _fact("The user lives in Denver.", "user: I live in Denver now.", "user", "lives in",
              "Denver")]}
    m = _scripted_memory(tmp_path / "t.db", script)
    stamps = ("2023/01/10 (Tue) 09:00", "2023/03/10 (Fri) 09:00", "2023/06/10 (Sat) 09:00")
    for text, ts in zip(list(script), stamps, strict=True):
        m.add([_message(text, ts=ts)], user_id="u")
    (active,) = _active(m, "user|lives in")
    assert active.object == "Denver" and m.conflict_stats["superseded"] == 2
    assert m.conflict_stats["candidates"] == 2, "Boston was inactive when Denver arrived"


def test_negation_across_sessions_supersedes_and_the_round_is_untouched(tmp_path):
    script = {
        "I like jazz.": [_fact("The user likes jazz.", "user: I like jazz.", "user", "likes",
                               "jazz")],
        "I dislike jazz now.": [_fact("The user dislikes jazz.", "user: I dislike jazz now.",
                                      "user", "dislikes", "jazz")],
    }
    m = _scripted_memory(tmp_path / "n.db", script)
    _two_sessions(m, list(script))
    (active,) = _active(m, "user|like")
    assert active.object == "jazz" and active.predicate == "dislikes"
    assert m.conflict_stats["conflicts_negation"] == 1
    rounds = [r for r, _ in m.store.search(m._query_embedding("jazz"), "u", k=5, kind="round")]
    assert len(rounds) == 2 and all(r.salience == 1.0 and r.supersedes is None for r in rounds), \
        "rounds are never superseded (D1)"


def test_read_path_is_untouched_a_superseded_fact_still_renders(tmp_path):
    m = _scripted_memory(tmp_path / "r.db", _VALUE_SCRIPT, active_only=False)
    _two_sessions(m, list(_VALUE_SCRIPT))
    context = m.get_context("where does the user live", "u")
    assert "The user lives in Boston." in context and "The user lives in Seattle." in context
    assert len(m.recall("where does the user live", "u")) == 2


def test_consolidate_is_idempotent_and_matches_add(tmp_path):
    m = _scripted_memory(tmp_path / "c1.db", _VALUE_SCRIPT)
    _two_sessions(m, list(_VALUE_SCRIPT))
    before = [(r.id, r.salience, r.supersedes) for r in m.store.facts_with_pair_key("u")]
    m.consolidate("u")
    m.consolidate("u")
    assert [(r.id, r.salience, r.supersedes) for r in m.store.facts_with_pair_key("u")] == before
    assert m.conflict_stats["superseded"] == 1, "consolidate found nothing to do"
    # A store whose facts were inserted with resolution off, then consolidated
    # with it on, reaches the same active set (its screens-off merges aside).
    off = _scripted_memory(tmp_path / "c2.db", _VALUE_SCRIPT, conflict_resolution=False)
    _two_sessions(off, list(_VALUE_SCRIPT))
    on = Memory(str(tmp_path / "c2.db"), HashingEmbedder(),
                MemoryConfig(dedup_cosine_threshold=0.5), extractor=off.extractor)
    on.consolidate("u")
    # Different sessions, so the session-scoped cosine gate never merged the
    # pair: both facts were stored, and the pass leaves the June one active.
    assert [r.object for r in _active(on, "user|lives in")] == ["Seattle"]
    assert on.conflict_stats["superseded"] == 1
    off2 = _scripted_memory(tmp_path / "c3.db", _VALUE_SCRIPT, conflict_resolution=False)
    _two_sessions(off2, list(_VALUE_SCRIPT))
    off2.consolidate("u")
    facts = off2.store.facts_with_pair_key("u")
    assert all(r.supersedes is None and r.salience > 0 for r in facts), (
        "off means no supersession; decay is its own pass (PHASE4 D3)")


def test_cosine_negation_pair_supersedes_only_without_a_pair_key(tmp_path):
    # Null triples: the pair cannot meet on the index, so the negation screen's
    # keep is the only path — and it supersedes (D7).
    script = {
        "Quick note: I live in Boston now.": [
            _fact("The user lives in Boston.", "user: I live in Boston now.")],
        "Update from me: I no longer live in Boston.": [
            _fact("The user no longer lives in Boston.", "user: I no longer live in Boston.")],
    }
    m = _scripted_memory(tmp_path / "n1.db", script)
    for text in script:
        m.add([_message(text, ts="2023-05-20")], user_id="u")
    assert m.conflict_stats["kept_negation"] == 1 and m.conflict_stats["superseded"] == 1
    # Keyed triples on DIFFERENT pairs (and the assistant's facts, never): the
    # screen keeps them apart, but nothing supersedes — conflict_between rules.
    keyed = {
        "Quick note: I like the new office.": [
            _fact("The assistant explained the office.", "assistant: the new office",
                  "assistant", "explained", "the office")],
        "Update from me: I don't like the new office.": [
            _fact("The assistant did not explain the office.", "assistant: not the office",
                  "assistant", "did not explain", "the office")],
    }
    k = _scripted_memory(tmp_path / "n2.db", keyed)
    for text in keyed:
        k.add([_message(text, ts="2023-05-20")], user_id="u")
    assert k.conflict_stats["kept_negation"] == 1 and k.conflict_stats["superseded"] == 0
    # The assistant's facts never conflict (D2), even through the null-triple
    # path: one side keyed to the assistant, the other with no triple.
    mixed = {
        "Quick note: I read the book.": [
            _fact("The assistant read the book.", "assistant: yes", "assistant", "read",
                  "the book")],
        "Update from me: I did not read the book.": [
            _fact("The assistant did not read the book.", "assistant: no")],
    }
    a = _scripted_memory(tmp_path / "n3.db", mixed)
    for text in mixed:
        a.add([_message(text, ts="2023-05-20")], user_id="u")
    assert a.conflict_stats["kept_negation"] == 1 and a.conflict_stats["superseded"] == 0


# -- Phase 4 Task 3: decay inside consolidate() ----------------------------------------------


def test_config_has_the_decay_fields_with_spec_defaults_and_validates_them(tmp_path):
    config = MemoryConfig()
    assert config.decay_half_life_days == 30.0 and config.decay_floor == 0.15
    for bad in ({"decay_half_life_days": 0.0}, {"decay_floor": 0.0}, {"decay_floor": 1.5}):
        with pytest.raises(ValueError, match="decay"):
            Memory(str(tmp_path / "bad.db"), HashingEmbedder(), MemoryConfig(**bad))


_OLD, _NEW, _LATER = "2023/05/01 (Mon) 09:00", "2023/06/30 (Fri) 09:00", "2023/08/29 (Tue) 09:00"


def _saliences(m):
    return {r.content.split("] ", 1)[1]: (r.salience, r.initial_salience)
            for r in m.store.active_records("u") if r.kind == "round"}


def _aged_memory(tmp_path, **config):
    m = Memory(str(tmp_path / "aged.db"), HashingEmbedder(), MemoryConfig(**config))
    m.add([_message("I planted tomatoes in the community garden", ts=_OLD)], user_id="u")
    m.add([_message("my violin lesson moved to thursdays", ts=_NEW)], user_id="u")
    return m


def test_consolidate_decays_every_active_record_by_logical_age(tmp_path):
    m = _aged_memory(tmp_path)
    m.consolidate("u")
    assert _saliences(m) == {
        "I planted tomatoes in the community garden": (0.25, 1.0),  # 60 days, two half-lives
        "my violin lesson moved to thursdays": (1.0, 1.0),  # the session now_logical names
    }
    assert m.decay_stats == {"passes": 1, "decayed": 1, "at_floor": 0}


def test_consolidate_decays_facts_from_their_extracted_salience(tmp_path):
    script = {"Which trellis should I buy for my tomatoes?": [
        _fact("The assistant recommended a cedar trellis.", "assistant: get a cedar trellis",
              "assistant", "recommended", "cedar trellis", salience=0.5)]}
    m = _scripted_memory(tmp_path / "f.db", script)
    m.add([_message("Which trellis should I buy for my tomatoes?", ts="2023/05/31 (Wed) 09:00"),
           _message("Get a cedar trellis.", role="assistant", ts="2023/05/31 (Wed) 09:00")],
          user_id="u")
    m.add([_message("my violin lesson moved to thursdays", ts=_NEW)], user_id="u")
    m.consolidate("u")
    (fact,) = [r for r in m.store.active_records("u") if r.kind == "fact"]
    assert fact.initial_salience == 0.5 and fact.salience == pytest.approx(0.25)


def test_decay_clamps_at_the_floor_and_never_touches_a_superseded_fact(tmp_path):
    m = _scripted_memory(tmp_path / "s.db", _VALUE_SCRIPT)
    _two_sessions(m, list(_VALUE_SCRIPT))  # January Boston, superseded by June Seattle
    m.consolidate("u")
    by_object = {r.object: r for r in m.store.facts_with_pair_key("u")}
    assert by_object["Boston"].salience == 0.0, "salience 0 is supersession's, never decayed"
    assert by_object["Seattle"].salience == 1.0
    (old_round,) = [r for r in m.store.active_records("u")
                    if r.kind == "round" and r.created_at.startswith("2023/01")]
    assert old_round.salience == 0.15 and m.decay_stats["at_floor"] == 1


def test_consolidate_twice_is_consolidate_once_exactly(tmp_path):
    m = _aged_memory(tmp_path)
    m.consolidate("u")
    first = [(r.id, r.salience) for r in m.store.active_records("u")]
    m.consolidate("u")
    assert [(r.id, r.salience) for r in m.store.active_records("u")] == first
    assert m.decay_stats["decayed"] == 1, "the second pass moved nothing"


def test_an_access_restores_the_initial_salience_at_the_next_pass(tmp_path):
    m = _aged_memory(tmp_path)
    m.consolidate("u")
    old = [r for r in m.store.active_records("u") if r.created_at == _OLD]
    m.store.touch([old[0].id], _NEW)
    m.consolidate("u")
    assert _saliences(m)["I planted tomatoes in the community garden"] == (1.0, 1.0)


def test_a_later_session_moves_now_logical(tmp_path):
    m = _aged_memory(tmp_path)
    m.consolidate("u")
    m.add([_message("booked a ferry to the islands", ts=_LATER)], user_id="u")
    m.consolidate("u")
    assert _saliences(m) == {
        "I planted tomatoes in the community garden": (0.15, 1.0),  # 120 days: 0.0625 clamps
        "my violin lesson moved to thursdays": (0.25, 1.0),
        "booked a ferry to the islands": (1.0, 1.0),
    }


def test_decay_logs_one_line_per_moved_record(tmp_path, caplog):
    import logging

    m = _aged_memory(tmp_path)
    with caplog.at_level(logging.INFO, logger="mnimi"):
        m.consolidate("u")
    (old,) = [r for r in m.store.active_records("u") if r.created_at == _OLD]
    assert f"decayed {old.id}: 60 days since last access, salience 1.0000 -> 0.2500" in caplog.text
    assert caplog.text.count("decayed ") == 1


def test_decay_runs_whatever_conflict_resolution_says(tmp_path):
    m = _aged_memory(tmp_path, conflict_resolution=False)
    m.consolidate("u")
    assert _saliences(m)["I planted tomatoes in the community garden"] == (0.25, 1.0)


# -- Phase 4 Task 4: the ranking layer, ScoredRecord, recall() --------------------------------


def test_config_has_the_ranking_fields_and_validates_them(tmp_path):
    config = MemoryConfig()
    assert config.ranking == "similarity"
    assert dict(config.salience_weights) == {"similarity": 1.0, "recency": 0.0}
    for bad in ({"ranking": "bm25"}, {"salience_weights": {"similarity": 1.0}}):
        with pytest.raises(ValueError, match="ranking|salience_weights"):
            Memory(str(tmp_path / "bad.db"), HashingEmbedder(), MemoryConfig(**bad))


def test_recall_returns_scored_records_in_the_v110_order(tmp_path):
    from mnimi import ScoredRecord

    m = _aged_memory(tmp_path)
    hits = m.recall("tomatoes in the garden", "u")
    assert all(isinstance(hit, ScoredRecord) for hit in hits)
    expected = m.store.search_rounds(m._query_embedding("tomatoes in the garden"), "u", k=10)
    assert [(hit.record.id, hit.relevance) for hit in hits] == [(r.id, c) for r, c in expected]
    assert all(hit.score == hit.relevance for hit in hits), "similarity: score is relevance"
    by_ts = {hit.record.created_at: hit for hit in hits}
    assert by_ts[_NEW].recency == 1.0 and by_ts[_OLD].recency == pytest.approx(0.25)
    assert all(hit.salience == hit.record.salience == 1.0 for hit in hits)


def test_score_ranking_reads_the_stored_salience(tmp_path):
    # January Boston is superseded by June Seattle: under "similarity" the
    # superseded fact still represents its round; under "score" it scores 0 and
    # the round is represented by its own record.
    ranked = {}
    for ranking in ("similarity", "score"):
        m = _scripted_memory(tmp_path / f"{ranking}.db", _VALUE_SCRIPT, ranking=ranking,
                             active_only=False)
        _two_sessions(m, list(_VALUE_SCRIPT))
        ranked[ranking] = {hit.record.created_at: hit for hit in m.recall("Boston", "u")}
    january = "2023/01/10 (Tue) 09:00"
    assert ranked["similarity"][january].record.kind == "fact"
    assert ranked["similarity"][january].record.salience == 0.0
    assert ranked["score"][january].record.kind == "round"
    assert ranked["score"][january].score == ranked["score"][january].relevance > 0


# -- Phase 4 Task 5: the retriever extras -------------------------------------------------


def test_config_has_the_retriever_extras_and_validates_them(tmp_path):
    config = MemoryConfig()
    assert config.active_only is False and config.recall_min_relevance == 0.0
    with pytest.raises(ValueError, match="recall_min_relevance"):
        Memory(str(tmp_path / "bad.db"), HashingEmbedder(), MemoryConfig(recall_min_relevance=1.5))


def test_active_only_hides_a_superseded_fact_from_ranking_and_rendering(tmp_path):
    for ranking in ("similarity", "score"):
        m = _scripted_memory(tmp_path / f"a-{ranking}.db", _VALUE_SCRIPT, active_only=True,
                             ranking=ranking)
        _two_sessions(m, list(_VALUE_SCRIPT))
        hits = m.recall("Boston", "u")
        assert all(hit.record.salience > 0 for hit in hits), ranking
        assert {hit.record.created_at for hit in hits} == {"2023/01/10 (Tue) 09:00",
                                                           "2023/06/10 (Sat) 09:00"}, ranking
        context = m.get_context("where does the user live", "u")
        assert "The user lives in Boston." not in context, ranking
        assert "The user lives in Seattle." in context, ranking
        assert "user: Quick note: I live in Boston now." in context, "the round is evidence (D1)"


def test_recall_min_relevance_zero_is_off_by_definition(tmp_path):
    from mnimi import ScoredRecord

    m = _aged_memory(tmp_path)
    (record,) = [r for r in m.store.active_records("u") if r.created_at == _OLD]
    negative = ScoredRecord(record=record, relevance=-0.2, recency=1.0, salience=1.0, score=-0.2)
    m._rank = lambda *args: [negative]
    assert m.recall("anything", "u") == [negative], "0.0 skips the drop; it is not 'cosine >= 0'"
    assert record.last_accessed == _NEW, "and the returned record was written back"


def test_recall_min_relevance_drops_below_the_floor_and_nothing_fills_in(tmp_path):
    m = _aged_memory(tmp_path)
    relevances = sorted(hit.relevance for hit in m.recall("tomatoes in the garden", "u"))
    assert len(relevances) == 2 and relevances[0] < relevances[1]
    floor = (relevances[0] + relevances[1]) / 2
    floored = Memory(str(tmp_path / "aged.db"), HashingEmbedder(),
                     MemoryConfig(recall_min_relevance=floor))
    assert [hit.relevance for hit in floored.recall("tomatoes in the garden", "u")] == [
        relevances[1]]


def test_recall_writes_last_accessed_on_the_returned_records_only(tmp_path):
    m = _aged_memory(tmp_path, top_k=1)
    mid = "2023/05/31 (Wed) 09:00"
    m.add([_message("the dentist appointment is on friday", ts=mid)], user_id="u")
    m.consolidate("u")
    (hit,) = m.recall("planted tomatoes in the community garden", "u")
    assert hit.record.created_at == _OLD and hit.record.last_accessed == _NEW
    rows = {r.created_at: r for r in m.store.active_records("u")}
    assert rows[_OLD].last_accessed == _NEW, "written back: now_logical"
    assert rows[mid].last_accessed == mid, "not returned, not touched"
    m.consolidate("u")
    rows = {r.created_at: r for r in m.store.active_records("u")}
    assert rows[_OLD].salience == 1.0, "the access restored it (D2, D3)"
    assert rows[mid].salience == pytest.approx(0.5 ** (30 / 30))
