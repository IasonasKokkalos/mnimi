"""The extraction package: import discipline, schema, prompt, fake, cache, prefilter, resolver."""

from __future__ import annotations

import subprocess
import sys


def test_importing_mnimi_extract_never_imports_llama_cpp():
    # The [extract] extra is optional: `import mnimi` and `import mnimi.extract`
    # must work with the two core deps alone, and CI installs [dev] only.
    code = (
        "import sys, mnimi, mnimi.extract; "
        "assert 'llama_cpp' not in sys.modules, 'llama_cpp imported eagerly'; "
        "assert 'huggingface_hub' not in sys.modules, 'huggingface_hub imported eagerly'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- Task 3: protocol, schema + grammar, prompt, the model-free RuleExtractor ---

from mnimi.extract import fake, prompt, protocol, schema  # noqa: E402


def test_schema_field_order_is_prose_first_and_locked():
    # CHANGELOG #12: free prose (content, raw) before the constrained slots;
    # `when` (a verbatim time mention) sits between them (PHASE2 D8).
    assert schema.field_order() == [
        "content", "raw", "when", "subject", "predicate", "object", "salience"
    ]
    props = schema.FACT_SCHEMA["items"]["properties"]
    assert list(props) == schema.field_order()
    assert schema.FACT_SCHEMA["items"]["required"] == schema.field_order()
    assert props["salience"]["enum"] == [0.25, 0.5, 1.0]
    assert schema.FACT_SCHEMA["maxItems"] == 8
    assert props["content"]["maxLength"] == 300 and props["raw"]["maxLength"] == 300


def test_parse_output_treats_truncation_as_empty_and_flags_it():
    ok, truncated = schema.parse_output(
        '[{"content":"The user owns a cat.","raw":"user: I have a cat","when":null,'
        '"subject":"user","predicate":"owns","object":"cat","salience":1.0}]'
    )
    assert not truncated
    assert ok[0].content == "The user owns a cat." and ok[0].object == "cat"
    assert ok[0].when is None and ok[0].salience == 1.0
    facts, truncated = schema.parse_output('[{"content":"The user owns a cat.","raw":"user: I')
    assert facts == [] and truncated
    assert schema.parse_output("[]") == ([], False)
    # Structurally valid JSON that is not the contract is a bad parse, not a crash.
    assert schema.parse_output('{"content": "x"}') == ([], True)
    # The runtime's grammar lets a verbatim excerpt carry a real newline inside
    # the string (measured 2026-09-13 on two of fifty rounds); the parser must
    # read what the grammar admits.
    with_newline = (
        '[{"content":"The assistant listed luaus.","raw":"assistant: 1. **A**\n2. **B**",'
        '"when":null,"subject":"assistant","predicate":"listed","object":"luaus",'
        '"salience":0.5}]'
    )
    facts, truncated = schema.parse_output(with_newline)
    assert not truncated and facts[0].raw == "assistant: 1. **A**\n2. **B**"


def test_parse_output_treats_an_unpaired_surrogate_as_truncation():
    # Measured on the LongMemEval corpus (1 round in 60,467, 2026-09-20): the
    # model ends a string mid-emoji, emitting the HIGH half of a surrogate pair
    # with no low half. json.loads materialises it faithfully, but the result is
    # not valid Unicode -- it cannot be UTF-8 encoded, and the embedder's Rust
    # tokenizer rejects the whole batch with "TextEncodeInput must be ...",
    # which crashed an 18-hour corpus pass. It is the same evidence of a cut-off
    # output that a truncated array is, so it takes the same path.
    cut = (
        '[{"content":"The assistant explained regression.",'
        '"raw":"assistant: between the input images and the output FA and\\uD83C",'
        '"when":null,"subject":"assistant","predicate":"explained",'
        '"object":"linear regression","salience":0.5}]'
    )
    assert schema.parse_output(cut) == ([], True)

    # A COMPLETE pair is ordinary text and must survive untouched -- the check
    # rejects malformed Unicode, not emoji.
    whole = (
        '[{"content":"The user won a trophy \\uD83C\\uDFC6.",'
        '"raw":"user: I won a trophy \\uD83C\\uDFC6","when":null,'
        '"subject":"user","predicate":"won","object":"trophy","salience":1.0}]'
    )
    facts, truncated = schema.parse_output(whole)
    assert not truncated
    assert facts[0].content == "The user won a trophy \U0001F3C6."
    facts[0].raw.encode("utf-8")          # encodable, so the tokenizer accepts it

    # An unpaired surrogate anywhere in the item is the same verdict.
    in_object = (
        '[{"content":"The user likes it.","raw":"user: I like it","when":null,'
        '"subject":"user","predicate":"likes","object":"\\uDC00","salience":1.0}]'
    )
    assert schema.parse_output(in_object) == ([], True)


def test_prompt_is_a_literal_with_thinking_disabled_and_a_stable_hash():
    text = prompt.build_prompt([{"role": "user", "content": "I moved to Athens."},
                                {"role": "assistant", "content": "Noted."}])
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "user: I moved to Athens.\nassistant: Noted." in text
    assert text.startswith("<|im_start|>system\n")
    assert prompt.extractor_prompt_hash("root ::= x") == prompt.extractor_prompt_hash("root ::= x")
    assert prompt.extractor_prompt_hash("root ::= x") != prompt.extractor_prompt_hash("root ::= y")
    assert prompt.INPUT_CAP_TOKENS == 1536


def test_rule_extractor_is_deterministic_and_declares_fake_pins():
    turns = [{"role": "user", "content": "Yesterday I adopted a cat named Miso. Any tips?"},
             {"role": "assistant", "content": "Congratulations!"}]
    a, b = fake.RuleExtractor().extract(turns), fake.RuleExtractor().extract(turns)
    assert a == b and len(a.facts) == 1 and a.truncated is False
    assert a.facts[0].raw.startswith("user: ") and a.facts[0].when == "Yesterday"
    assert a.facts[0].content.startswith("The user ")
    pins = fake.RuleExtractor().pins
    assert pins["extractor_model"] == "fake-rule"
    assert set(pins) == {
        "extractor_model", "extractor_quant", "extractor_runtime",
        "extractor_decode_hash", "extractor_prompt_hash",
    }
    assert protocol.extractor_pins_hash(pins) == protocol.extractor_pins_hash(dict(pins))
    assert isinstance(fake.RuleExtractor(), protocol.Extractor)


def test_counting_extractor_counts_and_delegates():
    inner = fake.CountingExtractor(fake.RuleExtractor())
    turns = [{"role": "user", "content": "I run every morning."}]
    assert inner.extract(turns) == fake.RuleExtractor().extract(turns)
    inner.extract(turns)
    assert inner.calls == 2 and inner.pins == fake.RuleExtractor().pins


# --- Task 4: the extraction cache ---


def test_cache_makes_the_second_pass_free_and_keeps_pins_transparent(tmp_path):
    from mnimi.extract.cache import CachedExtractor

    inner = fake.CountingExtractor(fake.RuleExtractor())
    cached = CachedExtractor(inner, tmp_path / "x.sqlite")
    turns = [{"role": "user", "content": "I run every morning."}]
    first = cached.extract(turns)
    reopened = CachedExtractor(inner, tmp_path / "x.sqlite")  # a later process
    second = reopened.extract(turns)
    assert first == second and inner.calls == 1
    assert reopened.stats == {"hits": 1, "misses": 0}
    assert cached.stats == {"hits": 0, "misses": 1}
    assert cached.pins == fake.RuleExtractor().pins
    assert isinstance(cached, protocol.Extractor)


def test_cache_key_ignores_ts_and_respects_roles():
    from mnimi.extract.cache import CachedExtractor

    key = CachedExtractor.key
    assert key([{"role": "user", "content": "x", "ts": "2023-01-01"}]) == key(
        [{"role": "user", "content": "x"}]
    )
    assert key([{"role": "user", "content": "x"}]) != key([{"role": "assistant", "content": "x"}])
    assert key([{"role": "user", "content": "x"}]) != key([{"role": "user", "content": "y"}])


def test_cache_path_is_one_file_per_extractor_configuration(tmp_path, monkeypatch):
    from mnimi.extract.cache import default_cache_path

    monkeypatch.chdir(tmp_path)
    a = default_cache_path(fake.RuleExtractor().pins)
    b = default_cache_path({**fake.RuleExtractor().pins, "extractor_prompt_hash": "other"})
    assert a != b and a.parent == tmp_path / ".cache" / "extract" and a.suffix == ".sqlite"


# --- Task 5: the stage-1 pre-filter ---

from mnimi.extract import prefilter  # noqa: E402


def test_prefilter_keeps_question_with_aside_and_drops_only_slotless_turns():
    aside = {
        "role": "user",
        "content": "Can you suggest some tips? By the way, I've been doing guided breathing "
        "sessions with my Fitbit.",
    }
    assert prefilter.decide(aside) is None, "a question with an aside is evidence, never dropped"
    assert prefilter.decide({"role": "user", "content": "Thanks, that helps!"}) == "ack-only"
    assert prefilter.decide({"role": "user", "content": "Hi there!"}) == "greeting-only"
    assert (
        prefilter.decide(
            {
                "role": "assistant",
                "content": "You're welcome! Let me know if there's anything else I can help with.",
            }
        )
        == "assistant-boilerplate"
    )
    assert prefilter.decide({"role": "user", "content": "Remember that."}) in {
        "memory-self-reference",
        "imperative-to-assistant",
    }
    # Questions are never slotless: a bare question costs one model call at most.
    question = {"role": "user", "content": "Can you explain how transformers work?"}
    assert prefilter.decide(question) is None
    assert prefilter.decide({"role": "user", "content": "so what do you think"}) is None
    assert prefilter.decide({"role": "user", "content": "ok cool"}) == "ack-only"
    slotless = {"role": "user", "content": "not really sure about any of those"}
    assert prefilter.decide(slotless) == "no-candidate-slot"
    imperative = {"role": "user", "content": "Explain that again."}
    assert prefilter.decide(imperative) == "imperative-to-assistant"
    assert prefilter.decide({"role": "user", "content": "I moved to Athens."}) is None
    assert prefilter.decide({"role": "user", "content": "Miso is my cat."}) is None
    # A long assistant answer is never boilerplate, whatever it opens with.
    long_answer = "You're welcome! " + "Here is a detailed plan with many specifics. " * 20
    assert prefilter.decide({"role": "assistant", "content": long_answer}) is None
    keep, dropped = prefilter.keep_round(
        [{"role": "user", "content": "Thanks!"}, {"role": "assistant", "content": "Anytime!"}]
    )
    assert keep is False and len(dropped) == 2
    keep, dropped = prefilter.keep_round([aside, {"role": "assistant", "content": "Anytime!"}])
    assert keep is True and dropped == ["assistant-boilerplate"]


def test_prefilter_hash_moves_with_the_lexicon(monkeypatch):
    before = prefilter.prefilter_lexicon_hash()
    monkeypatch.setattr(
        prefilter, "LEXICON", {**prefilter.LEXICON, "ack": (*prefilter.LEXICON["ack"], "ta")}
    )
    assert prefilter.prefilter_lexicon_hash() != before
    assert len(before) == 64


def test_prefilter_logs_the_rule(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="mnimi.extract"):
        prefilter.keep_round([{"role": "user", "content": "Thanks!"}])
    assert "filtered: ack-only" in caplog.text


# --- Task 6: the relative-date resolver ---

import pytest  # noqa: E402


@pytest.mark.parametrize("mention,ts,expected", [
    ("yesterday", "2022/11/18 (Fri) 17:03", "2022-11-17"),
    ("Yesterday", "2022/11/18 (Fri) 17:03", "2022-11-17"),
    ("today", "2023/02/14 (Tue) 09:06", "2023-02-14"),
    ("two weeks ago", "2023/05/20 (Sat) 20:05", "2023-05-06"),
    ("3 days ago", "2023/05/20 (Sat) 20:05", "2023-05-17"),
    ("last month", "2023/05/25 (Thu) 16:59", "2023-04"),
    ("last year", "2023/05/25 (Thu) 16:59", "2022"),
    ("the week before last", "2023/05/29 (Mon) 15:17", "2023-05-15"),
    ("last week", "2023/05/26 (Fri) 14:05", "2023-05-19"),
    ("last Saturday, exactly a week ago", "2023/05/26 (Fri) 14:05", "2023-05-20"),
    ("next Tuesday", "2023/05/26 (Fri) 14:05", "2023-05-30"),
    ("on October 15th", "2023/11/29 (Wed) 03:49", "2023-10-15"),
    ("on September 25th", "2023/11/29 (Wed) 05:08", "2023-09-25"),
    ("in June", "2023/11/29 (Wed) 22:12", "2023-06"),
    ("in December", "2023/11/29 (Wed) 22:12", "2022-12"),
    ("in November", "2023/11/29 (Wed) 07:51", "2023-11"),
    ("on 2/5", "2023/02/15 (Wed) 11:13", "2023-02-05"),
    ("February 5th", "2023/02/15 (Wed) 11:13", "2023-02-05"),
    ("on 2023-05-14", "2023/05/20 (Sat) 20:05", "2023-05-14"),
    ("in 3 weeks", "2023/05/20 (Sat) 20:05", "2023-06-10"),
    ("three months ago", "2023/05/21 (Sun) 22:01", "2023-02-21"),
    ("recently", "2023/06/15 (Thu) 10:02", None),
    ("a few days ago", "2023/06/15 (Thu) 10:02", None),
    ("last month", "not a timestamp", None),
    (None, "2023-05-20", None),
    ("", "2023-05-20", None),
])
def test_resolver_anchors_on_ts(mention, ts, expected):
    from mnimi.extract.resolver import resolve

    assert resolve(mention, ts) == expected


def test_resolver_never_reads_the_clock():
    import inspect

    from mnimi.extract import resolver

    src = inspect.getsource(resolver)
    for forbidden in ("datetime.now", "date.today", "time.time", "utcnow"):
        assert forbidden not in src
    assert resolver.RESOLVER_VERSION == "v3"


@pytest.mark.parametrize(
    "mention, ts, context, expected",
    [
        # PHASE6-ANALYSIS §6: the eight future-marked facts v1 dated a year back.
        ("in October", "2023/05/25 (Thu) 10:00",
         "The user is planning a birthday trip to Hawaii in October.", "2023-10"),
        ("in July", "2023/05/24 (Wed) 10:00",
         "The user is planning to travel to Seoul in July.", "2023-07"),
        ("in November", "2023/04/29 (Sat) 10:00",
         "The user is planning a 10-day trek in New Zealand in November.", "2023-11"),
        ("in July", "2023/05/17 (Wed) 10:00",
         "The user's sister's birthday is coming up in July.", "2023-07"),
        ("in June", "2023/02/20 (Mon) 10:00",
         "The user is planning a road trip to Chicago in June for a work conference.", "2023-06"),
        # The marker may sit in the raw span the fact came from ("tomorrow"), not
        # in the fact's own sentence: memory.round_pieces passes both.
        ("on April 24th", "2023/04/23 (Sun) 02:44",
         "The user is preparing for an open track day at VIR on April 24th. "
         "user: I'm preparing for an open track day at VIRginia International Raceway tomorrow",
         "2023-04-24"),
        # ... and without any marker the v1 rule stands, even for a plan (the list is frozen).
        ("on April 24th", "2023/04/23 (Sun) 02:44",
         "The user is preparing for an open track day at VIR on April 24th.", "2022-04-24"),
        ("on May 15th", "2023/04/23 (Sun) 02:44",
         "The user plans to participate in the 'Track Day Frenzy' event at VIR on May 15th.",
         "2023-05-15"),
        ("next Saturday", "2023/05/24 (Wed) 10:00", "The user will race next Saturday.",
         "2023-05-27"),
        # A past marker in the mention keeps the v1 rule whatever the fact says.
        ("back in April", "2022/02/19 (Sat) 10:00",
         "The user mentioned a charity yoga event they did back in April.", "2021-04"),
        ("last April", "2023/05/24 (Wed) 10:00", "The user will plan next year like last April.",
         "2023-04"),
        # No marker anywhere: v1 unchanged (a past event that already happened).
        ("on March 15-16", "2023/02/26 (Sun) 13:37",
         "The user attended a digital marketing workshop on March 15-16.", "2022-03-15"),
        ("on 2/8", "2023/03/03 (Fri) 15:10",
         "The user took their niece to the Natural History Museum on 2/8.", "2023-02-08"),
        # "this Friday" under a future marker: the first on or after ts.
        ("this Friday", "2023/05/24 (Wed) 10:00", "The user is going to the dentist this Friday.",
         "2023-05-26"),
        # v2's one new form: last weekend = the most recent Saturday strictly before ts.
        ("last weekend", "2023/01/22 (Sun) 12:52", "The user watched the game last weekend.",
         "2023-01-21"),
        ("last weekend", "2023/01/21 (Sat) 12:52", None, "2023-01-14"),
        ("this past weekend", "2023/01/24 (Tue) 12:52", None, "2023-01-21"),
    ],
)
def test_resolver_v2_resolves_future_marked_mentions_forward(mention, ts, context, expected):
    from mnimi.extract.resolver import resolve

    assert resolve(mention, ts, context=context) == expected


def test_resolver_v2_markers_are_frozen():
    from mnimi.extract import resolver

    assert resolver.FUTURE_MARKERS == (
        "next", "upcoming", "tomorrow", "planning", "plan to", "plans to", "will",
        "coming", "coming up", "going to",
    )
    assert resolver.PAST_MARKERS == (
        "last", "ago", "yesterday", "back in", "this past", "past", "earlier", "previous",
    )


def test_verbatim_mention_drops_an_invented_time():
    from mnimi.extract.resolver import verbatim_mention

    text = "Thanks! I think I'll start sharing my Lightroom edits.  By the way, I moved last week."
    assert verbatim_mention("last week", text) == "last week"
    assert verbatim_mention("Last  Week", text) == "Last  Week", "case and spacing are forgiven"
    assert verbatim_mention("next week", text) is None, "echoed from an example, not the round"
    assert verbatim_mention(None, text) is None and verbatim_mention("", text) is None


def test_round_pieces_drop_an_invented_when_but_keep_the_fact(tmp_path):
    from mnimi import MemoryConfig
    from mnimi.embeddings import HashingEmbedder
    from mnimi.extract.protocol import ExtractedFact, ExtractionResult
    from mnimi.memory import _messages_to_rounds, round_pieces

    class Inventor:
        pins = fake.RuleExtractor().pins

        def extract(self, turns):
            return ExtractionResult(
                facts=[
                    ExtractedFact("The user moved to Athens.", "user: I moved to Athens",
                                  "next week", "user", "moved to", "Athens", 1.0),
                    ExtractedFact("The user adopted a cat.", "user: I adopted a cat yesterday",
                                  "yesterday", "user", "adopted", "cat", 1.0),
                ],
                truncated=False, raw_output="[]",
            )

    (round_,) = _messages_to_rounds([
        {"role": "user", "content": "I moved to Athens. I adopted a cat yesterday.",
         "ts": "2023-05-20"},
    ])
    pieces = round_pieces(round_, HashingEmbedder(), MemoryConfig(), Inventor())
    facts = [p for p in pieces if p.kind == "fact"]
    assert [(p.time_mention, p.valid_time) for p in facts] == [
        (None, None), ("yesterday", "2023-05-19")
    ]


def test_scripted_extractor_returns_the_authored_facts_per_round():
    from mnimi.extract.fake import ScriptedExtractor
    from mnimi.extract.protocol import ExtractedFact

    fact = ExtractedFact("The user lives in Boston.", "user: I live in Boston.", None, "user",
                         "lives in", "Boston", 1.0)
    ex = ScriptedExtractor({"I live in Boston.": [fact]})
    hit = ex.extract([{"role": "user", "content": "I live in Boston."},
                      {"role": "assistant", "content": "Nice."}])
    assert hit.facts == [fact] and hit.truncated is False
    assert ex.extract([{"role": "user", "content": "unscripted"}]).facts == []
    assert ex.pins["extractor_model"] == "fake-scripted" and set(ex.pins) == set(protocol.PIN_KEYS)


# --- TESTFIXESPT1 Task 1: the post-hoc raw/role verification ---


def test_verify_raw_keeps_a_verbatim_excerpt_and_returns_the_turns_bytes():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user", "content": "I moved to Athens.  I adopted a cat yesterday."},
             {"role": "assistant", "content": "Noted."}]
    v = verify_raw("The user adopted a cat.", "user: i adopted a  cat yesterday", turns)
    assert v.how == "verbatim" and v.role == "user"
    assert v.raw == "user: I adopted a cat yesterday", "case and spacing come from the turn"


def test_verify_raw_fixes_a_wrong_role_prefix():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user", "content": "begin with task 0"},
             {"role": "assistant", "content": "E2 started 2026-10-13 evening. Branch paper-runs."}]
    v = verify_raw("E2 started.", "user: E2 started 2026-10-13 evening.", turns)
    assert v.how == "role" and v.role == "assistant"
    assert v.raw == "assistant: E2 started 2026-10-13 evening."


def test_verify_raw_extends_a_cut_excerpt_to_the_end_of_its_sentence():
    from mnimi.extract.verify import verify_raw

    # Issue 4: the grammar rescan after an unescaped backslash cut the excerpt and closed it.
    text = ("E2 started 2026-10-13 evening. Worktree D:\\Projects\\Personal\\mnimi-paper "
            "(paper-v1 = 21239e2), branch paper-runs. Rule commit e184ed9.")
    turns = [{"role": "user", "content": "begin"}, {"role": "assistant", "content": text}]
    v = verify_raw("Worktree D:\\\\", "user: E2 started 2026-10-13 evening. Worktree D:\\\\", turns)
    assert v.how == "prefix" and v.role == "assistant"
    assert v.raw == ("assistant: E2 started 2026-10-13 evening. Worktree D:\\Projects\\Personal\\"
                     "mnimi-paper (paper-v1 = 21239e2), branch paper-runs.")
    # Issue 2: the 300-cap squeezed the tail into CJK; the prefix walk restores the sentence.
    text2 = "The finding in two sentences (44 of 69 judge-stable misses are retrieval-bound), done."
    turns2 = [{"role": "assistant", "content": text2}]
    cut = "assistant: the finding in two sentences (44 of 69 judge-stable misses are检索-1"
    v2 = verify_raw("x", cut, turns2)
    assert v2.how == "prefix" and v2.raw == "assistant: " + text2


def test_verify_raw_treats_slash_and_backslash_as_one_character():
    from mnimi.extract.verify import verify_raw

    # The pre-escape wrapper shows the model "/" where the turn has "\"; the excerpt still
    # verifies, and the returned bytes are the turn's (backslashes restored).
    turns = [{"role": "assistant", "content": "Worktree D:\\Projects\\mnimi-paper is ready."}]
    v = verify_raw("x", "assistant: Worktree D:/Projects/mnimi-paper is ready.", turns)
    assert v.how == "verbatim"
    assert v.raw == "assistant: Worktree D:\\Projects\\mnimi-paper is ready."


def test_verify_raw_falls_back_to_the_best_overlapping_sentence():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user", "content": "Hi there. I live in Boston with two cats. Thanks!"},
             {"role": "assistant", "content": "Boston is lovely."}]
    v = verify_raw("The user lives in Boston with two cats.", "user: user lives Boston two cats",
                   turns)
    assert v.how == "overlap" and v.role == "user"
    assert v.raw == "user: I live in Boston with two cats."


def test_verify_raw_empty_body_never_verifies():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user", "content": "I live in Boston."}]
    assert verify_raw("The user lives in Boston.", "user: ", turns).how == "overlap"
    assert verify_raw("The user lives in Boston.", "", turns).how == "overlap"
    kept = verify_raw("Nothing in common here.", "", [{"role": "user", "content": "zzz qqq"}])
    assert kept.how == "kept" and kept.raw == "" and kept.role is None


def test_verify_raw_scans_every_turn_of_a_role():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user", "content": "First message."},
             {"role": "user", "content": "I adopted a cat."},
             {"role": "assistant", "content": "Lovely."}]
    v = verify_raw("The user adopted a cat.", "assistant: I adopted a cat.", turns)
    assert v.how == "role" and v.role == "user" and v.raw == "user: I adopted a cat."


def test_sentences_split_on_terminators_and_newlines():
    from mnimi.extract.verify import sentences

    assert sentences("A one. B two!\nC three? D") == ["A one.", "B two!", "C three?", "D"]
    assert sentences("  \n") == []


# --- TESTFIXESPT1 Task 2: round_pieces stores the verified excerpt; when is per turn ---


def _pieces_for(turn_texts: list[tuple[str, str]], facts: list, ts: str = "2026-10-12T00:00:00"):
    from mnimi import MemoryConfig
    from mnimi.embeddings import HashingEmbedder
    from mnimi.extract.protocol import ExtractionResult
    from mnimi.memory import _messages_to_rounds, new_extraction_stats, round_pieces

    class Fixed:
        pins = fake.RuleExtractor().pins

        def extract(self, turns):
            return ExtractionResult(facts=facts, truncated=False, raw_output="[]")

    (round_,) = _messages_to_rounds(
        [{"role": r, "content": c, "ts": ts} for r, c in turn_texts]
    )
    stats = new_extraction_stats()
    pieces = round_pieces(round_, HashingEmbedder(), MemoryConfig(), Fixed(), stats=stats)
    return [p for p in pieces if p.kind == "fact"], stats


def test_round_pieces_store_the_verified_excerpt_and_count_the_repair():
    from mnimi.extract.protocol import ExtractedFact

    assistant = ("E2 started 2026-10-13 evening. Worktree D:\\Projects\\mnimi-paper "
                 "(paper-v1 = 21239e2), branch paper-runs.")
    facts, stats = _pieces_for(
        [("user", "begin with task 0"), ("assistant", assistant)],
        [ExtractedFact("Worktree D:\\\\", "user: E2 started 2026-10-13 evening. Worktree D:\\\\",
                       None, "E2", "worktree", "D:\\\\", 1.0)],
    )
    (fact,) = facts
    assert fact.raw == "assistant: " + assistant
    assert fact.content == f"{fact.raw}\n{fact.fact}", "the embed text carries the repaired excerpt"
    assert stats["raw_prefix"] == 1 and stats["raw_verbatim"] == 0


def test_round_pieces_verify_when_against_the_facts_own_turn():
    from mnimi.extract.protocol import ExtractedFact

    # Issue 1: the model copied the assistant's "Sunday 2026-10-11" onto a user-turn fact.
    facts, _ = _pieces_for(
        [("user", "Yesterday I meal prepped until thursday, where I will be meal prepping again."),
         ("assistant", "Noted. On Sunday 2026-10-11 you meal prepped until Thursday 2026-10-15.")],
        [ExtractedFact("The user will meal prep again on Sunday 2026-10-11.",
                       "user: where I will be meal prepping again", "Sunday 2026-10-11",
                       "user", "will meal prep again", "Sunday 2026-10-11", 1.0),
         ExtractedFact("The user meal prepped yesterday.", "user: Yesterday I meal prepped",
                       "Yesterday", "user", "meal prepped", None, 1.0)],
    )
    assert [(p.time_mention, p.valid_time) for p in facts] == [
        (None, None), ("Yesterday", "2026-10-11")
    ], "a mention from the other turn is an invention for this fact; the own-turn one resolves"


def test_round_pieces_fall_back_to_the_round_when_the_role_is_unverified():
    from mnimi.extract.protocol import ExtractedFact

    facts, stats = _pieces_for(
        [("user", "zzz qqq yesterday")],
        [ExtractedFact("Nothing in common.", "user: nothing in common", "yesterday",
                       None, None, None, 0.25)],
    )
    (fact,) = facts
    assert stats["raw_kept"] == 1
    assert fact.raw == "user: nothing in common" and fact.valid_time == "2026-10-11"


# --- TESTFIXESPT1 Task 3: the three extractor wrappers ---


class _Recording:
    """An inner extractor that records what it was shown and answers from a script."""

    pins = fake.RuleExtractor().pins

    def __init__(self, answers=None):
        from mnimi.extract.protocol import ExtractionResult
        self.seen: list[list[dict]] = []
        self._answers = list(answers or [])
        self._empty = ExtractionResult(facts=[], truncated=False, raw_output="[]")

    def extract(self, turns):
        self.seen.append(turns)
        return self._answers.pop(0) if self._answers else self._empty


def test_escaping_wrapper_shows_the_model_slashes_and_keeps_pins():
    from mnimi.extract.wrappers import BackslashEscapingExtractor

    inner = _Recording()
    w = BackslashEscapingExtractor(inner)
    w.extract([{"role": "assistant", "content": "Worktree D:\\Projects\\x", "ts": "2026-10-13"}])
    assert inner.seen == [[{"role": "assistant", "content": "Worktree D:/Projects/x"}]]
    assert w.pins == inner.pins


def test_split_text_packs_sentences_under_the_budget_and_keeps_order():
    from mnimi.extract.wrappers import split_text

    text = "One one one. Two two two! Three three?\nFour four."
    assert split_text(text, 0) == [text] and split_text(text, 1000) == [text]
    assert split_text(text, 26) == ["One one one. Two two two!", "Three three? Four four."]


def test_split_text_hard_cuts_an_oversize_sentence():
    from mnimi.extract.wrappers import split_text

    assert split_text("aaaa bbbb cccc dddd", 9) == ["aaaa bbbb", "cccc dddd"]
    assert split_text("abcdefghij", 4) == ["abcd", "efgh", "ij"], "no whitespace: cut at the budget"


def test_splitting_wrapper_is_a_plain_call_when_nothing_exceeds_the_budget():
    from mnimi.extract.wrappers import SplittingExtractor

    inner = _Recording()
    turns = [{"role": "user", "content": "short"}, {"role": "assistant", "content": "also short"}]
    SplittingExtractor(inner, max_chars=600).extract(turns)
    SplittingExtractor(inner, max_chars=0).extract(turns)
    assert inner.seen == [turns, turns] and inner.seen[0] is turns


def test_client_facts_wrapper_returns_the_queue_once_and_fills_raw():
    from mnimi.extract.protocol import ExtractedFact
    from mnimi.extract.wrappers import ClientFactsExtractor

    inner = _Recording()
    w = ClientFactsExtractor(inner)
    turns = [{"role": "user", "content": "I run on Sundays. I live in Boston."}]
    w.provide([ExtractedFact("The user lives in Boston.", "", None, "user", "lives in",
                             "Boston", 1.0),
               ExtractedFact("The user runs on Sundays.", "user: I run on Sundays", None,
                             "user", "runs on", "Sundays", 1.0)])
    assert w.pending is not None
    r = w.extract(turns)
    assert [x.raw for x in r.facts] == ["user: I live in Boston.", "user: I run on Sundays"]
    assert r.truncated is False and inner.seen == [] and w.pending is None
    w.extract(turns)
    assert inner.seen == [turns], "with no queue the wrapper delegates"
    assert w.pins == inner.pins


def test_client_facts_wrapper_without_an_inner_extracts_nothing_and_has_its_own_pins():
    from mnimi.extract.wrappers import CLIENT_FACTS_PINS, ClientFactsExtractor

    w = ClientFactsExtractor(None)
    r = w.extract([{"role": "user", "content": "hello"}])
    assert r.facts == [] and r.truncated is False
    assert w.pins == CLIENT_FACTS_PINS and w.pins["extractor_model"] == "client-facts"


# --- TESTFIXESPT1 Task 4: resolver v3 ---


@pytest.mark.parametrize(
    "mention, ts, expected",
    [
        ("on Tuesday 10-20", "2026-10-12T00:00:00", "2026-10-20"),     # issue 3: MM-DD, forward
        ("Tuesday, 20-10", "2026-10-12T00:00:00", "2026-10-20"),       # DD-MM, the weekday decides
        ("on Monday 10/12", "2026-10-14T00:00:00", "2026-10-12"),       # slash, backward, nearest
        ("Saturday 01-02", "2026-12-30T00:00:00", "2027-01-02"),        # the year that fits
        ("Tuesday 10-20-26", "2026-10-12T00:00:00", "2026-10-20"),      # an explicit two-digit year
        ("on Tuesday", "2026-10-12T00:00:00", "2026-10-06"),            # v2 unchanged: no number
    ],
)
def test_resolver_v3_weekday_beside_a_numeric_date(mention, ts, expected):
    from mnimi.extract import resolver

    assert resolver.resolve(mention, ts) == expected


def test_resolver_v3_weekday_numeric_falls_through_when_no_reading_matches():
    from mnimi.extract import resolver

    # 2025-10-21 is a Tuesday but a year away (outside the half-year window); 2026-10-21 is a
    # Wednesday; "21-10" is no date. Today's rules take over: the most recent Tuesday.
    assert resolver.resolve("Tuesday 10-21", "2026-10-12T00:00:00") == "2026-10-06"
    # Neither reading is a date and no year fits: today's rules take over ("on Tuesday").
    assert resolver.resolve("on Tuesday 13-13", "2026-10-12T00:00:00") == "2026-10-06"


def test_resolver_version_is_v3_and_v2_markers_are_unchanged():
    from mnimi.extract import resolver

    assert resolver.RESOLVER_VERSION == "v3"
    assert resolver.FUTURE_MARKERS[0] == "next" and resolver.PAST_MARKERS[0] == "last"


# --- TESTFIXESPT1 review fix pass (2026-10-14) ---


def test_verify_raw_caps_a_grown_excerpt_at_the_schema_limit():
    from mnimi.extract.schema import MAX_TEXT_CHARS
    from mnimi.extract.verify import verify_raw

    turn = ("alpha beta gamma delta epsilon " * 30).strip() + " end"  # one unpunctuated run
    v = verify_raw("x", "assistant: alpha beta gamma delta epsilon alpha 检索-1",
                   [{"role": "assistant", "content": turn}])
    assert v.how == "prefix"
    assert len(v.raw) <= len("assistant: ") + MAX_TEXT_CHARS
    assert not v.raw.endswith(" ") and v.raw.startswith("assistant: alpha beta")


def test_verify_raw_knows_cjk_sentence_ends():
    from mnimi.extract.verify import sentences, verify_raw

    assert sentences("我喜欢猫。我住在北京！好吗？") == ["我喜欢猫。", "我住在北京！", "好吗？"]
    v = verify_raw("我住在北京", "", [{"role": "user", "content": "我喜欢猫。我住在北京。"}])
    assert v.how == "overlap" and v.raw == "user: 我住在北京。"


def test_verify_raw_stops_at_the_terminator_the_prefix_ends_on():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "assistant", "content": "Alpha beta gamma delta epsilon. Zeta eta theta."}]
    v = verify_raw("x", "assistant: Alpha beta gamma delta epsilon. 检索", turns)
    assert v.how == "prefix" and v.raw == "assistant: Alpha beta gamma delta epsilon."


def test_verify_raw_prefers_the_claimed_role_and_anchors_on_word_boundaries():
    from mnimi.extract.verify import verify_raw

    turns = [{"role": "user",
              "content": "I'm flying to Paris on Friday. Yesterday was long."},
             {"role": "assistant", "content": "flying to Paris on Friday, noted."}]
    v = verify_raw("x", "assistant: flying to Paris on Friday", turns)
    assert v.how == "verbatim" and v.role == "assistant"
    v2 = verify_raw("x", "assistant: Yes", turns)  # "Yes" is inside "Yesterday", not a word
    assert v2.how == "kept" and v2.raw == "assistant: Yes"


def test_verify_raw_overlap_needs_more_than_a_stopword():
    from mnimi.extract.verify import verify_raw

    v = verify_raw("The user owns a cat.", "user: I have a kitty",
                   [{"role": "user", "content": "The weather is bad."}])
    assert v.how == "kept" and v.raw == "user: I have a kitty"


@pytest.mark.parametrize(
    "mention, ts, expected",
    [
        ("on Friday 9-5", "2026-10-14T00:00:00", "2026-10-09"),      # a time range: "on Friday"
        ("Monday 1-2 pm", "2026-10-14T00:00:00", "2026-10-12"),      # a time range: bare weekday
        ("Tuesday 10-21", "2026-10-12T00:00:00", "2026-10-06"),      # no reading within half a year
    ],
)
def test_resolver_v3_only_dates_within_half_a_year(mention, ts, expected):
    from mnimi.extract import resolver

    assert resolver.resolve(mention, ts) == expected


def test_splitting_wrapper_splits_only_when_the_whole_round_overflows():
    from mnimi.extract.protocol import ExtractedFact, ExtractionResult
    from mnimi.extract.wrappers import SplittingExtractor

    def f(s):
        return ExtractedFact(s, "assistant: " + s, None, None, None, None, 0.5)

    turns = [{"role": "user", "content": "Q?"},
             {"role": "assistant", "content": "Aaaa aaaa. Bbbb bbbb. Cccc cccc."}]
    # The whole round extracts cleanly: one call, no split, its facts.
    inner = _Recording([ExtractionResult(facts=[f("Whole.")], truncated=False, raw_output="[w]")])
    r = SplittingExtractor(inner, max_chars=12).extract(turns)
    assert [x.content for x in r.facts] == ["Whole."] and inner.seen == [turns]
    # The whole round is truncated: the sub-rounds run and their facts are used.
    inner = _Recording([
        ExtractionResult(facts=[], truncated=True, raw_output="[w"),
        ExtractionResult(facts=[f("A.")], truncated=False, raw_output="[a]"),
        ExtractionResult(facts=[], truncated=True, raw_output="[b"),
        ExtractionResult(facts=[f("C.")], truncated=False, raw_output="[c]"),
    ])
    w = SplittingExtractor(inner, max_chars=12)
    r = w.extract(turns)
    assert [[t["content"] for t in s] for s in inner.seen] == [
        ["Q?", "Aaaa aaaa. Bbbb bbbb. Cccc cccc."], ["Q?", "Aaaa aaaa."], ["Bbbb bbbb."],
        ["Cccc cccc."],
    ]
    assert [x.content for x in r.facts] == ["A.", "C."] and w.sub_rounds == 3
    # The whole round hit MAX_FACTS: split, and keep whichever reading has more facts.
    eight = ExtractionResult(facts=[f(str(i)) for i in range(8)], truncated=False, raw_output="[8]")
    one = ExtractionResult(facts=[f("only")], truncated=False, raw_output="[1]")
    inner = _Recording([eight, one])
    r = SplittingExtractor(inner, max_chars=12).extract(turns)
    assert len(r.facts) == 8, "a split that yields fewer facts than the whole round loses"


def test_splitting_wrapper_also_splits_when_the_whole_round_extracts_nothing():
    from mnimi.extract.protocol import ExtractedFact, ExtractionResult
    from mnimi.extract.wrappers import SplittingExtractor

    def f(s):
        return ExtractedFact(s, "assistant: " + s, None, None, None, None, 0.5)

    turns = [{"role": "user", "content": "Q?"},
             {"role": "assistant", "content": "Aaaa aaaa. Bbbb bbbb. Cccc cccc."}]
    # A clean [] on a long round is the prompt's assistant rule firing, not a fit (run 3,
    # 2026-10-14: the "keep monitoring" round lost its 8 facts that way); the sub-rounds run.
    inner = _Recording([
        ExtractionResult(facts=[], truncated=False, raw_output="[]"),
        ExtractionResult(facts=[f("A.")], truncated=False, raw_output="[a]"),
        ExtractionResult(facts=[], truncated=False, raw_output="[]"),
        ExtractionResult(facts=[f("C.")], truncated=False, raw_output="[c]"),
    ])
    w = SplittingExtractor(inner, max_chars=12)
    r = w.extract(turns)
    assert [x.content for x in r.facts] == ["A.", "C."] and w.sub_rounds == 3
    # A short round that extracts nothing is left alone: one call.
    inner = _Recording([ExtractionResult(facts=[], truncated=False, raw_output="[]")])
    short = [{"role": "user", "content": "hi"}]
    assert SplittingExtractor(inner, max_chars=12).extract(short).facts == []
    assert len(inner.seen) == 1
