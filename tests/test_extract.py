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
    assert resolver.RESOLVER_VERSION == "v1"
