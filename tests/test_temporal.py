"""PHASE6 D2/D3: the time-aware term — the query prefix, the grammar, the match, the ranking."""

from __future__ import annotations

from datetime import date

import pytest

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder
from mnimi.extract.fake import RuleExtractor
from mnimi.ranking import combined_score
from mnimi.temporal import (
    SLACK_DAYS,
    TEMPORAL_RULES,
    dated_query,
    effective_time,
    parse_window,
    split_query,
    temporal_rules_hash,
    time_match,
    window_for,
)

ANCHOR = date(2023, 4, 1)  # a Saturday


def test_the_rules_are_frozen_and_hashed():
    assert len(TEMPORAL_RULES) == 8 and SLACK_DAYS == 3
    assert temporal_rules_hash() == temporal_rules_hash()
    assert len(temporal_rules_hash()) == 64


def test_the_prefix_is_documented_and_stripped():
    q = dated_query("2023/06/22 (Thu) 18:33", "What did I do a week ago?")
    assert q == "[Current date: 2023/06/22 (Thu) 18:33]\nWhat did I do a week ago?"
    assert split_query(q) == ("2023/06/22 (Thu) 18:33", "What did I do a week ago?")
    assert split_query("What did I do a week ago?") == (None, "What did I do a week ago?")
    assert dated_query(None, "bare") == "bare"
    # A prefix-shaped line inside the question body is not a prefix.
    assert split_query("x\n[Current date: 2023/06/22]\ny")[0] is None


@pytest.mark.parametrize(
    "question, start, end, precision",
    [
        # PHASE6-ANALYSIS §5: the six temporal rows with nothing in the top-10.
        ("What kitchen appliance did I buy 10 days ago?", "2023-03-19", "2023-03-25", "day"),
        ("I mentioned an investment for a competition four weeks ago? What did I buy?",
         "2023-03-01", "2023-03-07", "week"),
        ("What did I do with Rachel on the Wednesday two months ago?",
         "2023-02-01", "2023-02-28", "month"),
        ("What was the significant business milestone I mentioned four weeks ago?",
         "2023-03-01", "2023-03-07", "week"),
        ("Where was the art-related event two weeks ago held at?", "2023-03-15", "2023-03-21",
         "week"),
        ("What was the life event of a relative that I participated in a week ago?",
         "2023-03-22", "2023-03-28", "week"),
        # Ranges carry no slack.
        ("How many museums did I visit in the past three months?", "2023-01-01", "2023-04-01",
         "range"),
        ("How many art events did I attend in the past month?", "2023-03-01", "2023-04-01",
         "range"),
        ("What did I do last week?", "2023-03-25", "2023-04-01", "range"),
        ("What did I buy last month?", "2023-03-01", "2023-03-31", "range"),
        ("What did I do last year?", "2022-01-01", "2022-12-31", "range"),
        # A named month: the most recent on or before the anchor, whole.
        ("How many museums did I visit in the month of February?", "2023-02-01", "2023-02-28",
         "month"),
        ("What is the order of the sports events I watched in January?",
         "2023-01-01", "2023-01-31", "month"),
        ("What did I do in June?", "2022-06-01", "2022-06-30", "month"),
        ("What did I eat yesterday?", "2023-03-28", "2023-04-03", "day"),
        ("What did I do last Wednesday?", "2023-03-26", "2023-04-01", "day"),
        ("A year ago, what did I plan?", "2022-04-01", "2022-04-30", "year"),
    ],
)
def test_the_grammar_names_the_window(question, start, end, precision):
    w = parse_window(question, ANCHOR)
    assert w is not None, question
    assert (w.start.isoformat(), w.end.isoformat(), w.precision) == (start, end, precision)


@pytest.mark.parametrize(
    "question",
    [
        "What is my preferred gin-to-vermouth ratio?",
        "How many hours have I spent playing games in total?",
        "Which show did I start watching first?",
        "What did I do a few days ago?",  # 'few' is not a number: no guess
    ],
)
def test_no_expression_means_no_window(question):
    assert parse_window(question, ANCHOR) is None
    assert parse_window("10 days ago", None) is None, "no anchor, no window"
    assert window_for(question)[1] is None


def test_effective_time_and_the_match_rule():
    assert effective_time("2023-03-15", "2023/03/20 (Mon) 10:00") == "2023-03-15"
    assert effective_time(None, "2023/03/20 (Mon) 10:00") == "2023-03-20"
    assert effective_time(None, None) is None
    w = parse_window("what did I buy 10 days ago?", date(2023, 3, 25))
    assert time_match("2023-03-15", w) == 1.0
    assert time_match("2023-03-18", w) == 1.0 and time_match("2023-03-19", w) == 0.0
    assert time_match("2023-03", w) == 1.0, "a month overlapping the window matches"
    assert time_match("2023-02", w) == 0.0
    assert time_match("2023", w) == 0.0, "a bare year never matches"
    assert time_match(None, w) == 0.0 and time_match("2023-03-15", None) == 0.0
    assert time_match("not-a-date", w) == 0.0


def test_the_score_carries_the_term():
    w = {"similarity": 1.0, "recency": 0.0}
    assert combined_score(0.5, 1.0, 1.0, w) == 0.5
    assert combined_score(0.5, 1.0, 1.0, w, time_match=1.0, time_weight=0.05) == pytest.approx(
        0.55
    )
    half = combined_score(0.5, 1.0, 0.5, w, time_match=1.0, time_weight=0.05)
    assert half == pytest.approx(0.275)


def _session(ts, user, assistant):
    return [{"role": "user", "content": user, "ts": ts},
            {"role": "assistant", "content": assistant, "ts": ts}]


def _fill(memory):
    memory.add(_session("2023/03/15 (Wed) 11:56",
                        "I just got a smoker today and want BBQ sauce recipes.",
                        "Congratulations on the smoker! Here are recipes."), user_id="u")
    memory.add(_session("2023/02/10 (Fri) 12:00",
                        "I bought a stand mixer for my kitchen and need bread recipes.",
                        "A stand mixer is great for bread."), user_id="u")
    memory.add(_session("2023/03/24 (Fri) 12:00",
                        "What kitchen appliance should I buy next for smoking meat?",
                        "A smoker is the classic choice."), user_id="u")


def test_time_weight_zero_is_the_previous_ranking_and_the_vector_never_moves(tmp_path):
    # 0.0 is the published arm's configuration; the library default is 0.05 since v2.12.0.
    m = Memory(str(tmp_path / "a.db"), HashingEmbedder(), MemoryConfig(time_weight=0.0))
    _fill(m)
    bare = "What kitchen appliance did I buy 10 days ago?"
    dated = dated_query("2023/03/25 (Sat) 18:26", bare)
    assert m._query_embedding(dated) == m._query_embedding(bare), "the prefix is stripped"
    assert m._window(dated) is None, "time_weight 0: no window, the term is inert"
    a = [h.record.id for h in m.recall(bare, "u")]
    b = [h.record.id for h in m.recall(dated, "u")]
    assert a == b


def test_the_window_lifts_the_dated_round(tmp_path):
    cfg = MemoryConfig(time_weight=0.05)
    m = Memory(str(tmp_path / "b.db"), HashingEmbedder(), cfg)
    _fill(m)
    dated = dated_query("2023/03/25 (Sat) 18:26", "What kitchen appliance did I buy 10 days ago?")
    w = m._window(dated)
    assert (w.start, w.end) == (date(2023, 3, 12), date(2023, 3, 18))
    hits = m.recall(dated, "u")
    matched = [h for h in hits if h.time_match > 0]
    assert len(matched) == 1 and "smoker today" in matched[0].record.content
    assert matched[0].score == pytest.approx(matched[0].relevance + 0.05)
    unmatched = [h for h in hits if h.time_match == 0]
    assert all(h.score == pytest.approx(h.relevance) for h in unmatched)
    # The bare question (no prefix) under the same config: no anchor, no window.
    assert m._window("What kitchen appliance did I buy 10 days ago?") is None


def test_a_fact_valid_time_is_the_effective_time_under_extraction(tmp_path):
    cfg = MemoryConfig(time_weight=0.05)
    m = Memory(str(tmp_path / "c.db"), HashingEmbedder(), cfg, extractor=RuleExtractor())
    m.add(_session("2023/03/20 (Mon) 09:00",
                   "I bought a smoker on March 15th and love it.", "Great choice."), user_id="u")
    dated = dated_query("2023/03/25 (Sat) 18:26", "What did I buy 10 days ago?")
    hits = m.recall(dated, "u")
    assert hits and hits[0].time_match in (0.0, 1.0)
    facts = [r for r in m.store.all_records("u") if r.kind == "fact"]
    dated_facts = [f for f in facts if f.valid_time]
    if dated_facts:  # the rule extractor found the date: the fact's valid_time drives the match
        assert hits[0].time_match == 1.0


def test_time_weight_is_validated(tmp_path):
    with pytest.raises(ValueError, match="time_weight"):
        Memory(str(tmp_path / "d.db"), HashingEmbedder(), MemoryConfig(time_weight=-0.1))
    with pytest.raises(ValueError, match="time_weight"):
        Memory(str(tmp_path / "e.db"), HashingEmbedder(), MemoryConfig(time_weight=float("nan")))


def test_the_harness_hands_mnimi_the_dated_query_and_pins_the_term():
    from evals.systems.mnimi import MnimiSystem

    system = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(time_weight=0.05))
    system.set_question_date("2023/03/25 (Sat) 18:26")
    assert system.query_for("q?") == "[Current date: 2023/03/25 (Sat) 18:26]\nq?"
    pins = system.retrieval_pins()
    assert pins["time_weight"] == 0.05 and pins["temporal_rules_hash"] == temporal_rules_hash()
    off = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(time_weight=0.0))
    assert off.retrieval_pins()["time_weight"] == 0.0
    # The library default since v2.12.0 (gate 6-iv arm 1): the adopted 0.05.
    assert MnimiSystem(embedder=HashingEmbedder()).retrieval_pins()["time_weight"] == 0.05


def test_the_probe_ranks_the_dated_query():
    from evals.dataset import Question, Session
    from evals.probes.retrieval import probe_question
    from evals.systems.mnimi import MnimiSystem

    turns_a = [{"role": "user", "content": "I just got a smoker today.", "has_answer": True},
               {"role": "assistant", "content": "Nice.", "has_answer": True}]
    turns_b = [{"role": "user", "content": "I bought a stand mixer.", "has_answer": False},
               {"role": "assistant", "content": "Nice.", "has_answer": False}]
    q = Question(
        question_id="t", question_type="temporal-reasoning",
        question="What kitchen appliance did I buy 10 days ago?", answer="a smoker",
        question_date="2023/03/25 (Sat) 18:26",
        sessions=[Session("s_b", "2023/02/10 (Fri) 12:00", turns_b),
                  Session("s_a", "2023/03/15 (Wed) 11:56", turns_a)],
        answer_session_ids=["s_a"],
    )
    system = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(time_weight=0.05))
    row = probe_question(system, q)
    assert row.parsed_window == ["2023-03-12", "2023-03-18", "day", "10 days ago"]
    assert row.time_matches_top10 == 1, "the smoker round's session date is in the window"
    off = probe_question(
        MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(time_weight=0.0)), q
    )
    assert off.parsed_window is None and off.time_matches_top10 == 0
    # The term can only lift the dated evidence round (the hashing embedder's
    # cosines are coarse, so equality is allowed; the real embedder is the probe's job).
    assert row.evidence_ranks[0] <= off.evidence_ranks[0]
