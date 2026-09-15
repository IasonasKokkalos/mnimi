"""The retrieval probe: ingest through the real system, observe dedup, rank evidence."""

from __future__ import annotations

import json

from evals.dataset import Question, Session
from evals.probes import aggregate, retrieval
from evals.systems.mnimi import MnimiSystem
from evals.systems.naive_rag import NaiveRagSystem

from mnimi import MemoryConfig
from mnimi.embeddings import HashingEmbedder


def _question(qid="q1", category="single-session-user"):
    # Sessions are fed in list (file) order, as the runner feeds them. The embed
    # text folds the session DATE in, so an exact duplicate can only occur
    # within one session: s1's third round repeats its second, and the
    # ``has_answer`` flag sits on the repeat, so the evidence round is the one
    # the exact screen drops.
    return Question(
        question_id=qid,
        question_type=category,
        question="what did I adopt?",
        answer="a cat named Miso",
        question_date="2023/05/21 (Sun) 10:00",
        sessions=[
            Session(
                session_id="s2",
                date="2023/05/19 (Fri) 09:00",
                turns=[
                    {"role": "user", "content": "tell me a joke"},
                    {"role": "assistant", "content": "Why did the cat cross the road?"},
                ],
            ),
            Session(
                session_id="s1",
                date="2023/05/20 (Sat) 09:00",
                turns=[
                    {"role": "user", "content": "unrelated: the weather is grey"},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "I adopted a cat named Miso"},
                    {"role": "assistant", "content": "Lovely, what breed?"},
                    {"role": "user", "content": "I adopted a cat named Miso", "has_answer": True},
                    {"role": "assistant", "content": "Lovely, what breed?"},
                ],
            ),
        ],
        answer_session_ids=["s1"],
    )


def test_probe_tags_evidence_rounds_and_ranks_them():
    system = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig())
    row = retrieval.probe_question(system, _question())
    assert row.question_id == "q1" and row.n_rounds == 4
    assert row.n_evidence_rounds == 1
    # The evidence round is an exact duplicate of s1's second round and is
    # dropped by the exact screen: the strict rule counts that as a miss, and
    # the drop is attributed to the round that carried the flag.
    assert row.evidence_ranks == [-1]
    (drop,) = [d for d in row.drops if d.is_evidence]
    assert (drop.screen, drop.session_id, drop.round_index) == ("exact", "s1", 2)
    assert row.stored == 3 and row.stored + len(row.drops) == row.n_rounds
    assert len(row.top_sessions) == 3 and set(row.top_sessions) == {"s1", "s2"}


def test_probe_on_naive_rag_stores_every_round():
    system = NaiveRagSystem(embedder=HashingEmbedder(), config=MemoryConfig())
    row = retrieval.probe_question(system, _question())
    assert row.stored == 4 and row.drops == []
    assert row.evidence_ranks[0] in range(1, 51)


def test_aggregate_reports_recall_at_k_and_drop_counts(tmp_path):
    rows = [
        retrieval.probe_question(MnimiSystem(embedder=HashingEmbedder()), _question("a")),
        retrieval.probe_question(NaiveRagSystem(embedder=HashingEmbedder()), _question("b")),
    ]
    out = tmp_path / "probe.json"
    retrieval.write_rows(out, rows)
    summary = aggregate.summarize(retrieval.read_rows(out))
    assert summary["n_answerable"] == 2
    assert summary["any_at"][10] == 1 and summary["all_at"][10] == 1
    assert summary["drops"] == {"exact": 1, "cosine": 0, "evidence_lost": 1}
    assert summary["evidence_lost_rows"] == [("a", "exact")]
    text = aggregate.format_summary(summary)
    assert "ANY@10 1/2" in text and "evidence lost: a (exact)" in text


def test_misses_classify_retrieval_vs_reading(tmp_path):
    from evals.probes import misses

    rows = [
        retrieval.QuestionProbe("a", "multi-session", False, 5, 2, 5, [3, 7], ["s"] * 5),
        retrieval.QuestionProbe("b", "temporal-reasoning", False, 5, 2, 5, [-1, 12], ["s"] * 5),
        retrieval.QuestionProbe("c", "single-session-user", False, 5, 1, 5, [1], ["s"] * 5),
        retrieval.QuestionProbe("d_abs", "temporal-reasoning", True, 5, 0, 5, [], ["s"] * 5),
    ]
    correct = {"a": False, "b": False, "c": True, "d_abs": False}
    entries = misses.classify(rows, correct, k=10)
    assert [(e["question_id"], e["kind"]) for e in entries] == [
        ("a", "reading-miss"), ("b", "retrieval-miss"), ("d_abs", "reading-miss"),
    ]
    assert entries[0]["evidence_in_top_k"] == 2
    misses.annotate(entries, "naive", {"a": False, "b": True, "d_abs": False})
    text = misses.format_table(entries, 10, ["naive"])
    assert "1 retrieval-miss" in text and "2 reading-miss" in text
    assert "naive right on: 1 of the retrieval misses, 0 of the reading misses" in text

    probe_path = tmp_path / "probe.json"
    retrieval.write_rows(probe_path, rows)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({
        "pins": {"k": 10},
        "results": [{"question_id": q, "correct": ok} for q, ok in correct.items()],
    }), encoding="utf-8")
    assert misses.main([str(probe_path), str(results)]) == 0


def test_probe_maps_fact_records_to_their_round_and_counts_the_stage(tmp_path):
    """PHASE2: with an extractor a round is several records; the probe ranks
    the ROUND (best record wins), attributes drops by kind and carries the
    extraction counters. The question's evidence round is an exact duplicate
    of an earlier round; its round record is dropped, and so is its fact
    (same session, same fact text) — the strict rule still counts the miss."""
    from mnimi.extract.fake import RuleExtractor

    system = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(),
                         extractor=RuleExtractor(), extraction_cache=tmp_path / "c.sqlite")
    row = retrieval.probe_question(system, _question())
    assert row.n_rounds == 4 and row.n_evidence_rounds == 1
    # The "unrelated: the weather is grey" / "Noted." round has no slot and is
    # never sent; the other three are.
    assert row.rounds_sent_to_model == 3 and row.prefilter_skips == 1
    assert row.facts_stored == 1, "one fact from the first 'I adopted' round; its repeat is dropped"
    assert row.stored == 4, "three round records plus one fact record"
    assert {d.kind for d in row.drops} == {"round", "fact"}
    assert row.evidence_ranks == [-1] and row.lost_evidence == [["s1", 2, "exact"]]
    # Every ranked hit maps back to a round; the cat round is the best hit and
    # collapses its fact and its round record into one rank.
    assert len(row.top_sessions) == 3 and row.top_sessions[0] == "s1"
    summary = aggregate.summarize([row])
    assert summary["extraction"]["facts_stored"] == 1
    assert summary["drops_by_kind"] == {"fact/exact": 1, "round/exact": 1}
    assert "facts stored 1" in aggregate.format_summary(summary)
    # Two model calls in all: the joke round and the first cat round; the repeated
    # cat round is the same turns, hence the same cache key, and the probe's
    # mirror walk costs nothing (D10).
    assert system.cache_stats == {"hits": 4, "misses": 2}


# -- Phase 3 Task 2: the probe mirrors the screens; the identity tool ---------------------


def _conflict_question():
    # One session states a residence twice with different values (the value
    # screen's case, inside session scope); a second session is unrelated.
    return Question(
        question_id="q-conflict", question_type="knowledge-update",
        question="where does the user live?", answer="Seattle",
        question_date="2023/05/21 (Sun) 10:00",
        sessions=[
            Session(session_id="s1", date="2023/05/20 (Sat) 09:00", turns=[
                {"role": "user", "content": "I live in Boston now, by the way."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "Update: I live in Seattle now.", "has_answer": True},
                {"role": "assistant", "content": "Got it."},
            ]),
            Session(session_id="s2", date="2023/05/19 (Fri) 09:00", turns=[
                {"role": "user", "content": "tell me a joke"},
                {"role": "assistant", "content": "Why did the cat cross the road?"},
            ]),
        ],
        answer_session_ids=["s1"],
    )


def test_probe_mirrors_the_screens_and_counts_keeps(tmp_path):
    from mnimi.extract.fake import ScriptedExtractor
    from mnimi.extract.protocol import ExtractedFact

    def fact(content, raw, obj):
        return ExtractedFact(content, raw, None, "user", "lives in", obj, 1.0)

    script = {
        "I live in Boston now, by the way.": [
            fact("The user lives in Boston.", "user: I live in Boston now, by the way.", "Boston")],
        "Update: I live in Seattle now.": [
            fact("The user lives in Seattle.", "user: Update: I live in Seattle now.", "Seattle")],
    }
    system = MnimiSystem(embedder=HashingEmbedder(),
                         config=MemoryConfig(dedup_cosine_threshold=0.5),
                         extractor=ScriptedExtractor(script),
                         extraction_cache=tmp_path / "c.sqlite")
    row = retrieval.probe_question(system, _conflict_question())
    # The two s1 facts sit above 0.5 under HashingEmbedder and share the pair
    # key with different values: the value screen keeps the second (the walk
    # mirrors the same verdict, so its insert bookkeeping still matches).
    assert row.screen_keeps == {"negation": 0, "value": 1, "low-entropy": 0}
    assert row.facts_stored == 2
    assert row.stored == system._memory.store.count("eval")
    assert [d.kind for d in row.drops] in ([], ["round"]), "only the round screen may have dropped"
    off = MnimiSystem(embedder=HashingEmbedder(),
                      config=MemoryConfig(dedup_cosine_threshold=0.5, conflict_resolution=False),
                      extractor=ScriptedExtractor(script), extraction_cache=tmp_path / "d.sqlite")
    row_off = retrieval.probe_question(off, _conflict_question())
    assert row_off.facts_stored == 1 and "fact" in {d.kind for d in row_off.drops}
    assert row_off.screen_keeps == {"negation": 0, "value": 0, "low-entropy": 0}
    # aggregate carries the new counters and prints them.
    summary = aggregate.summarize([row, row_off])
    assert summary["screen_keeps"] == {"low-entropy": 0, "negation": 0, "value": 1}
    assert "screen keeps" in aggregate.format_summary(summary)


def test_identity_counts_identical_rows(tmp_path):
    from evals.probes import identity

    a = retrieval.QuestionProbe("a", "multi-session", False, 5, 2, 5, [3, 7], ["s"] * 5)
    b = retrieval.QuestionProbe("b", "temporal-reasoning", False, 5, 2, 5, [-1, 12], ["s"] * 5)
    b_moved = retrieval.QuestionProbe("b", "temporal-reasoning", False, 5, 2, 5, [1, 12],
                                      ["s"] * 5)
    assert identity.compare([a, b], [a, b]) == (["a", "b"], [])
    assert identity.compare([a, b], [a, b_moved]) == (["a"], ["b"])
    left, right = tmp_path / "l.json", tmp_path / "r.json"
    retrieval.write_rows(left, [a, b])
    retrieval.write_rows(right, [a, b_moved])
    assert identity.main([str(left), str(left)]) == 0
    assert identity.main([str(left), str(right)]) == 1
    assert "identical 1/2" in identity.format_report([a, b], [a, b_moved])


# -- Phase 3 Task 3: the probe counts conflicts and supersessions ---------------------------


def _cross_session_question():
    # The residence changes across two sessions: out of the cosine gate's
    # scope, found through the pair index, superseded by the ordering.
    return Question(
        question_id="q-update", question_type="knowledge-update",
        question="where does the user live?", answer="Seattle",
        question_date="2023/07/01 (Sat) 10:00",
        sessions=[
            Session(session_id="s1", date="2023/05/20 (Sat) 09:00", turns=[
                {"role": "user", "content": "I live in Boston now, by the way."},
                {"role": "assistant", "content": "Noted."},
            ]),
            Session(session_id="s2", date="2023/06/20 (Tue) 09:00", turns=[
                {"role": "user", "content": "Update: I live in Seattle now.", "has_answer": True},
                {"role": "assistant", "content": "Got it."},
            ]),
        ],
        answer_session_ids=["s2"],
    )


def test_probe_counts_conflicts_and_supersessions(tmp_path):
    from mnimi.extract.fake import ScriptedExtractor
    from mnimi.extract.protocol import ExtractedFact

    def fact(content, raw, obj):
        return ExtractedFact(content, raw, None, "user", "lives in", obj, 1.0)

    script = {
        "I live in Boston now, by the way.": [
            fact("The user lives in Boston.", "user: I live in Boston now, by the way.", "Boston")],
        "Update: I live in Seattle now.": [
            fact("The user lives in Seattle.", "user: Update: I live in Seattle now.", "Seattle")],
    }
    system = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(),
                         extractor=ScriptedExtractor(script),
                         extraction_cache=tmp_path / "c.sqlite")
    row = retrieval.probe_question(system, _cross_session_question())
    assert row.conflicts == {"negation": 0, "functional": 1, "numeric": 0}
    assert row.superseded == 1 and row.facts_stored == 2
    assert row.stored == 4, "a superseded loser is still stored (D8)"
    summary = aggregate.summarize([row])
    assert summary["superseded"] == 1 and summary["conflicts"]["functional"] == 1
    assert summary["by_category"]["knowledge-update"]["superseded"] == 1
    assert "superseded 1" in aggregate.format_summary(summary)
    off = MnimiSystem(embedder=HashingEmbedder(), config=MemoryConfig(conflict_resolution=False),
                      extractor=ScriptedExtractor(script), extraction_cache=tmp_path / "d.sqlite")
    assert retrieval.probe_question(off, _cross_session_question()).superseded == 0
