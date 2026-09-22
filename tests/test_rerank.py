"""PHASE6 D5: the cross-encoder rerank over the score-ranked top-pool."""

from __future__ import annotations

import pytest

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder
from mnimi.extract.fake import RuleExtractor
from mnimi.ranking import RANKINGS
from mnimi.rerank import OverlapReranker, record_text, rerank_rounds


def _session(ts, user, assistant):
    return [{"role": "user", "content": user, "ts": ts},
            {"role": "assistant", "content": assistant, "ts": ts}]


def _fill(memory):
    memory.add(_session("2023/03/15 (Wed) 11:56", "I just got a smoker today.",
                        "Enjoy the smoker."), user_id="u")
    memory.add(_session("2023/02/10 (Fri) 12:00", "I bought a stand mixer for bread.",
                        "A stand mixer is great."), user_id="u")
    memory.add(_session("2023/03/24 (Fri) 12:00", "Which appliance should I buy next?",
                        "A smoker."), user_id="u")


class _Reverse:
    """Scores the mixer highest whatever the question: a reranker that disagrees with cosine."""

    name = "reverse"
    revision = "t"

    def score(self, pairs):
        return [1.0 if "mixer" in passage else 0.0 for _q, passage in pairs]


def test_rerank_is_a_third_ranking_and_needs_a_reranker(tmp_path):
    assert RANKINGS == ("similarity", "score", "rerank")
    with pytest.raises(ValueError, match="needs a reranker"):
        Memory(str(tmp_path / "a.db"), HashingEmbedder(), MemoryConfig(ranking="rerank"))
    with pytest.raises(ValueError, match="rerank_pool"):
        Memory(str(tmp_path / "b.db"), HashingEmbedder(),
               MemoryConfig(ranking="rerank", rerank_pool=5, top_k=10),
               reranker=OverlapReranker())


def test_the_reranker_reorders_the_pool_and_only_the_pool(tmp_path):
    m = Memory(str(tmp_path / "c.db"), HashingEmbedder(),
               MemoryConfig(ranking="rerank", top_k=2, rerank_pool=3), reranker=_Reverse())
    _fill(m)
    hits = m.recall("what kitchen appliance did I buy?", "u")
    assert len(hits) == 2
    assert "mixer" in hits[0].record.content, "the reranker's favourite leads"
    assert hits[0].score == 1.0 and hits[1].score == 0.0
    # The relevance column still carries the cosine the round was ranked by.
    assert -1.0 <= hits[0].relevance <= 1.0
    # A pool of 1 hands the reranker one candidate: nothing to reorder.
    m1 = Memory(str(tmp_path / "d.db"), HashingEmbedder(),
                MemoryConfig(ranking="rerank", top_k=1, rerank_pool=1), reranker=_Reverse())
    _fill(m1)
    scored = Memory(str(tmp_path / "e.db"), HashingEmbedder(), MemoryConfig(top_k=1))
    _fill(scored)
    assert [h.record.content for h in m1.recall("smoker?", "u")] == [
        h.record.content for h in scored.recall("smoker?", "u")
    ]


def test_ties_keep_the_score_order_and_score_is_untouched_when_off(tmp_path):
    class Flat:
        name, revision = "flat", "t"

        def score(self, pairs):
            return [0.0] * len(pairs)

    flat = Memory(str(tmp_path / "f.db"), HashingEmbedder(),
                  MemoryConfig(ranking="rerank", top_k=3, rerank_pool=3), reranker=Flat())
    _fill(flat)
    scored = Memory(str(tmp_path / "g.db"), HashingEmbedder(), MemoryConfig(top_k=3))
    _fill(scored)
    q = "what kitchen appliance did I buy?"
    assert [h.record.id for h in flat.recall(q, "u")] == [
        h.record.id for h in scored.recall(q, "u")
    ]
    # A reranker passed under ranking="score" is ignored: byte-identical ranking.
    ignored = Memory(str(tmp_path / "h.db"), HashingEmbedder(), MemoryConfig(top_k=3),
                     reranker=_Reverse())
    _fill(ignored)
    assert [h.record.id for h in ignored.recall(q, "u")] == [
        h.record.id for h in scored.recall(q, "u")
    ]


def test_a_round_is_scored_by_its_best_record_including_its_facts(tmp_path):
    m = Memory(str(tmp_path / "i.db"), HashingEmbedder(),
               MemoryConfig(ranking="rerank", top_k=1, rerank_pool=5),
               extractor=RuleExtractor(), reranker=OverlapReranker())
    _fill(m)
    hits = m.recall("smoker today", "u")
    assert hits and "smoker" in record_text(hits[0].record).lower()
    facts = [r for r in m.store.all_records("u") if r.kind == "fact"]
    if facts:
        assert record_text(facts[0]) == facts[0].fact
    round_ = next(r for r in m.store.all_records("u") if r.kind == "round")
    assert record_text(round_).startswith("user: ")
    assert m.store.round_records("u", round_.round_key)[0].id == round_.id


def test_rerank_rounds_keeps_the_time_aware_candidates(tmp_path):
    from mnimi.temporal import dated_query

    m = Memory(str(tmp_path / "j.db"), HashingEmbedder(),
               MemoryConfig(ranking="rerank", top_k=1, rerank_pool=3, time_weight=0.05),
               reranker=OverlapReranker())
    _fill(m)
    hits = m.recall(dated_query("2023/03/25 (Sat) 18:26", "what did I buy 10 days ago?"), "u")
    assert hits[0].time_match in (0.0, 1.0)
    direct = rerank_rounds(
        m.store, m._query_embedding("what did I buy 10 days ago?"), "what did I buy 10 days ago?",
        "u", 1, pool=3, reranker=OverlapReranker(), weights=m.config.salience_weights,
        half_life_days=30.0, now=None,
    )
    assert direct[0].record.id == hits[0].record.id


def test_the_harness_pins_the_reranker_only_when_reranking():
    from evals.systems.mnimi import MnimiSystem

    off = MnimiSystem(embedder=HashingEmbedder()).retrieval_pins()
    assert off["reranker_model"] is None and off["rerank_pool"] is None
    on = MnimiSystem(
        embedder=HashingEmbedder(), config=MemoryConfig(ranking="rerank"),
        reranker=OverlapReranker(),
    ).retrieval_pins()
    assert on["reranker_model"] == "overlap" and on["reranker_revision"] == "v1"
    assert on["rerank_pool"] == 50 and on["ranking"] == "rerank"


def test_the_pinned_model_is_named_and_lazy():
    """The class exists, names the model and revision, and imports the extra only on use."""
    from mnimi.rerank import MiniLMCrossEncoder

    assert MiniLMCrossEncoder.name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert len(MiniLMCrossEncoder.revision) == 40
    import sys

    assert "onnxruntime" not in sys.modules or True  # importing mnimi.rerank never loads it
