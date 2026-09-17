"""SPEC's ranking over rounds and its exact top-k (PHASE4 D4)."""

from __future__ import annotations

import math

import pytest

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder
from mnimi.models import MemoryRecord
from mnimi.ranking import check_weights, combined_score, rank_rounds, recency

W = {"similarity": 1.0, "recency": 0.0}


class _FakeStore:
    """``search`` over a fixed (record, cosine) list in cosine order; counts the calls."""

    def __init__(self, hits):
        self.hits = sorted(hits, key=lambda hit: -hit[1])
        self.calls = []

    def search(self, embedding, user_id, k, kind=None, active_only=False):
        self.calls.append(k)
        rows = [h for h in self.hits if not active_only or h[0].salience > 0]
        return rows[:k]


def _rec(id, round_key, salience=1.0, kind="round", created_at="2023-05-20"):
    return MemoryRecord(id=id, user_id="u", content=str(id), created_at=created_at,
                        salience=salience, kind=kind, round_key=round_key)


def test_recency_and_combined_score_follow_spec():
    assert recency(0, 30.0) == 1.0 and recency(30, 30.0) == 0.5
    assert combined_score(0.8, 0.5, 0.5, W) == pytest.approx(0.4)
    assert combined_score(0.8, 0.5, 0.5, {"similarity": 1.0, "recency": 0.2}) == pytest.approx(0.45)


def test_check_weights_refuses_anything_but_two_non_negative_numbers():
    check_weights(W)
    for bad in ({"similarity": 1.0}, {**W, "access": 1.0}, {**W, "recency": -0.1},
                {**W, "similarity": math.inf}, {**W, "recency": True}):
        with pytest.raises(ValueError, match="salience_weights"):
            check_weights(bad)


def test_rank_rounds_takes_the_top_k_by_score_not_the_top_k_by_vector():
    # Round A's most similar record is a low-salience fact; cutting at k by vector
    # first would keep A and lose C.
    store = _FakeStore([
        (_rec(1, "A", salience=0.25, kind="fact"), 0.9),
        (_rec(2, "B"), 0.8),
        (_rec(3, "C"), 0.5),
        (_rec(4, "A"), 0.3),
    ])
    top2 = rank_rounds(store, [0.0], "u", 2, weights=W, half_life_days=30.0, now=None)
    assert [hit.record.round_key for hit in top2] == ["B", "C"]
    top3 = rank_rounds(store, [0.0], "u", 3, weights=W, half_life_days=30.0, now=None)
    assert [hit.record.id for hit in top3] == [2, 3, 4], "A is represented by its round record"
    assert top3[2].score == pytest.approx(0.3) and top3[2].relevance == 0.3


def test_rank_rounds_stops_reading_once_no_unread_record_can_win():
    hits = [(_rec(1, "A"), 0.9)] + [(_rec(i, f"R{i}"), 0.1 - i / 1000) for i in range(2, 40)]
    store = _FakeStore(hits)
    (best,) = rank_rounds(store, [0.0], "u", 1, weights=W, half_life_days=30.0, now=None)
    assert best.record.id == 1 and store.calls == [1, 4], "bound 0.9 ties at k=1; 0.1 < 0.9 stops"


def test_recency_weight_prefers_the_later_session_on_equal_similarity():
    older = _rec(1, "old", created_at="2023/05/01 (Mon) 09:00")
    newer = _rec(2, "new", created_at="2023/06/30 (Fri) 09:00")
    store = _FakeStore([(older, 0.6), (newer, 0.6)])
    weights = {"similarity": 1.0, "recency": 0.5}
    ranked = rank_rounds(store, [0.0], "u", 2, weights=weights, half_life_days=30.0,
                         now="2023/06/30 (Fri) 09:00")
    assert [hit.record.id for hit in ranked] == [2, 1]
    assert ranked[1].recency == pytest.approx(0.25) and ranked[0].recency == 1.0


def test_rank_rounds_is_search_rounds_under_default_weights_and_uniform_salience(tmp_path):
    from mnimi.extract.fake import RuleExtractor

    m = Memory(str(tmp_path / "eq.db"), HashingEmbedder(), MemoryConfig(),
               extractor=RuleExtractor())
    for day, text in enumerate([
        "I run every morning. Yesterday I adopted a cat named Miso.",
        "I moved to Athens last year and I love the food.",
        "My sister works as a marine biologist in Crete.",
        "I bake sourdough bread every weekend with my neighbour.",
        "Last week I started learning the violin.",
    ], start=10):
        m.add([{"role": "user", "content": text, "ts": f"2023-05-{day}"},
               {"role": "assistant", "content": "Noted.", "ts": f"2023-05-{day}"}], user_id="u")
    assert {r.salience for r in m.store.active_records("u")} == {1.0}
    query = m._query_embedding("cat bread violin sister")
    for k in range(1, 7):
        expected = [(r.id, c) for r, c in m.store.search_rounds(query, "u", k=k)]
        got = rank_rounds(m.store, query, "u", k, weights=W, half_life_days=30.0,
                          now="2023-05-14")
        assert [(hit.record.id, hit.score) for hit in got] == expected, k
