"""The retrieval probe: ingest through the real system, observe dedup, rank evidence."""

from __future__ import annotations

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
