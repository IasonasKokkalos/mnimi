"""The three Phase C systems: oracle (ceiling), naive_rag (bar), mnimi.

Runs under the ``[dev]`` extra alone: every test injects ``HashingEmbedder``,
so nothing here downloads a model. ``build_system`` is what hardcodes the real
BGE embedder for a run.

The load-bearing test in this file is granularity parity — naive_rag and mnimi
must build byte-identical records from the same messages, or the number their
comparison produces is measuring ingestion differences instead of dedup.
"""

from __future__ import annotations

from evals.base import MemorySystem
from evals.dataset import Question, Session
from evals.runner import _sessions_for
from evals.systems.full_history import FullHistorySystem
from evals.systems.mnimi import MnimiSystem
from evals.systems.naive_rag import NaiveRagSystem
from evals.systems.no_memory import NoMemorySystem
from evals.systems.oracle import OracleSystem

from mnimi import MemoryConfig
from mnimi.embeddings import HashingEmbedder

ALL_SYSTEMS = (NoMemorySystem, FullHistorySystem, OracleSystem, NaiveRagSystem, MnimiSystem)


def _mnimi(**kwargs) -> MnimiSystem:
    return MnimiSystem(embedder=HashingEmbedder(), **kwargs)


def _naive(**kwargs) -> NaiveRagSystem:
    return NaiveRagSystem(embedder=HashingEmbedder(), **kwargs)


def _round(user: str, assistant: str, ts: str = "2023-05-20") -> list[dict]:
    return [
        {"role": "user", "content": user, "ts": ts},
        {"role": "assistant", "content": assistant, "ts": ts},
    ]


def _question(evidence: list[str]) -> Question:
    turns = [{"role": "user", "content": "a turn"}]
    sessions = [
        Session(session_id=f"s{i}", date="2023-05-20", turns=turns) for i in range(4)
    ]
    return Question(
        question_id="q1",
        question_type="single-session-user",
        question="what?",
        answer="a",
        question_date="2023-06-01",
        sessions=sessions,
        answer_session_ids=evidence,
    )


def test_all_five_systems_implement_the_contract():
    for cls in ALL_SYSTEMS:
        assert issubclass(cls, MemorySystem), cls


def test_only_oracle_declares_evidence_only():
    declared = {cls.name for cls in ALL_SYSTEMS if cls.evidence_only}
    assert declared == {"oracle"}


def test_oracle_is_fed_only_the_annotated_evidence_sessions():
    q = _question(evidence=["s1", "s3"])
    fed = _sessions_for(OracleSystem(), q)
    assert [s.session_id for s in fed] == ["s1", "s3"]


def test_every_other_system_is_fed_the_whole_haystack():
    q = _question(evidence=["s1"])
    for cls in (NoMemorySystem, FullHistorySystem, NaiveRagSystem, MnimiSystem):
        fed = _sessions_for(cls.__new__(cls), q)  # no construction: capability is a class attr
        assert [s.session_id for s in fed] == ["s0", "s1", "s2", "s3"], cls


def test_oracle_formats_identically_to_full_history():
    """Formatting is worth up to 10 points at oracle retrieval (Fig 6) — so the
    ceiling must not differ from the baseline by its renderer."""
    assert OracleSystem.get_context is FullHistorySystem.get_context

    oracle, full = OracleSystem(), FullHistorySystem()
    for system in (oracle, full):
        system.reset()
        system.add(_round("I moved to Athens", "Noted", ts="2023-05-20"))
        system.add(_round("I adopted a dog", "Lovely", ts="2023-07-01"))
    assert oracle.get_context("q") == full.get_context("q")


def test_naive_rag_and_mnimi_store_identical_records():
    """Granularity parity: the ingestion variable must not float between them."""
    messages = [
        *_round("my sister is a marine biologist in Crete", "fascinating work", ts="2023-05-20"),
        *_round("the talk covered sqlite virtual tables", "a powerful extension point"),
        *_round("I bake sourdough on sundays", "fresh yeast helps", ts="2023-07-01"),
    ]
    naive, mnimi = _naive(), _mnimi()
    for system in (naive, mnimi):
        system.reset()
        system.add(messages)

    assert naive.contents() == mnimi.contents()
    assert len(naive.contents()) == 3


def test_dedup_is_the_single_delta():
    """The same duplicate round: naive_rag keeps both, mnimi keeps one."""
    messages = [*_round("I live in Athens", "noted"), *_round("I live in Athens", "noted")]
    naive, mnimi = _naive(), _mnimi()
    for system in (naive, mnimi):
        system.reset()
        system.add(messages)

    assert len(naive.contents()) == 2
    assert len(mnimi.contents()) == 1


def test_both_retrieval_systems_read_top_k_from_config():
    config = MemoryConfig(top_k=2)
    for system in (_naive(config=config), _mnimi(config=config)):
        system.reset()
        for i in range(5):
            # Distinct ts per round: the shared renderer emits one header per
            # timestamp CHANGE, so same-ts blocks would merge under one header
            # and the count would measure the renderer, not top_k.
            system.add(
                _round(f"distinct fact number {i} about topic {i}", f"reply {i}",
                       ts=f"2023-05-{10 + i}")
            )
        assert system.get_context("tell me about my life").count("[Session date:") == 2


def test_reset_gives_each_question_a_fresh_store_on_disk():
    """A file, not ``:memory:`` — the on-disk path is what a user runs."""
    for system in (_naive(), _mnimi()):
        system.reset()
        system.add(_round("I live in Athens", "noted"))
        assert len(system.contents()) == 1
        assert system.db_path().exists(), "store must be a real file on disk"

        first = system.db_path()
        system.reset()
        assert system.contents() == [], "state must not survive reset"
        assert system.db_path() != first, "each question gets its own file"


def test_retrieving_systems_declare_their_pins_and_others_declare_none():
    for cls in (NoMemorySystem, FullHistorySystem, OracleSystem):
        assert cls().retrieval_pins() == {}, cls

    from mnimi.memory import embed_template_hash

    naive = _naive().retrieval_pins()
    assert naive == {
        "embedder_name": "hashing",
        "embedder_dim": 256,
        "embedder_revision": "v1",
        "embed_template_hash": embed_template_hash(),
        "k": 10,
    }

    mnimi = _mnimi(config=MemoryConfig(top_k=4, dedup_cosine_threshold=0.9)).retrieval_pins()
    assert mnimi["k"] == 4
    assert mnimi["dedup_cosine_threshold"] == 0.9
    assert mnimi["embedder_name"] == "hashing"


def test_retrieval_pins_carry_the_embedder_revision_constant():
    """The declared revision must be the pinned constant on the embedder class,
    read without construction and without network — a resolver could pin
    whatever the hub currently serves instead of what actually ran."""
    from mnimi.embeddings import BgeSmallEmbedder

    for system in (_naive(), _mnimi()):
        assert system.retrieval_pins()["embedder_revision"] == HashingEmbedder.revision

    # The run path hardcodes BGE; its revision is a static class attribute (a
    # full 40-hex HF commit sha), so reading it can never trigger a download.
    assert isinstance(BgeSmallEmbedder.__dict__["revision"], str)
    assert len(BgeSmallEmbedder.revision) == 40
    assert set(BgeSmallEmbedder.revision) <= set("0123456789abcdef")


def test_retrieved_context_is_time_ordered_oldest_first():
    """A reader handed dated blocks in relevance order cannot recover the
    chronology, and temporal reasoning is 27% of the benchmark."""
    for system in (_naive(), _mnimi()):
        system.reset()
        system.add(_round("I adopted a dog", "lovely", ts="2023-07-01"))
        system.add(_round("I moved to Athens", "noted", ts="2023-01-15"))
        system.add(_round("I started a new job", "congratulations", ts="2023-04-20"))

        context = system.get_context("what happened?")
        dates = [line.split("]")[0].split(": ")[1] for line in context.splitlines() if "[" in line]
        assert dates == sorted(dates), f"{system.name}: {dates}"
        assert dates == ["2023-01-15", "2023-04-20", "2023-07-01"], system.name


def test_session_dates_reach_the_reader_for_every_context_bearing_system():
    """Temporal questions are unanswerable if the dates never reach the reader."""
    for system in (FullHistorySystem(), OracleSystem(), _naive(), _mnimi()):
        system.reset()
        system.add(_round("I moved to Thessaloniki", "noted", ts="2023-07-01"))
        assert "2023-07-01" in system.get_context("when did I move?"), system.name


def test_context_format_parity_across_all_context_bearing_arms():
    """One renderer, one code path: the same turns fed to full_history, oracle,
    naive_rag and mnimi must render BYTE-IDENTICAL reader context — speaker
    labels and the dataset's full timestamp everywhere. Divergent formats were
    a confound in every cross-arm comparison: the mnimi-oracle gap mixed
    retrieval quality with date granularity and role labelling."""
    ts_a = "2023/05/20 (Sat) 02:21"
    ts_b = "2023/07/01 (Sat) 14:05"
    session_a = _round("I moved to Athens", "noted, sounds lovely", ts=ts_a)
    session_b = _round("I adopted a dog", "congratulations", ts=ts_b)

    # top_k above the record count so retrieval returns everything and the
    # comparison isolates rendering, not ranking.
    config = MemoryConfig(top_k=10)
    systems = [FullHistorySystem(), OracleSystem(), _naive(config=config), _mnimi(config=config)]
    contexts = []
    for system in systems:
        system.reset()
        system.add(session_a)
        system.add(session_b)
        contexts.append(system.get_context("what happened?"))

    assert contexts[0] == (
        f"[Session date: {ts_a}]\n"
        "user: I moved to Athens\n"
        "assistant: noted, sounds lovely\n"
        f"[Session date: {ts_b}]\n"
        "user: I adopted a dog\n"
        "assistant: congratulations"
    )
    assert len(set(contexts)) == 1, {
        s.name: c for s, c in zip(systems, contexts, strict=True)
    }
