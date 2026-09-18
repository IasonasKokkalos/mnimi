"""The three Phase C systems: oracle (evidence-availability bound), naive_rag (bar), mnimi.

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
from mnimi.embeddings import BGE_QUERY_INSTRUCTION, HashingEmbedder

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
    oracle must not differ from the baseline by its renderer."""
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

    from mnimi.memory import embed_template_hash, render_unit_template_hash

    naive = _naive().retrieval_pins()
    assert naive == {
        "embedder_name": "hashing",
        "embedder_dim": 256,
        "embedder_revision": "v1",
        "embed_template_hash": embed_template_hash(),
        "k": 10,
        "query_instruction": BGE_QUERY_INSTRUCTION,  # shared with mnimi; default since R5
        "chunk_tokens": 0,  # and the embedded unit (R4): granularity parity
        "chunk_overlap": 64,
        # PHASE2: never extracts, so it declares the unit it effectively renders.
        "render_unit": "turns",
        "render_unit_template_hash": render_unit_template_hash("turns"),
    }

    mnimi = _mnimi(config=MemoryConfig(top_k=4, dedup_cosine_threshold=0.9)).retrieval_pins()
    assert mnimi["k"] == 4
    assert mnimi["dedup_cosine_threshold"] == 0.9
    assert mnimi["embedder_name"] == "hashing"
    # A mnimi arm without an extractor SAYS so — the store's guard rows carry the same value.
    assert mnimi["extractor_model"] == "none" and mnimi["extractor_prompt_hash"] == "none"
    # The library default since v1.9.0 (gate 4-iii); the unit is declared even without facts.
    assert mnimi["render_unit"] == "round+facts" and len(mnimi["prefilter_lexicon_hash"]) == 64
    assert mnimi["resolver_version"] == "v1" and len(mnimi["fact_embed_template_hash"]) == 64


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


# -- the extraction era (PHASE2 Task 6) ---------------------------------------------------


def test_with_an_extractor_mnimi_stores_facts_and_naive_rag_does_not(tmp_path):
    from mnimi.extract.fake import RuleExtractor

    messages = [
        *_round("I adopted a cat named Miso. Any tips?", "Lovely!", ts="2023-05-20"),
        *_round("the talk covered sqlite virtual tables", "a powerful extension point"),
    ]
    naive = _naive()
    plain = _mnimi()
    extracted = MnimiSystem(embedder=HashingEmbedder(), extractor=RuleExtractor(),
                            extraction_cache=tmp_path / "x.sqlite")
    for system in (naive, plain, extracted):
        system.reset()
        system.add(messages)
    # naive_rag and the v1 arm are unchanged; the extraction arm's ROUND records
    # are the same strings plus its fact records beside them.
    assert naive.contents() == plain.contents() and len(naive.contents()) == 2
    store = extracted._memory.store
    assert store.contents("eval", kind="round") == naive.contents()
    assert store.contents("eval", kind="fact") == [
        "user: I adopted a cat named Miso.\nThe user adopted a cat named Miso."
    ]
    # The sqlite round has no slot cue and never reaches the model (pre-filter).
    assert extracted.extraction_stats["rounds_sent"] == 1
    assert extracted.extraction_stats["prefilter_skips"] == 1
    assert extracted.cache_stats == {"hits": 0, "misses": 1} and plain.cache_stats is None
    # The pins say what made the facts; the v1 arm says "none".
    pins = extracted.retrieval_pins()
    assert pins["extractor_model"] == "fake-rule"
    assert plain.retrieval_pins()["extractor_model"] == "none"
    assert pins["k"] == plain.retrieval_pins()["k"] == naive.retrieval_pins()["k"]
    # A second reset + add is served from the cache: no extractor call.
    extracted.reset()
    extracted.add(messages)
    assert extracted.cache_stats == {"hits": 1, "misses": 1}


def test_render_unit_reaches_mnimi_and_not_the_other_arms(tmp_path):
    from mnimi.extract.fake import RuleExtractor
    from mnimi.memory import render_unit_template_hash

    config = MemoryConfig(render_unit="round+facts")
    naive = _naive(config=config)
    extracted = MnimiSystem(embedder=HashingEmbedder(), config=config, extractor=RuleExtractor(),
                            extraction_cache=tmp_path / "x.sqlite")
    for system in (naive, extracted):
        system.reset()
        system.add(_round("I adopted a cat named Miso.", "Lovely!", ts="2023-05-20"))
    assert "facts:" in extracted.get_context("cat")
    assert "facts:" not in naive.get_context("cat")
    assert extracted.retrieval_pins()["render_unit"] == "round+facts"
    assert naive.retrieval_pins()["render_unit"] == "turns"
    assert extracted.retrieval_pins()["render_unit_template_hash"] == render_unit_template_hash(
        "round+facts"
    )
    # The turns of the round are byte-identical across the two contexts once
    # the facts header is removed: same renderer, same format, one more block.
    lines = [line for line in extracted.get_context("cat").splitlines()
             if not line.startswith(("facts:", "- "))]
    assert "\n".join(lines) == naive.get_context("cat")


# -- Phase 3 Task 2: the pins (schema /9) ----------------------------------------------


def test_mnimi_pins_declare_the_phase3_rows_and_naive_rag_does_not():
    from mnimi.conflict.lexicon import negation_lexicon_hash
    from mnimi.conflict.normalize import conflict_rules_hash

    pins = _mnimi(
        config=MemoryConfig(dedup_entropy_gate=1.5, conflict_resolution=False)
    ).retrieval_pins()
    assert pins["dedup_entropy_gate"] == 1.5 and pins["conflict_resolution"] is False
    assert pins["negation_lexicon_hash"] == negation_lexicon_hash()
    assert pins["conflict_rules_hash"] == conflict_rules_hash()
    default = _mnimi().retrieval_pins()
    assert default["dedup_entropy_gate"] == 2.0 and default["conflict_resolution"] is True
    naive = _naive().retrieval_pins()
    assert "conflict_resolution" not in naive and "negation_lexicon_hash" not in naive


# -- Phase 4 Task 6: the read side's pins and the consolidate wiring --------------------------


def test_mnimi_pins_declare_the_phase4_rows_and_naive_rag_does_not():
    from mnimi.decay import decay_rules_hash

    pins = _mnimi(config=MemoryConfig(ranking="score", active_only=True, decay_floor=0.2),
                  consolidate=True).retrieval_pins()
    assert (pins["ranking"], pins["active_only"], pins["consolidate"]) == ("score", True, True)
    assert pins["decay_floor"] == 0.2 and pins["decay_half_life_days"] == 30.0
    assert pins["salience_weights"] == {"similarity": 1.0, "recency": 0.0}
    assert type(pins["salience_weights"]) is dict, "canonical JSON needs a plain dict"
    assert pins["recall_min_relevance"] == 0.0 and pins["decay_rules_hash"] == decay_rules_hash()
    default = _mnimi().retrieval_pins()
    assert (default["ranking"], default["active_only"], default["consolidate"]) == (
        "score", True, False)
    naive = _naive().retrieval_pins()
    assert not {"ranking", "active_only", "consolidate", "decay_rules_hash"} & set(naive)


def _consolidations(wired: bool) -> list[int]:
    """Consolidate calls seen after each step of a question's life, for one arm."""
    system = _mnimi(consolidate=wired)
    calls: list[str] = []
    system._memory.consolidate = calls.append
    seen = []
    system.add(_round("I planted tomatoes", "Nice."))
    system.add(_round("my violin lesson moved", "Noted.", ts="2023-05-21"))
    system.get_context("tomatoes")
    seen.append(len(calls))
    system.get_context("tomatoes")  # nothing added since: no second pass
    seen.append(len(calls))
    system.add(_round("booked a ferry", "Enjoy.", ts="2023-05-22"))
    system.get_context("ferry")
    seen.append(len(calls))
    system.reset()
    system._memory.consolidate = calls.append
    system.get_context("anything")  # a fresh store, nothing added
    seen.append(len(calls))
    return seen


def test_mnimi_consolidates_once_per_store_before_the_question_only_when_wired():
    assert _consolidations(True) == [1, 1, 2, 2]
    assert _consolidations(False) == [0, 0, 0, 0]
