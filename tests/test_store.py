"""The store does a real insert and a real vector search."""

from __future__ import annotations

import dataclasses

import pytest

from mnimi.embeddings import HashingEmbedder
from mnimi.models import MemoryRecord
from mnimi.store import MemoryMetaError, Store


def _guard(extractor=None) -> dict:
    # The extraction-era guard rows (PHASE2 D11), from the library's one definition.
    from mnimi.memory import guard_kwargs

    return guard_kwargs(extractor)


def _open(path, dim=256, name="hashing", revision="v1", template_hash="tmpl-a",
          extractor=None, **overrides) -> Store:
    return Store(
        str(path),
        dim=dim,
        embedder_name=name,
        embedder_revision=revision,
        embed_template_hash=template_hash,
        **{**_guard(extractor), **overrides},
    )


def test_meta_guard_accepts_matching_reopen(tmp_path):
    db = tmp_path / "mem.db"
    store = _open(db)
    store.insert(
        MemoryRecord(user_id="u1", content="a fact", embedding=None, created_at="2023-05-20")
    )
    store.close()

    reopened = _open(db)
    assert reopened.count("u1") == 1


def test_meta_guard_rejects_dim_mismatch_at_open_time(tmp_path):
    db = tmp_path / "mem.db"
    _open(db, dim=256).close()

    with pytest.raises(MemoryMetaError, match="embedder_dim"):
        _open(db, dim=384)


def test_meta_guard_rejects_name_mismatch(tmp_path):
    db = tmp_path / "mem.db"
    _open(db, name="hashing").close()

    with pytest.raises(MemoryMetaError, match="embedder_name"):
        _open(db, name="BAAI/bge-small-en-v1.5")


def test_meta_guard_rejects_revision_mismatch(tmp_path):
    db = tmp_path / "mem.db"
    _open(db, revision="v1").close()

    with pytest.raises(MemoryMetaError, match="embedder_revision"):
        _open(db, revision="v2")


def test_meta_guard_rejects_embed_template_mismatch(tmp_path):
    """An embed-template edit moves every vector while leaving the embedder
    pins untouched — the guard must refuse the reopen, closing what was a
    documented invariant with no enforcement."""
    db = tmp_path / "mem.db"
    _open(db, template_hash="tmpl-a").close()

    with pytest.raises(MemoryMetaError, match="embed_template_hash"):
        _open(db, template_hash="tmpl-b")


def test_record_turns_roundtrip_through_the_store(tmp_path):
    store = _open(tmp_path / "mem.db")
    turns = [
        {"role": "user", "content": "line one\nline two"},
        {"role": "assistant", "content": "reply with $pecial {chars}"},
    ]
    store.insert(
        MemoryRecord(
            user_id="u1", content="embed text", embedding=[0.0] * 256,
            created_at="2023-05-20", turns=turns,
        )
    )
    [(record, _cos)] = store.search([0.0] * 256, user_id="u1", k=1)
    assert record.turns == turns


def test_unguarded_database_is_refused(tmp_path):
    db = tmp_path / "mem.db"
    store = _open(db)
    store.db.execute("DROP TABLE memory_meta")
    store.db.commit()
    store.close()

    with pytest.raises(MemoryMetaError, match="memory_meta"):
        _open(db)


def test_search_surfaces_cosine_similarity(tmp_path):
    """L2 distance converts to cosine at one boundary: cos = 1 - d^2/2 for unit vectors."""
    store = _open(tmp_path / "mem.db", dim=2)
    for content, vector in [("aligned", [1.0, 0.0]), ("offset", [0.6, 0.8])]:
        store.insert(
            MemoryRecord(
                user_id="u1", content=content, embedding=vector, created_at="2023-05-20"
            )
        )

    results = store.search([1.0, 0.0], user_id="u1", k=2)

    (first, cos_first), (second, cos_second) = results
    assert first.content == "aligned"
    assert cos_first == pytest.approx(1.0, abs=1e-6)
    assert second.content == "offset"
    assert cos_second == pytest.approx(0.6, abs=1e-6)


def test_pinned_is_gone_but_salience_and_supersedes_stay(tmp_path):
    """`pinned` is spec'd dead (no producer). salience/supersedes stay, inert —
    there is no migration system, so re-adding a column later means a
    hand-written ALTER."""
    fields = {field.name for field in dataclasses.fields(MemoryRecord)}
    assert "pinned" not in fields
    assert {"salience", "supersedes"} <= fields

    store = _open(tmp_path / "mem.db")
    columns = {row["name"] for row in store.db.execute("PRAGMA table_info(memories)")}
    assert "pinned" not in columns
    assert {"salience", "supersedes"} <= columns


def test_insert_refuses_a_record_without_ts(tmp_path):
    """ts is the only clock. The store never reads wall-clock — a record with no
    timestamp is a caller bug, not something to paper over with now()."""
    store = _open(tmp_path / "mem.db")
    with pytest.raises(ValueError, match="ts"):
        store.insert(MemoryRecord(user_id="u1", content="undated", embedding=None))


def test_insert_keeps_the_injected_ts_verbatim(tmp_path):
    store = _open(tmp_path / "mem.db")
    stored = store.insert(
        MemoryRecord(user_id="u1", content="dated", embedding=None, created_at="2023-05-20")
    )
    (value,) = store.db.execute(
        "SELECT created_at FROM memories WHERE id = ?", (stored.id,)
    ).fetchone()
    assert value == "2023-05-20"


def test_vec0_ddl_pins_distance_metric_explicitly(tmp_path):
    store = _open(tmp_path / "mem.db")
    (ddl,) = store.db.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_memories'"
    ).fetchone()
    assert "distance_metric=L2" in ddl


def test_insert_and_nearest_neighbor_is_sane(tmp_path):
    embedder = HashingEmbedder(dim=256)
    store = _open(tmp_path / "mem.db", dim=embedder.dim)

    texts = [
        "python sqlite database vector search engine",
        "the golden retriever puppy chased a ball across the park",
        "a recipe for baking sourdough bread with yeast and flour",
    ]
    for text, vector in zip(texts, embedder.embed(texts), strict=True):
        store.insert(
            MemoryRecord(
                user_id="u1", content=text, embedding=vector, created_at="2023-05-20", source="test"
            )
        )

    assert store.count("u1") == 3

    # Query overlaps heavily with the first record and not the others.
    (query_vec,) = embedder.embed(["sqlite vector database search"])
    results = store.search(query_vec, user_id="u1", k=3)

    assert results, "expected at least one result"
    nearest, _ = results[0]
    assert "sqlite" in nearest.content  # nearest neighbour is the db/search record
    assert nearest.id is not None


def test_search_is_scoped_to_user(tmp_path):
    embedder = HashingEmbedder(dim=128)
    store = _open(tmp_path / "mem.db", dim=embedder.dim)

    (shared_vec,) = embedder.embed(["a shared note about vector databases"])
    for user in ("alice", "bob"):
        store.insert(
            MemoryRecord(
                user_id=user,
                content=f"{user}: vector databases",
                embedding=shared_vec,
                created_at="2023-05-20",
            )
        )

    (query_vec,) = embedder.embed(["vector databases"])
    results = store.search(query_vec, user_id="alice", k=5)

    assert results
    assert all(record.user_id == "alice" for record, _ in results)


def test_meta_guard_rejects_chunking_mismatch(tmp_path):
    from mnimi.memory import embed_template_hash

    kwargs = dict(dim=4, embedder_name="e", embedder_revision="r",
                  embed_template_hash=embed_template_hash(), **_guard())
    Store(str(tmp_path / "s.db"), **kwargs).close()
    Store(str(tmp_path / "s.db"), **kwargs).close()  # same chunking reopens
    with pytest.raises(MemoryMetaError, match="chunk_tokens"):
        Store(str(tmp_path / "s.db"), chunk_tokens=64, **kwargs)
    with pytest.raises(MemoryMetaError, match="chunk_overlap"):
        Store(str(tmp_path / "s.db"), chunk_overlap=8, **kwargs)


def test_search_rounds_collapses_windows_of_one_round_and_equals_search_otherwise(tmp_path):
    from mnimi.memory import embed_template_hash

    store = Store(str(tmp_path / "r.db"), dim=3, embedder_name="e", embedder_revision="r",
                  embed_template_hash=embed_template_hash(), **_guard())
    def rec(content, vec, key=None):
        return MemoryRecord(user_id="u", content=content, embedding=vec, created_at="2023-05-20",
                            round_key=key)
    # Three windows of round A sit nearest the query; rounds B and C follow.
    store.insert(rec("A1", [1.0, 0.0, 0.0], "A"))
    store.insert(rec("A2", [0.99, 0.1, 0.0], "A"))
    store.insert(rec("A3", [0.98, 0.15, 0.0], "A"))
    store.insert(rec("B", [0.7, 0.7, 0.0]))
    store.insert(rec("C", [0.0, 1.0, 0.0]))
    q = [1.0, 0.0, 0.0]
    windows = [r.content for r, _ in store.search(q, "u", k=3)]
    assert windows == ["A1", "A2", "A3"], "window-level search: one round fills k"
    rounds = [r.content for r, _ in store.search_rounds(q, "u", k=3)]
    assert rounds == ["A1", "B", "C"], "round-level search: A once, at its best window"
    assert [r.content for r, _ in store.search_rounds(q, "u", k=10)] == ["A1", "B", "C"]

    plain = Store(str(tmp_path / "p.db"), dim=3, embedder_name="e", embedder_revision="r",
                  embed_template_hash=embed_template_hash(), **_guard())
    for content, vec in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
        plain.insert(rec(content, vec))
    assert plain.search_rounds(q, "u", k=2) == plain.search(q, "u", k=2), "no windows: identical"


# -- the extraction era (PHASE2 Task 6): guard rows, kinds, fact columns -----------------


def test_meta_guard_requires_the_extraction_era_keys_and_refuses_a_v1_store(tmp_path):
    store = _open(tmp_path / "a.db")
    for key in ("extractor_model", "fact_embed_template_hash", "prefilter_lexicon_hash"):
        assert store.db.execute(
            "SELECT value FROM memory_meta WHERE key = ?", (key,)
        ).fetchone() is not None
    # A v1.8 store never wrote the extraction-era rows: deleting them is what
    # opening such a store looks like, and it is refused, not upgraded.
    store.db.execute("DELETE FROM memory_meta WHERE key LIKE 'extractor_%'")
    store.db.commit()
    store.close()
    with pytest.raises(MemoryMetaError, match="extractor_model"):
        _open(tmp_path / "a.db")


def test_meta_guard_rejects_an_extractor_mismatch_in_both_directions(tmp_path):
    from mnimi.extract.fake import RuleExtractor

    _open(tmp_path / "b.db").close()  # built with no extractor: the rows say "none"
    with pytest.raises(MemoryMetaError, match="extractor_model"):
        _open(tmp_path / "b.db", extractor=RuleExtractor())
    _open(tmp_path / "c.db", extractor=RuleExtractor()).close()
    with pytest.raises(MemoryMetaError, match="extractor_prompt_hash"):
        _open(tmp_path / "c.db")
    _open(tmp_path / "c.db", extractor=RuleExtractor()).close()  # same pins reopen
    with pytest.raises(MemoryMetaError, match="resolver_version"):
        _open(tmp_path / "c.db", extractor=RuleExtractor(), resolver_version="v0")


def test_fact_record_columns_roundtrip_and_kinds_are_searched_apart(tmp_path):
    store = _open(tmp_path / "k.db", dim=3)

    def rec(kind, content, vec, key=None, **fields):
        return MemoryRecord(user_id="u", content=content, embedding=vec, created_at="2023-05-20",
                            round_key=key, kind=kind, **fields)

    store.insert(rec("round", "the round", [1.0, 0.0, 0.0], "R"))
    store.insert(rec("fact", "user: I own a cat\nThe user owns a cat.", [0.99, 0.1, 0.0], "R",
                     fact="The user owns a cat.", raw="user: I own a cat", subject="user",
                     predicate="owns", object="cat", valid_time="2023-05-19",
                     time_mention="yesterday", salience=0.5))
    store.insert(rec("round", "another round", [0.0, 1.0, 0.0]))
    q = [1.0, 0.0, 0.0]
    both = [r.kind for r, _ in store.search(q, "u", k=3)]
    assert both == ["round", "fact", "round"], "unscoped search merges the kinds by distance"
    assert [r.kind for r, _ in store.search(q, "u", k=3, kind="fact")] == ["fact"]
    assert [r.content for r, _ in store.search(q, "u", k=3, kind="round")] == [
        "the round", "another round"
    ]
    (fact, _), = store.search(q, "u", k=1, kind="fact")
    assert (fact.fact, fact.raw, fact.subject, fact.predicate, fact.object) == (
        "The user owns a cat.", "user: I own a cat", "user", "owns", "cat"
    )
    assert (fact.valid_time, fact.time_mention, fact.salience) == ("2023-05-19", "yesterday", 0.5)
    assert store.contents("u", kind="fact") == ["user: I own a cat\nThe user owns a cat."]
    assert store.contents_with_ts("u", kind="round")[0] == ("2023-05-20", "the round")
    assert [f.fact for f in store.facts_of("u", "R")] == ["The user owns a cat."]
    assert store.count("u") == 3 and store.count("u", kind="fact") == 1
    # k counts rounds: the round record and its fact collapse to one hit.
    rounds = [r.content for r, _ in store.search_rounds(q, "u", k=2)]
    assert rounds == ["the round", "another round"]
    with pytest.raises(ValueError):
        store.search(q, "u", k=1, kind="window")
    with pytest.raises(ValueError):
        store.insert(rec("window", "x", [0.0, 0.0, 1.0]))


# -- Phase 3 Task 2: the two guard rows, the pair_key column -------------------------------


def test_meta_guard_requires_the_phase3_rows_and_refuses_a_v19_store(tmp_path):
    from mnimi.conflict.lexicon import negation_lexicon_hash
    from mnimi.conflict.normalize import conflict_rules_hash

    store = _open(tmp_path / "p3.db")
    rows = dict(store.db.execute("SELECT key, value FROM memory_meta").fetchall())
    assert rows["negation_lexicon_hash"] == negation_lexicon_hash() != "none"
    assert rows["conflict_rules_hash"] == conflict_rules_hash()
    # Seventeen rows since Phase 4: the four embedder rows, the R4 pair, five
    # extractor rows, the four extraction-era rows, this one, decay_rules_hash.
    assert len(rows) == 17
    store.db.execute("DELETE FROM memory_meta WHERE key = 'conflict_rules_hash'")
    store.db.execute("UPDATE memory_meta SET value = 'none' WHERE key = 'negation_lexicon_hash'")
    store.db.commit()
    store.close()
    with pytest.raises(MemoryMetaError, match="conflict_rules_hash"):
        _open(tmp_path / "p3.db")


def test_pair_key_column_roundtrips_and_is_indexed(tmp_path):
    store = _open(tmp_path / "pk.db")
    vec = [1.0] + [0.0] * 255
    record = store.insert(MemoryRecord(user_id="u", content="x", embedding=vec,
                                       created_at="2023-05-20", kind="fact",
                                       fact="The user lives in Boston.",
                                       raw="user: I live in Boston.", subject="user",
                                       predicate="lives in", object="Boston",
                                       pair_key="user|lives in"))
    ((back, _cos),) = store.search(vec, "u", k=1, kind="fact")
    assert back.id == record.id and back.pair_key == "user|lives in"
    indexes = {row["name"] for row in store.db.execute("PRAGMA index_list('memories')")}
    assert "idx_memories_pair" in indexes


# -- Phase 3 Task 3: supersede, the pair-index reads ----------------------------------------


def test_supersede_updates_salience_and_the_winner_link(tmp_path):
    store = _open(tmp_path / "s.db")
    vec = [1.0] + [0.0] * 255
    loser = store.insert(MemoryRecord(user_id="u", content="a", embedding=vec,
                                      created_at="2023-01-01", kind="fact", fact="a",
                                      raw="user: a", subject="user", predicate="lives in",
                                      object="Boston", pair_key="user|lives in"))
    winner = store.insert(MemoryRecord(user_id="u", content="b", embedding=vec,
                                       created_at="2023-02-01", kind="fact", fact="b",
                                       raw="user: b", subject="user", predicate="lives in",
                                       object="Seattle", pair_key="user|lives in"))
    assert [r.id for r in store.active_facts_by_pair("u", "user|lives in")] == [loser.id, winner.id]
    store.supersede(loser.id, winner.id)
    (active,) = store.active_facts_by_pair("u", "user|lives in")
    assert active.id == winner.id and active.supersedes == loser.id
    rows = {r.id: r for r in store.facts_with_pair_key("u")}
    assert rows[loser.id].salience == 0.0 and rows[winner.id].salience == 1.0
    store.supersede(loser.id, winner.id)  # idempotent at the store too
    assert store.active_facts_by_pair("u", "user|lives in")[0].supersedes == loser.id
    assert store.active_facts_by_pair("u", "user|other") == []


# -- Phase 4 Task 2: last_accessed, initial_salience, the seventeenth guard row ---------------


def test_meta_guard_requires_decay_rules_hash_and_refuses_a_v110_store(tmp_path):
    from mnimi.decay import decay_rules_hash

    store = _open(tmp_path / "p4.db")
    rows = dict(store.db.execute("SELECT key, value FROM memory_meta").fetchall())
    assert rows["decay_rules_hash"] == decay_rules_hash() and len(rows) == 17
    store.db.execute("DELETE FROM memory_meta WHERE key = 'decay_rules_hash'")
    store.db.commit()
    store.close()
    with pytest.raises(MemoryMetaError, match="decay_rules_hash"):
        _open(tmp_path / "p4.db")


def test_insert_initializes_last_accessed_and_initial_salience(tmp_path):
    store = _open(tmp_path / "la.db")
    vec = [1.0] + [0.0] * 255
    fact = store.insert(MemoryRecord(user_id="u", content="f", embedding=vec,
                                     created_at="2023/05/20 (Sat) 09:00", kind="fact",
                                     fact="f", raw="assistant: f", salience=0.5))
    assert fact.last_accessed == "2023/05/20 (Sat) 09:00" and fact.initial_salience == 0.5
    given = store.insert(MemoryRecord(user_id="u", content="g", embedding=vec,
                                      created_at="2023-05-20", last_accessed="2023-06-01",
                                      salience=0.3, initial_salience=1.0))
    rows = {r.id: r for r in store.active_records("u")}
    assert (rows[fact.id].last_accessed, rows[fact.id].initial_salience) == (
        "2023/05/20 (Sat) 09:00", 0.5)
    assert (rows[given.id].last_accessed, rows[given.id].salience,
            rows[given.id].initial_salience) == ("2023-06-01", 0.3, 1.0)


def test_insert_refuses_a_salience_outside_the_unit_interval(tmp_path):
    store = _open(tmp_path / "range.db")
    for field in ({"salience": 1.5}, {"salience": -0.1}, {"initial_salience": 2.0}):
        with pytest.raises(ValueError, match="salience"):
            store.insert(MemoryRecord(user_id="u", content="x", embedding=None,
                                      created_at="2023-05-20", **field))
    assert store.count("u") == 0


def test_decay_store_reads_and_writes(tmp_path):
    store = _open(tmp_path / "rw.db")
    vec = [1.0] + [0.0] * 255
    a = store.insert(MemoryRecord(user_id="u", content="a", embedding=vec,
                                  created_at="2023/05/20 (Sat) 09:00"))
    b = store.insert(MemoryRecord(user_id="u", content="b", embedding=vec,
                                  created_at="2023/06/01 (Thu) 08:00"))
    c = store.insert(MemoryRecord(user_id="u", content="c", embedding=vec,
                                  created_at="2023/06/01 (Thu) 08:00", salience=0.0))
    store.insert(MemoryRecord(user_id="other", content="d", embedding=vec, created_at="2024-01-01"))
    assert store.created_ats("u") == ["2023/05/20 (Sat) 09:00", "2023/06/01 (Thu) 08:00"]
    assert [r.id for r in store.active_records("u")] == [a.id, b.id], "salience 0 is not active"
    store.set_saliences([(a.id, 0.25), (b.id, 0.5)])
    store.touch([a.id], "2023/06/01 (Thu) 08:00")
    rows = {r.id: r for r in store.active_records("u")}
    assert rows[a.id].salience == 0.25 and rows[a.id].initial_salience == 1.0
    assert rows[a.id].last_accessed == "2023/06/01 (Thu) 08:00"
    assert rows[b.id].salience == 0.5 and rows[b.id].last_accessed == "2023/06/01 (Thu) 08:00"
    store.set_saliences([])
    store.touch([], "2099-01-01")
    assert c.id not in {r.id for r in store.active_records("u")}
def test_search_survives_a_k_whose_overfetch_exceeds_the_vec0_limit(tmp_path):
    # sqlite-vec caps a KNN query at k = 4096 rows; _knn asks for k * 8 to leave
    # room for the user filter, so a caller asking for more than 512 rows used to
    # raise "k value in knn query too large". rank_rounds grows its candidate pool
    # (k, 4k, 16k, ...) and reaches 800 once decayed saliences lower the k-th score
    # (PHASE4 Task 8: the p4d probe died there on question 1).
    store = _open(tmp_path / "knnlimit.db")
    vec = [1.0] + [0.0] * 255
    for i in range(3):
        store.insert(MemoryRecord(user_id="u", content=f"r{i}", embedding=vec,
                                  created_at="2023-05-20"))
    assert len(store.search(vec, "u", k=800)) == 3, "every record the user has, no raise"
    assert len(store.search_rounds(vec, "u", k=800)) == 3
