"""The store does a real insert and a real vector search."""

from __future__ import annotations

import pytest

from mnimi.embeddings import HashingEmbedder
from mnimi.models import MemoryRecord
from mnimi.store import MemoryMetaError, Store


def _open(path, dim=256, name="hashing", revision="v1") -> Store:
    return Store(str(path), dim=dim, embedder_name=name, embedder_revision=revision)


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


def test_unguarded_database_is_refused(tmp_path):
    db = tmp_path / "mem.db"
    store = _open(db)
    store.db.execute("DROP TABLE memory_meta")
    store.db.commit()
    store.close()

    with pytest.raises(MemoryMetaError, match="memory_meta"):
        _open(db)


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
            MemoryRecord(user_id="u1", content=text, embedding=vector, source="test")
        )

    assert store.count("u1") == 3

    # Query overlaps heavily with the first record and not the others.
    (query_vec,) = embedder.embed(["sqlite vector database search"])
    results = store.search(query_vec, user_id="u1", k=3)

    assert results, "expected at least one result"
    assert "sqlite" in results[0].content  # nearest neighbour is the db/search record
    assert results[0].id is not None


def test_search_is_scoped_to_user(tmp_path):
    embedder = HashingEmbedder(dim=128)
    store = _open(tmp_path / "mem.db", dim=embedder.dim)

    (shared_vec,) = embedder.embed(["a shared note about vector databases"])
    store.insert(
        MemoryRecord(user_id="alice", content="alice: vector databases", embedding=shared_vec)
    )
    store.insert(
        MemoryRecord(user_id="bob", content="bob: vector databases", embedding=shared_vec)
    )

    (query_vec,) = embedder.embed(["vector databases"])
    results = store.search(query_vec, user_id="alice", k=5)

    assert results
    assert all(r.user_id == "alice" for r in results)
