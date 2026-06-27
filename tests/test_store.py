"""The store does a real insert and a real vector search."""

from __future__ import annotations

from agentmem.embeddings import HashingEmbedder
from agentmem.models import MemoryRecord
from agentmem.store import Store


def test_insert_and_nearest_neighbor_is_sane(tmp_path):
    embedder = HashingEmbedder(dim=256)
    store = Store(str(tmp_path / "mem.db"), dim=embedder.dim)

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
    store = Store(str(tmp_path / "mem.db"), dim=embedder.dim)

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
