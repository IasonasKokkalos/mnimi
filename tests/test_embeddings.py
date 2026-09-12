"""Embedder contract: the CI-safe hashing default and the pinned ONNX BGE model.

BGE tests need the ``[embed]`` extra plus a ~130MB model download, so they are
marked ``bge`` and skipped in CI — HashingEmbedder stays the CI path.
"""

from __future__ import annotations

import math
import os

import pytest

from mnimi.embeddings import HashingEmbedder


def _has_embed_deps() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


requires_bge = pytest.mark.skipif(
    os.environ.get("CI", "").lower() in {"1", "true"} or not _has_embed_deps(),
    reason="needs the [embed] extra and a model download; HashingEmbedder is the CI path",
)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_is_unit_norm_and_deterministic():
    embedder = HashingEmbedder(dim=64)
    (first,), (second,) = embedder.embed(["hello world"]), embedder.embed(["hello world"])
    assert first == second
    assert math.isclose(sum(x * x for x in first), 1.0, rel_tol=1e-5)


@pytest.mark.bge
@requires_bge
def test_bge_embedder_pins_model_dim_and_revision():
    from mnimi.embeddings import BgeSmallEmbedder

    embedder = BgeSmallEmbedder()
    assert embedder.name == "BAAI/bge-small-en-v1.5"
    assert embedder.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert embedder.dim == 384


@pytest.mark.bge
@requires_bge
def test_bge_embeddings_are_unit_norm_and_semantically_sane():
    from mnimi.embeddings import BgeSmallEmbedder

    embedder = BgeSmallEmbedder()
    vectors = embedder.embed(
        [
            "the cat sat on the mat",
            "a feline rested on the rug",
            "sqlite is an embedded relational database",
        ]
    )
    assert [len(v) for v in vectors] == [384, 384, 384]
    for vector in vectors:
        assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-4)
    # Paraphrases must sit closer than unrelated text — the hashing embedder
    # cannot do this; it is the reason the ONNX model exists.
    assert _cosine(vectors[0], vectors[1]) > _cosine(vectors[0], vectors[2])


# -- split (R4, 2026-09-12) ------------------------------------------------------


def test_split_covers_the_text_with_overlapping_windows():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    pieces = HashingEmbedder().split(text, max_tokens=10, overlap=3)
    assert pieces[0].split() == words[:10]
    assert pieces[1].split() == words[7:17]
    assert pieces[2].split() == words[14:24]
    assert pieces[3].split() == words[21:25], "the tail is covered, never dropped"
    assert HashingEmbedder().split("short text", max_tokens=10, overlap=3) == ["short text"]
    with pytest.raises(ValueError):
        HashingEmbedder().split(text, max_tokens=10, overlap=10)


@pytest.mark.bge
def test_bge_split_respects_the_real_tokenizer():
    if not _has_embed_deps():
        pytest.skip("[embed] extra not installed")
    from mnimi.embeddings import BgeSmallEmbedder

    e = BgeSmallEmbedder()
    text = "hello " * 700
    pieces = e.split(text, max_tokens=510, overlap=64)
    assert len(pieces) >= 2
    assert all(len(e._splitter.encode(p, add_special_tokens=False).ids) <= 510 for p in pieces)
    assert sum(p.count("hello") for p in pieces) >= 700
