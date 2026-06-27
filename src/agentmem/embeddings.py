"""Embedding interface and a dependency-light default.

The library promises to import with only ``sqlite-vec`` and ``numpy`` installed,
so the default embedder cannot reach for ``sentence-transformers`` or a hosted
API. It uses signed feature hashing (numpy only): crude, deterministic, and good
enough to exercise the store. Swap in a real model by implementing ``Embedder``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-dimensional vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this embedder produces."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into unit-length vectors."""
        ...


class HashingEmbedder:
    """Deterministic signed-feature-hashing embedder, numpy-only.

    Tokens are hashed (BLAKE2b, not Python's salted ``hash``) into buckets with a
    sign, summed, and L2-normalized. Lexically similar texts land near each other
    under cosine/L2 distance. No training, no deps, fully reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()
