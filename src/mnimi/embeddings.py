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
    def name(self) -> str:
        """Model identity, pinned into ``memory_meta`` at DB creation."""
        ...

    @property
    def revision(self) -> str:
        """Immutable model revision (e.g. an HF commit sha), pinned alongside name."""
        ...

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
    under cosine/L2 distance. No training, no deps, fully reproducible. This is
    the CI path; real semantic similarity comes from ``BgeSmallEmbedder``.
    """

    name = "hashing"
    revision = "v1"

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


class BgeSmallEmbedder:
    """``BAAI/bge-small-en-v1.5`` via ONNX Runtime, revision-pinned.

    Hardcoded for the eval per SPEC — no user-facing model choice. Needs the
    ``[embed]`` extra (onnxruntime, tokenizers, huggingface_hub); the core
    import path never touches it. CLS-token pooling per the model card, and
    vectors are unit-normalized here, inside ``embed()`` — the one normalize
    boundary any insert or query path is allowed to use.
    """

    name = "BAAI/bge-small-en-v1.5"
    revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    _MAX_TOKENS = 512

    def __init__(self) -> None:
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "BgeSmallEmbedder needs the [embed] extra: pip install mnimi[embed]"
            ) from exc

        model_path = hf_hub_download(self.name, "onnx/model.onnx", revision=self.revision)
        tokenizer_path = hf_hub_download(self.name, "tokenizer.json", revision=self.revision)
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_names = {node.name for node in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=self._MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
        feed = {name: value for name, value in feed.items() if name in self._input_names}
        (hidden,) = self._session.run([self._output_name], feed)
        cls = hidden[:, 0]  # BGE pools the [CLS] token, not the mean
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (cls / norms).astype(np.float32).tolist()
