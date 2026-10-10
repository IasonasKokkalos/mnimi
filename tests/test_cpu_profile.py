"""The CPU extractor recipe (MERGED-PLAN T5, LAUNCH M7): no model, no library change.

The recipe is ``decode={**DECODE, "n_gpu_layers": 0}`` passed to the public
``QwenLlamaExtractor(decode=)`` keyword, exposed once as ``examples/chat.py``'s
``CPU_DECODE``. It yields its own ``extractor_decode_hash``, so a store built on
the CPU is a different, self-identifying configuration: the ``memory_meta`` guard
refuses to open it under the GPU pins and vice versa. Measured 2026-10-10 on an
AMD Ryzen 7 PRO 8845HS with the CPU wheel of llama-cpp-python 0.3.35:
``flash_attn=True`` holds on the CPU build (``llama_context: flash_attn = enabled``),
so exactly one key differs from ``DECODE``.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from mnimi.embeddings import HashingEmbedder
from mnimi.extract.fake import RuleExtractor
from mnimi.extract.llama import DECODE, extractor_decode_hash
from mnimi.memory import Memory
from mnimi.store import MemoryMetaError

#: The two hashes as measured (DECODE's is the paper's pin; the CPU one is the recipe's).
GPU_DECODE_HASH = "ba1a81ab349dd09ed5fb77d49b4d44e4a2835b8e05a7873808883999171b8241"
CPU_DECODE_HASH = "40552b1e3de03253781d8a48abfcfd7111b355da6c568ccfe5919075e92f6d65"


@pytest.fixture(scope="module")
def chat():
    spec = importlib.util.spec_from_file_location(
        "mnimi_example_chat_cpu", pathlib.Path("examples/chat.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PinnedExtractor(RuleExtractor):
    """A model-free extractor wearing the real decode hash of one profile."""

    def __init__(self, decode_hash: str) -> None:
        super().__init__()
        self._decode_hash = decode_hash

    @property
    def pins(self) -> dict:
        return {**super().pins, "extractor_decode_hash": self._decode_hash}


def test_cpu_decode_changes_exactly_one_key_of_the_pinned_decode(chat):
    assert chat.CPU_DECODE == {**DECODE, "n_gpu_layers": 0}
    changed = {k for k in DECODE if chat.CPU_DECODE[k] != DECODE[k]}
    assert changed == {"n_gpu_layers"}, "flash_attn held on the CPU build (measured)"
    assert chat.CPU_DECODE["flash_attn"] is True


def test_cpu_decode_hash_differs_from_the_gpu_pin_and_both_are_the_measured_values(chat):
    assert extractor_decode_hash(DECODE) == GPU_DECODE_HASH
    assert extractor_decode_hash(chat.CPU_DECODE) == CPU_DECODE_HASH
    assert extractor_decode_hash(chat.CPU_DECODE) != extractor_decode_hash(DECODE)


def test_a_cpu_store_is_refused_under_the_gpu_pins_and_vice_versa(tmp_path):
    gpu, cpu = _PinnedExtractor(GPU_DECODE_HASH), _PinnedExtractor(CPU_DECODE_HASH)
    message = [{"role": "user", "content": "I live in Boston.", "ts": "2026-10-10T00:00:00"}]

    a = Memory(str(tmp_path / "gpu.db"), HashingEmbedder(), extractor=gpu)
    a.add(message, "u")
    a.store.close()
    with pytest.raises(MemoryMetaError, match="extractor_decode_hash"):
        Memory(str(tmp_path / "gpu.db"), HashingEmbedder(), extractor=cpu)
    Memory(str(tmp_path / "gpu.db"), HashingEmbedder(), extractor=gpu).store.close()

    b = Memory(str(tmp_path / "cpu.db"), HashingEmbedder(), extractor=cpu)
    b.add(message, "u")
    b.store.close()
    with pytest.raises(MemoryMetaError, match="extractor_decode_hash"):
        Memory(str(tmp_path / "cpu.db"), HashingEmbedder(), extractor=gpu)
    Memory(str(tmp_path / "cpu.db"), HashingEmbedder(), extractor=cpu).store.close()


def test_chat_cpu_is_the_default_extractor(chat):
    assert chat.DEFAULT_EXTRACTOR == "cpu"
