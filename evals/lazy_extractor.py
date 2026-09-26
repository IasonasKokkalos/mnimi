"""The pinned extractor, constructed on first use (PLAN A15; PHASE8 D10).

``QwenLlamaExtractor`` loads the GGUF and asks for GPU offload in its constructor,
so a store built entirely from the extraction cache — the normal case since the
corpus pass — still needed a CUDA build of llama-cpp-python to *start*. This
wrapper exposes the same five pins (functions of the module's constants, the
schema's grammar and ``llama_cpp.__version__``, which the CPU wheel provides) and
constructs the real extractor only when a cache miss calls ``extract``. It is a
harness object: the library stays frozen.
"""

from __future__ import annotations

import json


def _construct(**kwargs):
    from mnimi.extract.llama import QwenLlamaExtractor

    return QwenLlamaExtractor(**kwargs)


def _pins(**kwargs) -> dict:
    """The real class's ``pins`` without constructing it: the same constants, the same hashes."""
    import llama_cpp

    from mnimi.extract import llama as llama_mod
    from mnimi.extract.prompt import extractor_prompt_hash
    from mnimi.extract.schema import FACT_SCHEMA

    repo = kwargs.get("model_repo", llama_mod.MODEL_REPO)
    model_file = kwargs.get("model_file", llama_mod.MODEL_FILE)
    revision = kwargs.get("revision", llama_mod.MODEL_REVISION)
    quant = kwargs.get("quant", llama_mod.MODEL_QUANT)
    decode = dict(kwargs.get("decode", llama_mod.DECODE))
    gbnf = llama_cpp.llama_grammar.json_schema_to_gbnf(json.dumps(FACT_SCHEMA))
    return {
        "extractor_model": f"{repo}/{model_file}@{revision}",
        "extractor_quant": quant,
        "extractor_runtime": (
            f"llama-cpp-python {llama_cpp.__version__}; {llama_mod.BUILD_CMAKE_ARGS}; "
            f"cuda {llama_mod.BUILD_CUDA}"
        ),
        "extractor_decode_hash": llama_mod.extractor_decode_hash(decode),
        "extractor_prompt_hash": extractor_prompt_hash(gbnf),
    }


class LazyQwenExtractor:
    """``QwenLlamaExtractor``'s pins now, its model on the first cache miss."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = dict(kwargs)
        self._inner = None
        self._pins = _pins(**self._kwargs)

    @property
    def pins(self) -> dict:
        return dict(self._pins)

    @property
    def inner(self):
        if self._inner is None:
            self._inner = _construct(**self._kwargs)
        return self._inner

    def extract(self, turns: list[dict]):
        return self.inner.extract(turns)
