"""The pinned extractor: Qwen3-1.7B (Q8_0 GGUF) in-process through llama-cpp-python.

Needs the ``[extract]`` extra; nothing here is imported by ``mnimi.extract``
itself. What is pinned (PHASE2 D7, DECISIONS "Phase 2 pre-registration"):

* the artifact — ``Qwen/Qwen3-1.7B-GGUF`` at one HF revision, one file, one
  quant; the download is by revision, like the embedder's;
* the decode — greedy under the grammar, a seed, the context and batch
  sizes, the thread count, full GPU offload, flash attention, the output
  budget, the input cap, ONE sequence and a COLD prefill per round
  (``reset()`` before every call: llama-cpp-python otherwise reuses the
  longest shared prompt prefix, which changes the batch at the logits
  position and makes a round's output depend on the round before it — the
  reader's prompt-cache lesson, and a cache key that would then be a lie);
* the prompt surface — the literal prompt and the GBNF the runtime actually
  samples under, hashed together.

**How the grammar is applied.** llama.cpp's grammar sampler checks every
vocabulary entry against the grammar at every step — measured 2026-09-13:
~55 ms per token over Qwen3's 151k-token vocabulary, 7.5 s of a 10.4 s round,
regardless of the grammar's size. This module samples the way llama-server
does instead: take the greedy token from the raw logits, verify that one
candidate against the grammar, and only when it is rejected rescan the whole
vocabulary under the grammar and take the best valid token. The result is
the same token the grammar-then-greedy chain would pick (the global argmax
is valid ⇒ it is the valid argmax; otherwise the rescan is the chain), at
the cost of one single-candidate check per token.

``n_gpu_layers=99`` is a disclosed deviation from SPEC's CPU-only decode pin;
a build that cannot offload is refused rather than silently run on the CPU,
because that would be a different configuration wearing these pins.
"""

from __future__ import annotations

import ctypes
import json
from types import MappingProxyType

import numpy as np

from .prompt import INPUT_CAP_TOKENS, build_prompt, extractor_prompt_hash
from .protocol import ExtractionResult, canonical, sha256_text
from .schema import FACT_SCHEMA, parse_output

MODEL_REPO = "Qwen/Qwen3-1.7B-GGUF"
MODEL_FILE = "Qwen3-1.7B-Q8_0.gguf"
MODEL_REVISION = "90862c4b9d2787eaed51d12237eafdfe7c5f6077"
MODEL_QUANT = "Q8_0"

#: The build this module was pinned against (DECISIONS "Extractor runtime").
#: Recorded into ``extractor_runtime``; a rebuild with other flags is a new era.
BUILD_CUDA = "13.2"
BUILD_CMAKE_ARGS = "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF"

#: Every value that shapes the sampled tokens. Frozen; hashed as a whole.
DECODE = MappingProxyType(
    {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "seed": 0,
        "n_ctx": 4096,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 8,
        "n_gpu_layers": 99,
        "flash_attn": True,
        "max_tokens": 1024,
        "input_cap_tokens": INPUT_CAP_TOKENS,
        "sequences": 1,
        "cold_prefill": True,
        "grammar_mode": "greedy-verify-rescan",
    }
)

_NEG_INF = float("-inf")


def extractor_decode_hash(decode=DECODE) -> str:
    return sha256_text(canonical(dict(decode)))


def _require():
    try:
        import llama_cpp
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "QwenLlamaExtractor needs the [extract] extra: pip install mnimi[extract] "
            "(llama-cpp-python built with CUDA, see docs/DECISIONS.md 'Extractor runtime')"
        ) from exc
    return llama_cpp, hf_hub_download


class QwenLlamaExtractor:
    """SPEC's one LLM, pinned end to end."""

    def __init__(
        self,
        *,
        model_repo: str = MODEL_REPO,
        model_file: str = MODEL_FILE,
        revision: str = MODEL_REVISION,
        quant: str = MODEL_QUANT,
        decode=DECODE,
        verbose: bool = False,
    ) -> None:
        llama_cpp, hf_hub_download = _require()
        self._lc = llama_cpp
        self._decode = dict(decode)
        self._model_repo, self._model_file, self._revision, self._quant = (
            model_repo, model_file, revision, quant,
        )
        if self._decode["n_gpu_layers"] and not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError(
                "this llama-cpp-python build cannot offload to the GPU, but the decode pins "
                "ask for n_gpu_layers>0; rebuild with GGML_CUDA=ON (or pin n_gpu_layers=0 as "
                "a new era) rather than run a different configuration under these pins"
            )
        path = hf_hub_download(model_repo, model_file, revision=revision)
        self._llm = llama_cpp.Llama(
            model_path=path,
            n_ctx=self._decode["n_ctx"],
            n_batch=self._decode["n_batch"],
            n_ubatch=self._decode["n_ubatch"],
            n_threads=self._decode["n_threads"],
            n_threads_batch=self._decode["n_threads"],
            n_gpu_layers=self._decode["n_gpu_layers"],
            seed=self._decode["seed"],
            flash_attn=self._decode["flash_attn"],
            logits_all=False,
            verbose=verbose,
        )
        # The GBNF the runtime derives from the schema is what is hashed: the
        # derivation belongs to the runtime, so the text is recorded, not trusted.
        self._gbnf = llama_cpp.llama_grammar.json_schema_to_gbnf(json.dumps(FACT_SCHEMA))
        self._prompt_hash = extractor_prompt_hash(self._gbnf)
        self._vocab = llama_cpp.llama_model_get_vocab(self._llm._model.model)
        self._n_vocab = int(llama_cpp.llama_vocab_n_tokens(self._vocab))
        self._grammar = llama_cpp.llama_sampler_init_grammar(
            self._vocab, self._gbnf.encode("utf-8"), b"root"
        )
        if not self._grammar:
            raise RuntimeError("llama.cpp refused the fact grammar")
        # Reused per step: a one-candidate array for the verify pass and a
        # full-vocabulary array for the rare rescan.
        self._one = (llama_cpp.llama_token_data * 1)()
        self._all = np.zeros(
            self._n_vocab, dtype=[("id", "<i4"), ("logit", "<f4"), ("p", "<f4")]
        )
        self._all["id"] = np.arange(self._n_vocab, dtype=np.int32)
        self._runtime = (
            f"llama-cpp-python {llama_cpp.__version__}; {BUILD_CMAKE_ARGS}; cuda {BUILD_CUDA}"
        )
        self.rescans = 0  # diagnostic: how often the greedy token was grammar-invalid

    def __del__(self) -> None:  # pragma: no cover - resource release
        grammar = getattr(self, "_grammar", None)
        if grammar:
            try:
                self._lc.llama_sampler_free(grammar)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass
            self._grammar = None

    @property
    def grammar_text(self) -> str:
        return self._gbnf

    @property
    def pins(self) -> dict:
        return {
            "extractor_model": f"{self._model_repo}/{self._model_file}@{self._revision}",
            "extractor_quant": self._quant,
            "extractor_runtime": self._runtime,
            "extractor_decode_hash": extractor_decode_hash(self._decode),
            "extractor_prompt_hash": self._prompt_hash,
        }

    def _cap(self, turns: list[dict]) -> tuple[list[dict], bool]:
        """Round text capped at ``input_cap_tokens`` of the model's own tokens.

        Turns are allocated in order (the user turn first in a round), each
        cut at a token boundary when the remaining budget runs out.
        """
        budget = int(self._decode["input_cap_tokens"])
        capped: list[dict] = []
        truncated = False
        for turn in turns:
            content = str(turn.get("content", "")).strip()
            ids = self._llm.tokenize(content.encode("utf-8"), add_bos=False, special=False)
            if len(ids) > budget:
                truncated = True
                ids = ids[:budget]
                content = self._llm.detokenize(ids).decode("utf-8", errors="ignore").strip()
            budget -= len(ids)
            capped.append({"role": turn.get("role", ""), "content": content})
        return capped, truncated

    def _grammar_allows(self, token: int) -> bool:
        """Verify one candidate against the grammar without advancing it."""
        lc = self._lc
        self._one[0].id = token
        self._one[0].logit = 0.0
        self._one[0].p = 0.0
        arr = lc.llama_token_data_array(data=self._one, size=1, selected=-1, sorted=False)
        lc.llama_sampler_apply(self._grammar, ctypes.byref(arr))
        return self._one[0].logit != _NEG_INF

    def _best_valid(self, logits: np.ndarray) -> int | None:
        """The highest-logit grammar-valid token, scanning the whole vocabulary."""
        lc = self._lc
        self._all["logit"] = logits
        self._all["p"] = 0.0
        arr = lc.llama_token_data_array(
            data=self._all.ctypes.data_as(ctypes.POINTER(lc.llama_token_data)),
            size=self._n_vocab, selected=-1, sorted=False,
        )
        lc.llama_sampler_apply(self._grammar, ctypes.byref(arr))
        valid = self._all["logit"]
        best = int(np.argmax(valid))  # the first maximum, like llama.cpp's greedy sampler
        return None if valid[best] == _NEG_INF else int(self._all["id"][best])

    def _generate(self, prompt_tokens: list[int]) -> tuple[list[int], str]:
        """Greedy decoding under the grammar; ``(tokens, finish_reason)``."""
        lc, llm = self._lc, self._llm
        llm.reset()  # cold prefill, every round (see the module docstring)
        llm.eval(prompt_tokens)
        lc.llama_sampler_reset(self._grammar)
        out: list[int] = []
        max_tokens = int(self._decode["max_tokens"])
        while True:
            logits = np.ctypeslib.as_array(
                lc.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(self._n_vocab,)
            )
            token = int(np.argmax(logits))
            if not self._grammar_allows(token):
                self.rescans += 1
                token = self._best_valid(logits)
                if token is None:
                    return out, "grammar"  # nothing valid: treated as a truncated output
            lc.llama_sampler_accept(self._grammar, token)
            if lc.llama_vocab_is_eog(self._vocab, token):
                return out, "stop"
            out.append(token)
            if len(out) >= max_tokens:
                return out, "length"
            llm.eval([token])

    def extract(self, turns: list[dict]) -> ExtractionResult:
        capped, truncated_input = self._cap(turns)
        prompt_text = build_prompt(capped)
        tokens = self._llm.tokenize(prompt_text.encode("utf-8"), add_bos=False, special=True)
        out_tokens, finish = self._generate(tokens)
        text = self._llm.detokenize(out_tokens).decode("utf-8", errors="replace")
        facts, bad_parse = parse_output(text)
        truncated = bad_parse or finish != "stop"
        if truncated:
            facts = []
        # Diagnostics for the throughput probe; never part of the result or the cache.
        self.last_truncated_input = truncated_input
        self.last_usage = {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(out_tokens),
            "total_tokens": len(tokens) + len(out_tokens),
            "finish_reason": finish,
        }
        return ExtractionResult(facts=facts, truncated=truncated, raw_output=text,
                                truncated_input=truncated_input)
