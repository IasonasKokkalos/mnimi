# DECISIONS

Locked choices and why. Append-only; supersede, don't delete.

- **Language = Python.** Embeddable, ubiquitous in the agent ecosystem.
- **Storage = sqlite-vec.** Local-first, zero-infra, one file on disk. No hosted
  vector DB, no server process.
- **Core deps = `sqlite-vec` + `numpy`, and only those.** Every core dep is a
  liability the library forces on its users. The default embedder is therefore a
  numpy-only hashing embedder — no `sentence-transformers`, no API calls — so the
  library imports clean.
- **Benchmark = LongMemEval (`longmemeval_s`, ~500 questions).** The harness is
  the source of truth for every claim.
- **Baselines = no-memory, full-history, naive-RAG.** Floor, ceiling, and the bar
  mnimi must clear.
- **Eval loader uses `huggingface-hub`, not `datasets`.** The dataset's nested
  `haystack_sessions` breaks the HF Arrow viewer; we download the raw
  `longmemeval_s_cleaned.json` via `hf_hub_download` and parse with stdlib `json`.
  Leaner extra, no Arrow dependency, no schema-casting surprises.
- **Reader + judge = Anthropic API.** Key from `ANTHROPIC_API_KEY`. The judge
  follows the LongMemEval paper's per-question-type prompts (incl. abstention).
  *(Superseded: reader is now local Ollama, judge is `gpt-4o-2024-08-06`.)*

## Reader decode config: a wrong finding and its correction (2026-07-26)

Kept in full, including the wrong conclusion. A self-overturned finding
documents a failure mode the next person would otherwise repeat.

**First conclusion (WRONG): "GPU inference is non-deterministic; the reader
must be CPU-only."** Evidence was real: at `temperature=0`, three identical
GPU calls returned two distinct answers, and answers diverged mid-sentence
after an identical prefix — the classic signature of logits shifting enough to
flip greedy decoding at a near-tie. This was attributed to CUDA atomics
reordering floating-point reductions, i.e. an unfixable hardware property.

**Why it was wrong.** That test never pinned `num_batch`. Ollama picked a batch
size per load, and llama.cpp logits are not bit-identical across batch sizes.
The drift came from an *unpinned harness setting*, not the GPU. An unpinned
batch size produces exactly the signature attributed to atomics, which is what
made the misdiagnosis easy — and is the reason the correction is recorded
rather than quietly overwritten.

**Re-test with `num_batch=512` pinned**, on a real dataset question, comparing
byte-for-byte:

| Condition | Result |
|---|---|
| 3 calls, same loaded instance | byte-identical |
| Cold model reload | byte-identical |
| Full machine reboot | byte-identical |
| 91% GPU utilisation, concurrent load | byte-identical |

All 12 generations hashed to the same value. GPU is also ~23x faster on a 32k
prefill (10.4s vs 237.2s CPU), so **full offload (`num_gpu=99`) is the pinned
default**.

**Load-bearing pins:** `num_batch` (the pin whose absence caused the wrong
finding), `top_k=1` (temperature=0 alone is not greedy), `seed`, `num_gpu`.
`num_thread` was measured to have **no** effect at full offload (4/8/16
byte-identical) and is retained only because `--num-gpu 0` returns the reader
to CPU, where it *is* load-bearing.

**Open, unresolved:** an identical request under an identical logged
configuration (same `n_batch`/`n_ubatch`/`n_ctx`, flash attention enabled,
29/29 layers offloaded, same model digest, same `cuda_v13` runtime and driver)
produced different output *after* the model store was moved and the daemon
restarted several times. Output is stable within an environment but shifted
across one. The responsible variable has not been identified, so GPU
reproducibility is currently evidenced **within** a stable environment, not
across arbitrary ones. The verdict cache detects this: an unchanged re-run that
misses the cache means the reader drifted.
