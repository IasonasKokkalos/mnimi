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

**RESOLVED (2026-07-28): pinning `num_batch` is insufficient.** The drift that
survived it was the llama.cpp **prompt cache** reintroducing variable batch size
at the logits position. A cold prefill computes the final logits inside a
512-token batch; a cache hit computes them in a batch of **one**
(`n_past = 27622`, `need to evaluate at least 1 token`). Different batch size at
the logits position, different float reduction order, different argmax — the
exact mechanism `num_batch` was pinned to prevent, re-entering through a
different door. Both states are individually deterministic and both survive a
daemon restart, so this presents as a **bistable output, not as noise**.

Three stacked variables were found, all now pinned:

| Variable | Effect | Fix | Verified |
|---|---|---|---|
| Prompt cache (content-addressed) | cache hit → logits in a batch of 1 | `LLAMA_ARG_CACHE_RAM=0` | yes |
| Slot prefix reuse | 64-token shared system prompt reused, so question *i*'s prefill boundary depends on question *i−1* | per-question cache-bust prefix (`plain-prose-v2`) | yes |
| CUDA graph warmup | first request after model load reuses 555 graphs vs 1,608 once saturated | **not yet fixed** — see below | no |

With the first two pinned, three consecutive probe runs on one daemon were
byte-identical, and prefill went from "27,191 of 27,255 tokens" to the full
count every time. `cache_prompt: false` in the request options is **ignored** by
this Ollama build; `--cache-ram 0` via the `LLAMA_ARG_CACHE_RAM` passthrough is
the only switch that works. Note `prompt_eval_count` reports the full prompt
length on a cache hit, which is why the existing instrumentation never saw this.

**CLOSED — first-request-after-load.** The first inference after a model load
runs against a cold CUDA graph cache (555 graphs reused vs 1,608 once warm) and
answers differently from the same input. The fix is `_force_model_load` in
`evals/__main__.py`, which already runs before question 1 to make the daemon log
its resolved settings for preflight, and incidentally absorbs this state.

The mechanism matters more than the fix: determinism here does **not** come from
saturating the graph cache — an early synthetic warmup reached only 1,033 of
1,608 and still did not reproduce the steady state. It comes from an **identical
request sequence**. Both runs of a restart pair begin with the same fixed
30-token generate, so graph reuse tracks identically (1, 95, 205, 458 in both)
and every question sees the same state. `_force_model_load` is therefore
load-bearing for reproducibility despite looking like a preflight helper;
removing it returns question 1 to the cold-graph state.

**Measured error bar (2026-07-28).** Same pins, same `pins_hash`, daemon killed
and relaunched between runs, on a clean GPU:

| System | Predictions changed | Score |
|---|---|---|
| `no_memory` | **0/20** | 10.0% → 10.0% |
| `full_history` | **0/20** | 20.0% → 20.0% |

Zero. The drift is closed, and the honest error bar across a daemon restart is
0/20 predictions and 0 points — not the retired 12/20 figure.

**Flash attention is a separate, real variable.** With cache state held
constant, FA=0 vs FA=1 changed 2/2 probe predictions (byte 0 of a 1,924-char
answer; byte 255 of a 383-char one). Unset, the daemon resolves
`flash_attn = auto`, and what `auto` picks is a property of the host GPU — so an
unpinned run is reproducible on one machine only. Now pinned to `1` and asserted
at preflight.

**Run precondition.** The daemon must be launched manually with
`OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0`, and **the tray app must not
serve**: it starts a daemon on 11434 with neither set, which silently yields
`auto` plus a live 8 GiB prompt cache. `preflight_reader_env` reads the
daemon's *resolved* values from its load log and refuses to run otherwise —
requested-vs-resolved, the same discipline that caught flash attention. Set
`OLLAMA_SERVE_LOG` when launching manually, or the harness reads the tray app's
log path instead.

**Killing the daemon is not enough — kill `llama-server` too.** Ollama spawns
`llama-server.exe` child runners that **outlive** `Stop-Process -Name ollama`.
Six accumulated unnoticed across restarts during this investigation, holding
5.3 GiB of VRAM on a 6 GiB card with the GPU pinned at 100%; a 27k prefill that
takes 7.3s on a clean card took 50-59s under that contention. Ollama's own
`/api/ps` reported one loaded model and gave no hint of the other five — only
`nvidia-smi --query-compute-apps` showed them. Always:

```
Get-Process -Name "ollama","ollama app","llama-server" | Stop-Process -Force
```

Contention of this kind did **not** change output — controlled comparisons run
under it were still byte-identical — but it makes every timing number
meaningless, so verify VRAM is released before trusting a wall-clock figure.

**Why the earlier 4-step GPU protocol passed while missing all of this.** Every
step replayed the **same question**, which held the cache-state sequence
constant by accident. In a real run each question has a distinct context but
shares the system-prompt prefix, so question *i*'s prefill boundary depends on
question *i−1*, and a daemon restart shifts the whole chain. The variable is
**order dependence, not process identity**. The transferable lesson: *a protocol
that replays a single input cannot detect an order-dependent variable.* Vary the
sequence, not just the repetition count.

**Retired:** the "12/20 predictions changed, 5 points, one question flipped"
figure must **not** be cited as an error bar. It was a cache-state artifact and
is reducible; publishing it as irreducible noise would overstate the floor. The
honest error bar is a re-run under these pins, and is not yet measured.

It was also weaker evidence than it appeared. Both 20-q artifacts it came from
record `stage='judge'` — they were judge-stage **replays** over a stored
`predictions.jsonl`, not fresh predict runs, and `full_history`'s
`judge_cache_hits=8 / misses=12` *is* the "8/20 identical, 12/20 changed"
figure. That is verdict-cache bookkeeping against predictions of unknown
provenance, not a controlled comparison. When quoting a reproducibility number,
check `run.stage` first: only `stage='all'` (or a fresh `predict`) re-runs the
reader, and only that can measure reader drift.
