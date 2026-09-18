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
  mnimi must clear. *(Superseded 2026-07-28: oracle retrieval added as the
  ceiling; full-history demoted to a truncated-context baseline — see "Oracle is
  the ceiling" below.)*
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

## Oracle is the ceiling; full-history is not (2026-07-28)

**Decision:** add `systems/oracle.py` and label it **the ceiling** — the
paper's own choice for the role (§5.5). Oracle's context is only the annotated
evidence sessions (`answer_session_ids`, which `dataset.py` already loads and
never uses), through the same reader and the same prompt: the score a perfect
retriever would get. full_history is relabelled a **truncated-context
baseline** everywhere it was called a ceiling.

**Why full-history cannot be the ceiling:** at the pinned 32K reader context
it truncated 20/20 smoke-slice questions — ~27,210 tokens fed, ~91,844
dropped, so the reader saw ~23% of each history. The band it anchored (floor
10% → "ceiling" 20% at n=20) was about two questions wide; nothing can be
ranked inside it. A "ceiling" that truncates measures the context window, not
achievable accuracy.

Oracle is ~20 lines over data already loaded, produces small contexts, and is
near-free to run. The W3 artifact becomes five systems (no_memory,
full_history, oracle, naive_rag, mnimi-v1) on one stratified slice.

## Sample size: n=20 is smoke, n=100+ is evidence (2026-07-28)

**Decision:** n=20 runs prove the loop closes and nothing else. Their numbers
never appear in a published table: the six categories sit at n=3-4 each, so a
single question moves a category cell by 25-33 points (and the overall score
by 5).

The evidence table is **n≥100 on one stratified slice, all five systems, one
sitting**. Cost, measured: full_history ≈ 3.2 min per 20 questions on GPU, so
n=100 ≈ 16 min for the most expensive system — affordable.

## v1 build: PLANNED vs ACTUAL (2026-07-29)

The v1 library (v0.3.0–v0.3.2, nine commits) was built from a nine-item plan.
This is the audit of plan against shipped code, item by item, **including the
small divergences** — Phase C may be wired from a different session, and a
divergence that is only in someone's head is a confound waiting to happen.
`docs/SPEC.md` §v1 as built carries the resulting state; this file carries why
each one differs.

**Verified as planned, no divergence** (recorded so the audit is a closed set,
not a highlight reel): `MemoryConfig` holds exactly `dedup_cosine_threshold`
and `top_k` — nothing crept in; the `memory_meta` check **raises**
`MemoryMetaError` at open and does not log-and-continue; `distance_metric=L2`
is explicit in the vec0 DDL with a test asserting the stored SQL string;
`pinned` is gone from the dataclass and the table; a grep of `src/mnimi/` for
`datetime` / `time.time` / `now()` / `utcnow` / `monotonic` / `perf_counter`
returns **zero hits**, and `created_at = None` raises rather than falling back;
dedup is exact-normalize plus a single `k=1` cosine probe and nothing else,
with the threshold read from `self.config` at call time; `add()` takes
`list[dict]` only, at per-round granularity, with the session date folded into
content and role excluded from the embedded string.

### BGE revision: pinned by the assistant, not maintainer-vetted

**Planned:** pin `BAAI/bge-small-en-v1.5` to an HF commit sha, never a tag,
with the sha shown before use. **Actual:** pinned to
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. The sha was resolved by Claude Code
during implementation from the HF API and independently cross-checked with
`git ls-remote` (both agreed, re-confirmed 2026-07-29). It is the current
`main` HEAD of the repo, i.e. "latest", **not** a maintainer-selected release,
and the human has not verified it. Recorded as machine-resolved so nobody later
reads it as a vetted choice. Re-checkable in one command:
`git ls-remote https://huggingface.co/BAAI/bge-small-en-v1.5`. The pin's
*function* — freezing the corpus's vectors for the life of a run — holds
regardless of who chose it.

### ONNX weights are the repo's published export, not a local conversion

**Planned:** unstated. **Actual:** `hf_hub_download(name, "onnx/model.onnx",
revision=…)` — the export BAAI publishes inside the model repo at that same
revision. Nothing is converted on this machine, so the graph is pinned by the
same sha as everything else. A locally converted graph would have been an
unpinned artifact produced by whatever converter version happened to be
installed — exactly the class of thing the guards exist to prevent.

### `tokenizers` added to the `[embed]` extra

**Planned:** onnxruntime + huggingface_hub, explicitly not
sentence-transformers/torch. **Actual:** three packages — `onnxruntime`,
`tokenizers`, `huggingface-hub`. The ONNX graph consumes token ids, so
WordPiece has to come from somewhere; hand-rolling it over `vocab.txt` is a
silent-correctness risk on a component that decides every vector. `tokenizers`
is the HF Rust tokenizer and pulls in no torch. The core deps are still
`sqlite-vec` + `numpy` and the library still imports with only those.

### `Embedder` protocol widened with `name` and `revision`

**Planned:** not specified. **Actual:** the protocol grew two properties beyond
`dim`/`embed`, because the `memory_meta` guard has to get the identity from
*somewhere* and reading it off the class would break the seam. Consequence
worth stating: any third-party embedder must now declare an identity, which is
the intended pressure — an embedder that cannot name itself cannot be pinned.

### `memory_meta` ships 3 of 11 rows; `embed_template_hash` is the hole

**Planned:** "embedder name/revision/dim written once, checked on load."
**Actual:** exactly that — and nothing else. Eight of the remaining rows are
extraction artifacts that do not exist yet, so they are correctly absent.
`embed_template_hash` is **not** in that category: v1 has a content template
(the `[Session date: …]` fold, the `"\n"` join) that is hashable today, and it
is not hashed. Changing that template silently changes every vector in every
store without tripping the guard. Logged as the one place where SPEC documents
an invariant the code does not enforce; close it before a published mnimi
number, not after.

### Pre-guard databases are refused, not upgraded

**Planned:** unstated. **Actual:** a database with a `memories` table and no
`memory_meta` raises at open. There is no migration system, so the alternative
would be inventing an identity for vectors whose embedder is unknowable —
which is precisely the failure the guard exists to make loud. The remedy is
re-ingest into a fresh store.

### `HashingEmbedder` kept, and it declares an identity too

**Planned:** keep it as the CI path. **Actual:** kept, and given
`name="hashing"`, `revision="v1"`, dim 256. It is still the default embedder in
the constructor. This means the guard is exercised on every CI run, and
swapping CI's embedder for BGE against an existing DB fails at open on all
three keys rather than at insert with an opaque sqlite-vec dimension error.

### Default config is a module constant, not an inline `MemoryConfig()`

**Planned:** SPEC shows `config: MemoryConfig = MemoryConfig()`. **Actual:**
`_DEFAULT_CONFIG = MemoryConfig()` at module level, used as the default
argument. Ruff's B008 flags a call in a default argument, and the usual danger
(a shared mutable default) does not apply because the dataclass is frozen —
so the semantics are identical and the lint stays clean. SPEC's signature block
is unchanged; the note lives beside it.

### `store.search` lost its default `k`

**Planned:** surface cosine from `search()`. **Actual:** that, plus the removal
of the old `k=5` default. A silent default at the store layer is how a
hardcoded `k` sneaks back into the read path after config-reading was made a
rule; making `k` required means `Memory.recall` has to pass `config.top_k`
explicitly and any future caller has to make the same decision consciously.

### `recall()` returns `list[MemoryRecord]`, dropping the cosine

**Planned:** unstated for v1; SPEC's target is `list[ScoredRecord]`.
**Actual:** `store.search` returns `(record, cosine)` pairs and the facade
discards the score. `ScoredRecord` needs `relevance`/`recency`/`score`
components that do not exist without a ranking layer, and inventing a
one-field version now would freeze a shape the ranking work has to change.
The cosine is available one layer down for anything that needs it — dedup
already uses it.

### `get_context` is a newline join; the locked block format is not built

**Planned:** unstated for v1. **Actual:** `"\n".join(record.content …)` — no
token budget, no per-record block, no chronological re-ordering. Timestamps
still reach the reader, but through a *different mechanism* than SPEC
describes: the date is folded into `content` at write time instead of rendered
at read time. Worth flagging because the two are not interchangeable — the
fold is inside the embedded string and inside the dedup key, so removing it
later changes vectors, not just formatting.

### `salience` and `supersedes` are stored and read by nothing

**Planned:** "keep salience/supersedes inert." **Actual:** exactly inert — both
are written on insert and returned on read, and no code path consults either.
Recorded rather than assumed: a reader of the schema could reasonably think
ranking already multiplies by salience. It does not. Result order is raw vec0
L2 ascending, which for unit vectors is cosine-descending — so v1 coincides
with the spec'd default weights (`similarity 1.0`, `recency 0.0`) by
construction, not by computing them.

### The vector table is `vec_memories`, not `memories`

**Planned:** SPEC's snippet showed `CREATE VIRTUAL TABLE memories USING vec0`.
**Actual:** two tables — `memories` (metadata) and `vec_memories` (vec0),
joined by shared rowid. The spec snippet was wrong, not the code; SPEC is
corrected. Anything querying the store directly (a debug script, a future
export) needs the join.

### `store.search` over-fetches then scopes to the user

**Planned:** unstated. **Actual:** vec0 KNN is global, so the query takes
`k*8` rows, joins to metadata, filters to `user_id`, then truncates to `k`. A
store holding many users can therefore under-return for a user whose records
are crowded out of the global window. Inert under the eval protocol (one user
per DB, reset per question) and wrong at multi-user scale — logged so it is
found by reading rather than by a support ticket.

### Per-round semantics, stated exactly (Phase C confound risk)

**Planned:** "one record per user+assistant round." **Actual:** that, plus four
decisions the phrase does not settle, each of which `naive_rag` must copy
verbatim or the comparison is confounded: (1) turns with empty/whitespace
content are dropped **before** pairing; (2) pairing is greedy left-to-right and
only `user` → `assistant` pairs, so assistant-first, user-user and a trailing
user turn each produce a **solo** record; (3) content is
`f"[Session date: {date}] {text}"` with `text` = stripped contents joined by
`"\n"` and `date` taken from the **round's first turn**; (4) `source` is the
roles joined with `"+"`. See SPEC §`add()` as built for the enumerated
contract. Note also that mnimi dedups and `naive_rag` must not — they differ in
policy, never in how a round is constructed.

### A missing `ts` fails at the store, not at `add()`

**Planned:** "insert honors injected ts." **Actual:** `add()` builds the round
without a date prefix and `store.insert` then raises `ValueError`. So the
exception a caller sees for an undated message comes from the storage layer and
mentions `created_at`, not from argument validation. Acceptable — there is
exactly one clock check and it sits at the boundary that would otherwise have
needed a fallback — but worth knowing when reading a traceback.

### `export()` was never in the v1 batch

**Planned:** the nine items did not include it. **Actual:** the public surface
is four methods, not the five SPEC locks. Not a slip; recorded because
"five methods, locked" appears in three documents and the code has four.

## `dedup_cosine_threshold` retuned 0.85 → 0.95 (2026-07-29)

**Decision:** raise the default from 0.85 to 0.95, on measurement taken before
the first mnimi run rather than after a bad one.

**What 0.85 was doing.** On a stratified LongMemEval slice with the pinned BGE
embedder, ingesting real haystack sessions through `Memory.add`:

| threshold | corpus discarded | evidence rounds dropped | questions losing ≥1 evidence round |
|---|---|---|---|
| 0.85 | 37.7% (567/1,505) | **44.2%** (57/129) | **12/12** |
| 0.95 | — | 2.3% (3/129) | 2/12 |

Retrieval correctness gates ~90% of correct answers (Fig 14), so a write path
deleting 44% of the annotated evidence is not a tuning nicety.

**Mechanism, after eliminating the wrong suspect.** The first hypothesis was
that the `[Session date: …]` fold is inside the embedded string, and shared
boilerplate in every vector drags pairwise cosines up — the mechanism
CHANGELOG #15 locked out for role prefixes. **Measured and false:** stripping
the prefix moves the drop rate 42.5% → 46.3%, i.e. nowhere. The real cause is
the threshold against this data's distribution. For real rounds the
max-cosine-to-any-earlier-round distribution has **median 0.839**, so 0.85 sits
at the median of the noise and halves the corpus by construction. Drop rate by
threshold: 0.85 → 42.5%, 0.90 → 20.1%, 0.92 → 10.4%, 0.95 → 1.9%, 0.97 → 0.4%.

**Why the old number was wrong in a way worth recording.** 0.85 (and SPEC's
0.80–0.92 tune range) came from NovAScore, which thresholds **extracted
facts**. v1 embeds **whole conversational rounds** — long, topically clustered
text where BGE-small's cosines compress into a high, narrow band. The constant
was imported across a change of embedded unit, which is exactly the kind of
borrowed number that survives review because it has a citation. When
extraction lands and the embedded unit becomes a short fact again, this must be
re-measured, not reverted by memory.

**Consequence for the tests.** Two tests asserted behaviour that depended on
the default's *value* (a near-pair merging under the default). They now state
thresholds explicitly and bracket the fixture's measured cosine of 0.923 —
0.90 merges, 0.95 does not. A test that fails when a default is legitimately
retuned is testing the wrong thing.

## The n=20 stratified slice is a DEV slice (2026-07-29)

**Decision:** the n=20 stratified slice is development data. Its numbers are
not publishable and never were — not because n=20 is small (it is), but because
**the dedup threshold was selected against it**. The selection was made on
evidence-round retention rather than on accuracy, which is the more defensible
of the two, but it was still made against this slice. That makes every number
computed on it a training number.

**How the evidence slice relates to it, stated plainly rather than implied.**
`sample_stratified` seeds one RNG, shuffles each category bucket once, then
takes round-robin until `limit`. The shuffle does not depend on `limit`, so
slices nest. Verified, not assumed:

- n=20 is a strict **prefix** of n=100 — the same 20 questions, in the same
  order, `s100[:20] == s20`.
- n=100 is a subset of n=500.

So **the n=100 evidence slice contains the entire dev slice**: 20 of its 100
questions are the ones the threshold was tuned against, 20% contamination.

**Consequence for the W3 artifact.** Report the n=100 headline *and* the 80
held-out questions as a separate line. It costs nothing — it is a slice of the
same run's results, no extra compute — and it is the number that carries no
contamination. A reader who wants the clean claim gets it without having to
trust that 20% did not matter.

**What exactly was selected against the dev slice, stated precisely so the
claim is neither overstated nor quietly minimised.** The threshold was chosen
on **evidence-round retention** — what fraction of the annotated evidence
rounds survived dedup — measured on the n=20 dev slice's haystacks. It was
**not** chosen on accuracy: no score was consulted, and the 0.85-vs-0.95
accuracy comparison was run afterwards, as a check on a decision already made.
That is the weaker form of contamination, and it is still contamination:
retention was measured on the same questions the evidence run will score, so a
threshold that happens to suit those particular haystacks cannot be
distinguished from one that suits the benchmark generally.

**Prohibited: further threshold sweeps against this benchmark.** Fixing a
default that was destroying 44% of evidence is bug-fixing; scanning 0.90 /
0.93 / 0.97 for the best score is tuning on the test set, and it is the single
easiest way to get the whole number dismissed.

**Rule for any future threshold tuning: the dev source must not be
LongMemEval questions at all** — not a held-out slice of them, not a different
seed. A held-out LongMemEval slice still shares the dataset's construction
(synthetic haystacks, planted evidence, one generator), so tuning against it
fits the corpus's idiosyncrasies and reports the result on the same
idiosyncrasies. Tune on a separate conversational corpus, or on a synthetic
retention fixture built for the purpose, and report the chosen value on
LongMemEval untouched.

## Oracle is a ceiling on evidence AVAILABILITY, not on retrieval quality (2026-07-29)

On the n=20 dev slice mnimi scored 45.0% and oracle 35.0%. At n=20 that gap is
two questions and inside the noise — but "a system beat the ceiling" will be
read as a bug unless the mechanism is written down, so:

**Oracle dumps whole evidence sessions, including every irrelevant turn in
them** (5,126 mean prompt tokens). mnimi retrieves the top-10 rounds (4,708).
Oracle therefore bounds what a system could know — perfect *availability* of the
annotated evidence — and does not bound how well a system presents it. A
focused retriever can hand the reader a cleaner context than the evidence
session it came from, and a 1.5B reader is sensitive to exactly that: weak
readers degrade past ~3k retrieved tokens (LongMemEval §5.2).

So a system scoring above oracle is not prima facie a bug, and oracle scoring
below a retrieval system is not evidence the oracle is broken. What *would* be
a bug: a system scoring above oracle while missing evidence oracle had.

## Pins schema /2: everything that moves the score is in the hash (2026-07-29)

**Decision:** `artifact_schema` → `mnimi-eval-artifact/2`, adding
`dedup_cosine_threshold` and the three reader-trim fields, and actually filling
`embedder_name` / `embedder_dim` / `k` for retrieving systems.

**Why, in one measurement.** Two mnimi runs differing only in dedup threshold
produced **19/20 different predictions and a 20-point score difference**, and
carried byte-identical `pins_hash` (`649681397649b99d…`). Under /1 the threshold
had no slot at all, and the retrieval fields carried a docstring saying "a
system that retrieves fills them" while nothing filled them. A hash that cannot
tell those two runs apart is not a weak guarantee, it is a false one — the third
time the pins contract has been falsified by measurement rather than review.

**The trim trio came along for the same reason.** `reader_num_ctx` was pinned
but `answer_reserve`, `_SCAFFOLD_TOKENS` and `_CHARS_PER_TOKEN` were not, and
the budget is `num_ctx − answer_reserve − scaffold` compared against a
`chars_per_token` estimate. full_history truncates on every question, so all
three move its score. Pinning one input of an arithmetic expression and not the
others is not a partial guarantee either.

**Mechanism:** `MemorySystem.retrieval_pins()` returns `{}` by default; a
retrieving system returns the knobs it uses. The runner branches on nothing —
it splats whatever the system declares. naive_rag deliberately omits
`dedup_cosine_threshold`: it does not dedup, and reporting a threshold it never
applies would misdescribe the run.

**The six pre-fix smoke runs were not rewritten.** Back-filled pins are written
alongside as `pins.backfilled.json`, carrying
`artifact_schema = mnimi-eval-artifact/2-backfilled` so they can never be
mistaken for emitted ones, and recording `answer_reserve = 1024` because that is
what those runs actually used. Overwriting `pins.json` would manufacture
emitted-looking metadata for runs that never emitted it — the same species of
false claim this schema bump exists to stop.

**Superseded 2026-07-29 (same day):** those six runs predate the
`answer_reserve` 1024→800 change and the time-ordered `get_context`, so their
predictions are no longer reproducible from current code. The back-filled pins
stay as an archival record of the 0.85-vs-0.95 comparison and nothing else.

## The paper is in the repo, and it moved three things (2026-07-29)

`docs/longmemeval-arxiv-2410.10813.pdf` (ICLR 2025 camera-ready, v2, 28pp).
The reference judge implementation was also read directly from
`src/evaluation/evaluate_qa.py` in the authors' repo, which is the byte-level
source the paper's Figure 10 only typesets.

**1. Two judge templates deviate from the reference by one character.**
`_STANDARD` and `_TEMPORAL` are missing a space before the `\n\nQuestion:`
block — the reference reads `"...answer no. \n\n"`, ours `"...answer no.\n\n"`.
The other three are byte-identical. Not fixed: a one-character edit moves every
verdict key and it needs an explicit decision, not a drive-by.

**2. The judge was being sent a system message the paper never sends.** The
reference sends exactly one user message. Ours prepended
`"You are a strict grader. Answer with only 'yes' or 'no'."`, which was not in
the 97%-human-agreement configuration. Removed; `_JUDGE_SYSTEM = None` rather
than deleted, so its absence is a value the hash covers.

**3. "JSON" in the paper is the format of the RETRIEVED CONTEXT, not of the
reader's output.** §5.5: *"we present retrieved items in a structured JSON
format (Yin et al., 2023), which helps the model clearly recognize memory items
as the data for reading"*. Figure 13's CoN prompt ends `Answer (step by step):`
and asks for prose. There is no JSON object to parse, no `answer` field, and
therefore no parse-failure mode. The planned `json-con-v1` name and the whole
JSON-output parse/fallback policy were built on a misreading of what the
JSON+CoN result measured. Presenting *context* as JSON is a `get_context`
change affecting all five arms, which is a different and larger decision.

## Statistics: pre-specified comparisons and what the intervals mean (2026-07-29)

**Module:** `evals/stats.py`, one code path for n=20 / n=100 / n=500. No
statistic quoted in any doc is computed anywhere else.

**Pre-specified before any n=100 or n=500 data exists:**

- **Primary:** mnimi vs naive_rag, exact McNemar, reported **uncorrected** and
  labelled primary.
- **Secondary:** mnimi vs no_memory, mnimi vs full_history, mnimi vs oracle,
  **Holm-corrected** as one family.
- **No other pairwise test is reported.** A comparison chosen after seeing the
  data is not a test.

**Exact McNemar, not chi-square**, because the approximation needs a decent
discordant count and we do not have one — the primary pair produced *two*
discordant pairs at n=20. `b`, `c` and `n` are emitted alongside every p-value,
so a reader can see whether a result rests on 3 pairs or 300. Question-id sets
are asserted identical across arms before any test; a mismatch raises rather
than silently comparing unpaired numbers.

**Wilson score intervals on every headline number.** At 2/20 the normal
approximation produces a negative lower bound, which is not an interval.

**What the interval covers, stated because it is narrower than it looks.** The
CI is question-sampling error only: how much the number would move if a
different sample of LongMemEval questions had been drawn. It does **not**
cover judge error (the paper's own judge agrees with human experts ~90-97%
depending on category, so a category cell carries judge noise the interval
never sees), reader nondeterminism (measured separately as a
predictions-changed count), or dataset construction error (the haystacks are
synthetic and the evidence is planted). Two arms whose intervals overlap have
not been shown equal, and two whose intervals exclude each other have not been
shown different by the interval alone — that is what the paired test is for.

**The daemon-restart repeat is reproducibility, not a confidence interval.** It
is reported in its own section as a predictions-changed count and a score
delta, and is never printed as an error bar on an accuracy number. They measure
different things: one asks whether the same inputs give the same outputs, the
other asks how much the answer depends on which questions were drawn.

## Power: n=100 cannot test the primary thesis (2026-07-29)

**Computed from the n=20 dev-slice artifacts, post judge-fix.** mnimi vs
naive_rag: `b=2, c=0`, **2 discordant pairs out of 20**, observed discordance
rate 0.10, p=0.50.

Minimum true accuracy gap detectable by exact McNemar at α=0.05, power=0.8
(`evals.stats.minimum_detectable_gap`, averaging conditional power over the
random discordant count):

| discordance rate | MDE at n=100 | MDE at n=500 |
|---|---|---|
| 0.05 | **unreachable** | 2.9 pts |
| 0.10 *(observed)* | **8.8 pts** | 4.1 pts |
| 0.15 | 11.0 pts | 5.0 pts |
| 0.20 | 12.8 pts | 5.8 pts |
| 0.30 | 15.9 pts | 7.0 pts |

"Unreachable" at 0.05/n=100 is a floor effect, not a rounding artifact: ~5
discordant pairs are expected, and the exact test needs **at least 6** all
falling one way before a two-sided p can reach 0.05 at all.

**The conclusion, and it is a stop.** mnimi-v1 is expected to open ≈0 points
over naive_rag — that is the project's own prediction, by construction: at
threshold 0.95 dedup is nearly inert, the two arms differed on 3 of 20
predictions and 2 of 20 verdicts, and v1 has no other mechanism to separate
them. **n=100 would need an 8.8-point true gap. n=500 would need 4.1.** Neither
tests the primary thesis, and the pairing does not rescue it — pairing is
already what makes these numbers as small as they are.

Recorded as a design fact rather than a run-size recommendation: **no run size
is recommended here.** The primary comparison is underpowered for v1 at any
slice of this benchmark, so the decision to make is what mnimi must *do*
differently before the comparison is worth running, not how many questions to
spend confirming a null.

The secondary pairs are differently placed — observed discordance on the dev
slice is 0.35 (no_memory), 0.55 (full_history), 0.40 (oracle) — so those
comparisons carry far more discordant pairs and are the ones n=100 could speak
to. mnimi vs no_memory already reaches p=0.0156 (Holm 0.0469) at n=20.

**Caveat that limits all of the above:** the 0.10 discordance rate is estimated
from two pairs. The table is therefore given across a range rather than at a
point, and every row of it says the same thing about n=100.

## Judge fidelity (ruling) (2026-07-30)

`_STANDARD` and `_TEMPORAL` corrected to byte-match the reference
implementation (one missing space each before the question block); unknown
`question_type` now raises `NotImplementedError`, matching the reference's
strictness; judge decode configuration (`temperature=0`, `max_tokens=10`)
promoted to pinned constants, digested in `judge_prompt_hash`, and recorded in
the artifact judge block. Rationale: the 97% human-agreement figure was
measured on the reference judge; a judge that deviates by any byte is a
different instrument, and every configuration input to a verdict must
invalidate the cache when it changes. Re-grade under judge_prompt_version
`longmemeval-paper-v3`: 1 verdict flip across 193 gradings; per-run deltas in
the commit report.

## Superseded power analysis (correction) (2026-07-30)

The n=20 five-system table was internally consistent — all six arms share one
harness configuration (`harness_git_sha` 4415b8aa, `answer_reserve` 1024,
pre-time-ordering) — so the derived power analysis (b=2, c=0, discordance
0.10) was computed within one configuration, not across two. It is superseded
regardless: that configuration no longer exists, and the change to it was
measured to alter 16/20 of mnimi's predictions at an unchanged score, so the
discordance estimate does not transfer to the current harness. The mixing is
between eras, and it appears in SPEC's error-bar table, whose rows pair those
six arms against the post-Phase-D artifacts (`answer_reserve` 800,
post-time-ordering). `stats.py` hard-fails such pairings via
`HARNESS_PARITY_FIELDS`.

## embedder_revision (header) (2026-07-30)

Retrieval arms now declare the pinned HF commit of the embedder in pins. This
closes the last identified input that could change every number without
changing the header.

## Judge instrument error (measured) (2026-07-30)

Question eace081b changed verdict between the v2 and v3 judge on a request
verified byte-identical. Re-graded 20 times under the v3 configuration with
the verdict cache bypassed: 9 yes / 11 no. Together with the 0 flips across
120 gradings at the v1->v2 re-judge, the observed spontaneous flip rate is 9
in 140 gradings. gpt-4o at temperature=0 is therefore not deterministic on
borderline rows, and absolute scores carry judge instrument error in addition
to question-sampling error. No absolute score difference smaller than this
rate is interpretable. This does not affect paired comparisons — see the
verdict-cache note in SPEC.

## Reader prompt (ruling) (2026-07-30)

mnimi-con-v1 is LongMemEval Figure 13 verbatim plus one abstention sentence
and the cache_bust determinism prefix. It is not the paper's prompt and is
not named as one; json-con-paper-v1 is reserved for a byte-exact
reproduction. The abstention sentence is required because 30 dataset
questions are abstention questions and abstention is the no_memory arm's only
route to a correct answer; Figure 13 has no such instruction. The retired
json-con-v1 rested on reading §5.5's "structured JSON format" as the reader's
output format; it is the context format. No JSON parser was built and none is
needed.

## JSON context presentation (deferred) (2026-07-30)

Presenting retrieved items as structured JSON per §5.5 is a get_context
change across all five arms and forces a full re-run. It is a fidelity
improvement, not a precondition for a defensible number, and it is deferred
rather than bundled. Recorded in FUTURE.md.

## Context format parity (correctness, not cosmetics) (2026-07-30)

mnimi and naive_rag previously rendered reader context with date-only
timestamps and no speaker attribution, while full_history and oracle carried
the dataset's full timestamp and user:/assistant: labels. This was a confound
in every cross-arm comparison — the mnimi-oracle gap mixed retrieval quality
with date granularity and role labelling — and a correctness defect for mnimi
on single-session-assistant questions (11% of the benchmark), which ask about
assistant turns mnimi did not attribute. All five arms now share one
renderer.

## Pre-registered interpretation of the n=100 evidence run (2026-07-30)

Pre-registered interpretation of the n=100 evidence run (recorded before
the run, from the n=20 dev slice and the power analysis):

- mnimi vs no_memory is the W3 criterion. The floor is beaten if the exact
  McNemar test is Holm-significant at alpha=0.05 within the secondary
  family. At n=20 this pair was b=7, c=0, p_holm=0.0469; it is expected to
  hold and strengthen.
- mnimi vs naive_rag is the pre-registered PRIMARY and a pre-registered
  NULL. mnimi-v1 stores verbatim rounds and retrieves top-k, which is
  functionally naive RAG on a benchmark constructed without conflicting
  facts; dedup earns nothing here by construction. The power analysis
  states n=100 requires an 8.8-point true gap at the observed 0.10
  discordance. A non-significant result is the expected outcome and is not
  a negative finding. A SIGNIFICANT result on this pair is to be treated
  first as evidence of a harness asymmetry between the two arms, and
  investigated as such, before being reported as a capability difference.
- mnimi vs full_history is confounded by truncation: full_history is a
  truncated-context baseline, not a ceiling. Its truncation rate is
  reported beside its score and any comparison is read through it.
- mnimi vs oracle: oracle bounds evidence AVAILABILITY, not retrieval
  quality. It supplies whole evidence sessions including irrelevant turns;
  a focused retriever handing the reader fewer, cleaner tokens can exceed
  it. mnimi above oracle is not a defect and is not to be reported as one.
- Per-category scores are reported for completeness only. At n=100 across
  seven question types the cells are too small to support any per-category
  claim, and none will be made.
- Absolute scores carry judge instrument error in addition to
  question-sampling error; no absolute difference smaller than the measured
  flip rate is interpretable.
- The n=100 headline includes the 20-question dev slice on which the 0.95
  dedup threshold was selected (on evidence-round retention, not accuracy).
  The 80 held-out questions are reported as a separate line and carry no
  contamination.

## Embed/render separation (2026-07-30)

The text embedded and used as the dedup key is now built separately from the
text rendered into reader context, and the two are hashed separately
(embed_template_hash, render_template_hash). The embed text is frozen
byte-identical across this change set, verified by test. Without this, the
format-parity change would have moved every vector and voided the evidence
the 0.95 dedup threshold was selected on — evidence that cannot be
regenerated, since further threshold selection against LongMemEval is
prohibited.

## The published number is the 0.32.13 sitting (2026-09-10)

**Decision:** `runs/replay/{no_memory,full_history,oracle,naive_rag,mnimi}__100q/`
— the five-arm n=100 sitting of 2026-08-16 on Ollama 0.32.13, harness
`ae5da2b` with a clean tree, `provisional: []` on every arm — is promoted to
`results/published/<arm>__100q/` and is the number this repo quotes:
no_memory 4, full_history 17, oracle 50, naive_rag 41, mnimi 35 (of 100).

**The alternatives, and why not.** (a) The 2026-07-30 sitting on 0.32.5
(`runs/<arm>__100q/`, mnimi 43 = oracle 43) is provisional on a dirty tree and
cannot be regenerated: the transport moved under it, 20/20 predictions changed
on byte-identical prompts, and the headline it supported reversed on the next
build. (c) Waiting for a daemon-restart pair on a frozen build would leave the
repo with no published n=100 number at all while the only build with a
publishable-grade set is already gone from the development machine (daemon now
0.33.3). The 0.32.13 set is the only one the harness itself marks quotable.

**What it claims.** Tier 1 auditable on every row; Holm-significant separation
from `no_memory` (b=33, c=2); mnimi below the evidence-availability bound
(b=3, c=18). **What it does not claim.** Tier 2 reproducibility on 0.32.13
(restart pair unmeasured); a confirmatory primary — mnimi vs naive_rag
(b=0, c=6, p=0.0313) is post-hoc relative to the 0.32.5 pre-registration and
sits at the minimum discordance where p<0.05 exists; any per-category cell.

**The rule that comes with it.** A benchmark number never travels without its
provenance — reader build, harness commit, daemon environment, date. Two
transport moves under identical `pins_hash` are the reason; the build itself
becomes a pin in the next commit (pins schema /5).

## Pins schema /5: the reader transport build is a pin (2026-09-10)

**Decision:** `artifact_schema` → `mnimi-eval-artifact/5`, adding
`reader_transport_version` — the Ollama server build — to `build_pins`, hence to
`pins_hash`, and to `HARNESS_PARITY_FIELDS`. The constant is
`runner.READER_TRANSPORT_VERSION = "0.32.13"`. Preflight reads
`GET /api/version` right after the tags check and BEFORE the warm-up load, and
refuses on any other string, or on no string at all (unreachable endpoint or
missing field = cannot confirm = fail closed). Exact compare: no semver parsing,
no prefix match.

**Why, in one measurement.** The tray app auto-updated the daemon 0.32.5 →
0.32.13 (2026-08-16). The replay gate — the mnimi arm's 20-question prefix of
the n=100 slice, identical pins, identical `pins_hash` — changed **20/20
predictions**, with `reader_prompt_tokens` identical on every row: ingest,
dedup, retrieval, render and trim reproduced byte-for-byte and only the reader
binary moved. The full re-baseline confirmed it across all five arms (100/100
rows moved on three, 99/100 on the other two) and reversed the headline:
mnimi 43 = oracle 43 became mnimi 35 < oracle 50. Nothing in the header could
tell the two sittings apart, because the build lived only in the diagnostic
`run.environment` block, which by design never enters the hash. A hash that
cannot separate two runs differing on 100/100 predictions is the same false
guarantee /2 closed for the dedup threshold — the fourth time the pins contract
has been falsified by measurement rather than review. The daemon has since
moved again, to 0.33.3.

**Mechanism.** The same requested-vs-resolved split as flash attention. The pin
holds the build the harness *requested* (the constant);
`run.environment.ollama_version` keeps holding the build that *resolved*; and
`preflight_reader_transport` guarantees they agree or the run never starts — no
model load, no run directory. A cross-build pairing in `evals.stats` now fails
naming `reader_transport_version` and both builds rather than surfacing as a
bare `harness_git_sha` mismatch; /4 artifacts lack the field on both sides
(`None == None`), so the published 0.32.13 set keeps pairing.

**What was deliberately not done.** No `_provisional_reasons` entry for a
missing field — re-judging the published /4 set would otherwise flip the only
zero-provisional artifacts to provisional. No rollback of the daemon and no
change to the tray's auto-update: which build the machine serves is an
operational decision outside this commit. The constant is 0.32.13 because it is
the only build with a publishable-grade n=100 set; a run against the current
0.33.3 daemon is refused (measured 2026-09-10: exit 2 at preflight naming both
builds, no runner spawned, no run directory touched) until either that build is
restored or the constant is changed deliberately and every arm re-baselined.

## Reader transport frozen at 0.32.13 by rollback; restart pair 0/100 (2026-09-10)

**Decision:** the development machine serves the pinned build from the release
zip, `D:\ollama-0.32.13\` (`ollama-windows-amd64.zip` of tag v0.32.13, sha256
`20d61a8075038694f5b6db1e937551dbc79d470e85217003facf6ecaac394258`, verified
against the release's `sha256sum.txt`). The desktop app (0.33.3) is
uninstalled, not merely stopped. Rollback chosen over a 0.33.3 re-baseline:
the re-baseline costs a three-hour sitting and a third mutually incomparable
set, and would supersede the published number the day after it was published;
the rollback costs thirty minutes and lets the published set gain the Tier 2
figure it lacked.

**What was found on the machine.** The desktop app's updater checks ollama.com
hourly and had already downloaded the v0.34.0 installer (1.47 GB, staged
under `%LOCALAPPDATA%\Ollama\updates_v2\`) for application at its next
launch — the mechanism that moved this machine 0.32.5 → 0.32.13 (2026-08-16)
and → 0.33.3 (2026-09-04, `upgrade.log`). Its `auto_update_enabled` setting
only suppresses the download; the check continues and a staged installer is
still applied. There is no supported way to freeze a desktop install, so the
freeze is structural: no desktop app, a version-named folder, PATH pointing at
it, and `READER_TRANSPORT_VERSION` refusing anything else.

**Procedure, as executed.** (0) Tray and daemon killed, staged installer and
the `Startup\Ollama.lnk` shortcut deleted — before anything else, because a
reboot would have installed 0.34.0. (1) Zip downloaded, checksum verified,
extracted; client reports 0.32.13. (2) 0.33.3 uninstalled silently (exit 0):
app folder, uninstall key and `%LOCALAPPDATA%\Ollama` removed; models in
`D:\ollama-models` untouched (same blob, digest `635e70c8…`); the stale PATH
entry replaced with `D:\ollama-0.32.13`. (3) Daemon launched with
`OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0`, stderr redirected to the
file `OLLAMA_SERVE_LOG` names; `/api/version` = 0.32.13, `server config`
shows `OLLAMA_FLASH_ATTENTION:true`. (4) `no_memory --limit 1 --stage predict`
passed every preflight (build, `flash_attn = enabled`, `prompt cache is
disabled`, 29/29 layers) and wrote the first schema /5 artifact; its one
prediction matched the published row byte for byte. (5) Restart pair below.

**The measurement.** `mnimi --limit 100 --stage predict` into
`runs/restart_pair/`, after `ollama stop` so the run began with the pinned
warm-up load; 3,700 s; the serve log shows no model reload during the run.
Against `results/published/mnimi__100q/predictions.jsonl`: `predicted`
identical 100/100, `reader_prompt_tokens` identical 100/100, `truncated`
identical 100/100. That is across a daemon restart, a reinstall of the server
binary from a different distribution (installer → zip), 25 days, several
reboots, three harness commits (`ae5da2b` → `511edd2`) and the pins schema
/4 → /5 bump. Promoted as `results/published/mnimi__100q_restart_2026-09-10/`
(predict-only) so the claim is a diff. The elapsed time differed (2,136 s in
August; the CPU-bound ingest ran ~4× slower for the first 20 questions this
time, then at the August pace) — timing is not a pin and the rows did not
care.

**What it does and does not establish.** Tier 2 on 0.32.13 for the mnimi arm:
0/100 changed. The other four arms were not replayed; they carry Tier 1 only
until they are. Nothing about 0.33.3 or any other build; those are refused.
The 2026-07-30 four-arm n=20 figure on 0.32.5 is consistent and superseded.

**Residual hazards, named.** NVIDIA driver updates (595.95 today, the value in
every published `run.environment`) are recorded but not pinned and can move
the reader the same way; pause them. Reinstalling the desktop app on any
machine restarts the whole problem. The keep-alive is 5 min: an ingest slower
than that between two reads would unload and reload the model mid-run, which
changes the CUDA graph state the warm-up load exists to absorb — check the
runner-start marker count in the serve log after any long arm.

## The gpt-4o era begins: a second reader transport (2026-09-11)

**Decision:** the harness gains `--reader-transport openai` with
`gpt-4o-2024-08-06` as the reader, pins schema /5 → /6 (`reader_transport`
added, the six Ollama-only pins nullable). The local family
(`qwen2.5:1.5b-instruct-q4_0` on Ollama 0.32.13) is unchanged, stays
published, and is never paired with the new family (`HARNESS_PARITY_FIELDS`
refuses).

**Why the reader moves.** The 1.5B reader's evidence-availability bound is
50/100 on the published slice: with every evidence session in context it
scores 50, so no write-side policy can show above it. The project's
completion criterion is SPEC-complete *or* a Wilson lower bound ≥ 85% on
LongMemEval, and 85 is only reachable on a reader whose bound is above it.
The paper (Fig 3b, LongMemEval-S, Chain-of-Note) puts GPT-4o at 92.4 with
oracle evidence and 64.0 with the full 115k-token context; Llama-3.1-8B at
84.8 / 28.6; Phi-3.5-mini (4B) at 65.2 / 32.4.

**Why this snapshot.** `gpt-4o-2024-08-06` is the paper's reader and its
judge; the reader behind Zep's 71.2 (vs 60.2 full context), Supermemory's
85.4 and Mastra's 84.2; and not on OpenAI's deprecation page (checked
2026-09-11 — only `chatgpt-4o-latest` was retired). Reader and judge share a
snapshot, as in the paper; disclosed, not hidden. Official price $2.50 /
$10.00 per million tokens, Batch API half.

**Why 128K.** The model's window. `full_history` becomes the paper's
untruncated full-context baseline (the 64.0 row) instead of a truncation
policy; the same trim gate still reports the few histories longer than that.

**What is pinned, what is recorded.** Pinned: snapshot, `temperature=0`,
`seed=0`, `max_tokens=800`, `num_ctx=128000`, the unchanged `mnimi-con-v1`
prompt (its hash `50c6fe10…` is asserted unchanged by a test — the transport
must not move the prompt identity). Recorded, never pinned: the response's
`system_fingerprint`, aggregated per run into `reader_resolved.json` beside
the predictions and into `run.environment` at judge time. The family's Tier 2
claim is "reproducible within measured drift", measured by one n=100 re-run.

**Guards.** `OPENAI_API_KEY` is required before any call; `--limit` above
100 is refused without `--allow-large-run`. The programme's API budget is $50
in total (`mnimi docs/PLAN.md`), an accidental `full_history` sitting at
128K and n=500 would cost ~$150, and the per-sitting cost projection is a
later task (0.3). Until it lands, the limit is the budget guard.

**What did not change.** The Ollama `Reader`, its preflights and its pins are
byte-for-byte as before; the local family's published artifacts stay valid
and its restart pair stays measured. Batch API mode (0.2), the `models.list`
preflight and the cost projection (0.3) are the next tasks.

## Batch API mode for the gpt-4o family (2026-09-11)

**Decision:** `--batch` sends the predict stage of the openai transport
through the OpenAI Batch API: one JSONL of `{custom_id, method, url, body}`
lines uploaded with `purpose="batch"`, one batch on `/v1/chat/completions`
with the only allowed window (`24h`), polled to a terminal status, outputs
downloaded and re-ordered by `custom_id` into the same `predictions.jsonl` a
synchronous run writes.

**Why.** The programme's reader budget is $50 in total and the Batch API
halves the price; the separate rate-limit pool is a bonus. Every sitting on
this family goes through it.

**Not a pin.** The batch line's body is byte-for-byte the synchronous
request (`OpenAIReader.request_body`, shared by both paths): same snapshot,
temperature 0, seed, `max_tokens`. So a batch run and a sync run share a
`pins_hash`, and a pair of them is a valid drift measurement under identical
pins. What the batch resolved to — id, status, request counts, output and
error file ids, the served `system_fingerprint` histogram — is recorded in
`reader_resolved.json`, never hashed.

**Resume contract.** `batch_requests.jsonl` and `pins.json` are written
before submission, `batch_state.json` (batch id, status, `pins_hash`, the
per-item row metadata without bodies) right after it. Re-running the same
command in the same `--run-dir` with no `predictions.jsonl` present resumes:
it refuses if the pins hash differs, otherwise polls the recorded batch and
never re-uploads. `--batch-no-wait` submits and prints the resume command,
so a 1-hour day can submit and a later day can collect.

**Fallback rule.** A terminal `completed` or `expired` batch delivers what it
finished; every item without a 200 body is answered with ONE synchronous
call using the item's own body — same configuration, different transport —
and listed under `sync_fallbacks`. `failed` (validation) and `cancelled`
write nothing; the state is kept for the record and the errors are printed.

**Run-directory extras.** `batch_requests.jsonl`, `batch_state.json` and
`reader_resolved.json` are diagnostics beside the run; the three-file
promotion rule for `results/published/` is unchanged.

**First drift observation (2026-09-11, no_memory, one shared question).** The
same request sent synchronously (served fingerprint `fp_c9a0e786b8`) and
through a batch (`fp_f923e16c69`) returned different text — identical prompt
tokens (132), divergence at character 182, different served builds. One
question is not a rate; it is the reason the family's Tier 2 statement is a
measured N/100 (task 0.7) and not an assumption of stability.

Also in this change: `pyproject.toml` `1.4.0 → 1.5.1` (the previous commit was
titled v1.5.0 without bumping the file; this one is v1.5.1 and does); the library under `src/`
is unchanged since v1.3.0.

## The cost gate: project, then spend (2026-09-11)

**Decision:** on the API family nothing is sent before it is priced,
confirmed served, projected and gated. `evals/pricing.py` holds the dated
price table (official pricing page, `PRICES_AS_OF`; a model with no row
cannot be projected and therefore cannot run), the projection, the ledger and
the gate. The `--limit > 100` guard and `--allow-large-run` are gone; the gate
replaces them.

**The projection is an upper bound, stated as such.** Input tokens at the
trim gate's chars/4 convention (gpt-4o tokenises this dataset closer to 4.7
chars per token, so the estimate runs high); output at `max_tokens` per
request (actual answers run about a third of it); judge calls at ~600+10
tokens each with no cache hit assumed. Computed from the real request bodies
— the synchronous openai path now builds every item first, exactly like the
batch path, so both project the same way and differ only in transport.

**The snapshot preflight.** `models.retrieve(snapshot)` for the reader before
any context is built, and for the judge before any verdict is asked. A retired
or mistyped snapshot found out at question 1 would already have cost the
ingest of the whole slice; `NotFoundError` and `AuthenticationError` refuse
with the snapshot named, and any other API error fails closed.

**The ledger.** `.cache/api_ledger.jsonl` (gitignored — a fact about this
machine's key, not about the code; `MNIMI_API_LEDGER` redirects it, and the
tests always do). One line per invocation: projected, actual from the API's
own `usage` counts (reader at the rate it ran, judge at standard rate), token
totals, the budget in force and whether it was overridden. A batch submitted
with `--batch-no-wait` is booked at its projection until the resume writes
the actual line, which names the submitted line it supersedes.
`python -m evals.pricing` prints the ledger and the remaining budget.

**The gate.** Projection + ledger spend > budget → refuse before any call,
naming all three numbers. The budget is `API_BUDGET_USD = 50` — the programme
cap from `mnimi docs/PLAN.md`. `--api-budget-usd` overrides it for one run,
is echoed as `BUDGET OVERRIDE`, and is recorded in the ledger line. Changing
the constant is a decision, not a flag.

**Not a pin.** Projection, actual and ledger are diagnostics; `pins_hash` does
not know what a run cost. The judge stage is gated too, on every family — the
judge is API spend regardless of which reader produced the rows.

## Pre-registration for the gpt-4o era; `full_history` is cited, not run (2026-09-12)

Recorded before the family's first scored run. The only API calls on this
family so far are the three smoke checks behind tasks 0.1–0.3 (one question
sync, two questions batch, one question through the cost gate): $0.0211 of
the $50 programme budget, all in the ledger. Nothing below was written with a
gpt-4o-era score in hand.

**What the era's first sitting is.** Four arms — `no_memory`, `oracle`,
`naive_rag`, `mnimi` — at n=100, the same stratified seed-0 slice as the
published local sitting, so every question has a local-family row beside it.
The library is v1 (v1.3.0 @ `658f516`: verbatim rounds, exact-normalize
collapse, one cosine dedup probe at 0.95, no extraction, no conflict, no
decay). Reader `gpt-4o-2024-08-06`, prompt `mnimi-con-v1` (hash
`50c6fe10…`, unchanged), `k=10`, `num_ctx=128000`, judge
`gpt-4o-2024-08-06` / `longmemeval-paper-v3`. The renderer is decided by the
presentation pair below, before the sitting.

**`full_history` is cited from the paper, not run.** Decided 2026-09-12
(PLAN §7.3). At the 128K window the arm feeds ~119k tokens per question — ≈
$15 per n=100 sitting by batch, 30% of the programme budget — for a baseline
that never changes across the programme (same reader, same harness) and that
the paper already reports on this exact reader. The harness now refuses
`--system full_history --reader-transport openai` at the predict stage,
batch or sync, before any client is built (`tests/test_batch.py`); the local
family's `full_history` arm is untouched. The cited row is **Fig. 3b,
LongMemEval-S, GPT-4o, with Chain-of-Note: 64.0%** (its oracle column: 92.4%).
The non-CoN row is 60.6% / 87.0%. `mnimi-con-v1` is the Fig. 13 CoN prompt
with two deviations (one abstention sentence, the `${cache_bust}` prefix), so
the CoN row is the comparator. Caveats that travel with the citation: it is
over all ~500 questions, under the paper's own prompt and its own judge run;
it is a point estimate from another lab, enters no paired test, and the
harness produces no `predictions.jsonl` for it. The README table carries it
in its own row marked *cited (paper Fig. 3b)*, never beside a Wilson
interval of ours. Consequence: mnimi vs `full_history` is **not** a
pre-registered comparison in this era, and the local family's "truncated
baseline" reading of the arm does not carry over — on this family the arm
would have been the paper's untruncated one, which is exactly why the
paper's number stands in for it.

**Primary: mnimi vs `naive_rag`**, exact McNemar, alpha 0.05, one test. It
stays a **pre-registered NULL at v1**, for the reason given on 2026-07-30:
the v1 write side is a near-inert dedup screen on a benchmark built without
conflicting facts. A non-significant result is the expected outcome and not
a negative finding. Directional note, recorded now: the local family
observed b=0, c=6 against mnimi, all six on rows where the dedup screen
changed the retrieved set. If the discordance on this family again runs
against mnimi, that is the input to R3 (PLAN 1.2: fix the six without
touching the threshold) and is reported as such; a *significant* result in
either direction is first investigated as a harness asymmetry between the
two arms before it is reported as a capability difference. Power: at n=100
and the observed discordance (0.06–0.10) only a gap of ~8 points or more is
detectable; the era's primary is decided at n=500 in Phase 5.

**Secondary family**, Holm-corrected together: mnimi vs `no_memory`
(expected large — the paper's floor on this reader is single digits) and
mnimi vs `oracle` (a bound on evidence availability, not on presentation:
mnimi at or above oracle is not a defect and is not reported as one). No
other pair is tested. Per-category scores are reported for completeness and
support no claim at n=100 (cells of 14–17). Absolute scores carry judge
instrument error (9 flips in 140 re-gradings on one borderline row, measured
2026-07-30); no absolute difference smaller than that rate is interpreted.
The n=100 headline includes the 20-question dev slice on which 0.95 was
selected, as before.

**The 85 criterion** (PLAN §4.4, decided 2026-09-11): read off the Phase 5
n=500 sitting, mnimi arm, this reader, the era's frozen prompt and renderer;
the **Wilson 95% lower bound must be ≥ 85.0%**, i.e. 441/500 (88.2%, interval
[85.1, 90.7]) computed with `evals.stats.wilson`. A point estimate of 85.0
does not meet it. Every n=100 sitting before that is a working number, not a
verdict. If the verdict lands short the write-up reports the gap by category
and the unspent bucket-2 items; criterion (a), SPEC-complete, stands on its
own.

**The presentation pair (R1) and what it decides.** FUTURE.md's trigger for
JSON context presentation (paper §5.5: "a planned full re-run of all five
arms") fires — the API era is that re-run — so the format rides along
without costing a run of its own. Before anything is submitted, a second
renderer is committed beside `RENDER_TEMPLATE` (current hash
`9c03ddae5c33…`): a JSON array, one object per rendered item, carrying the
same three fields the current format carries (session date, role, content),
the same item granularity per arm (rounds for the retrieval arms, sessions
for oracle), and nothing else — only the framing changes, and its hash is
the pin. mnimi and oracle run at n=100 under both renderers, by batch. **The
JSON renderer becomes the era's renderer iff its score is strictly higher
than the current renderer's on both arms; a tie or a split on either arm
keeps the current renderer.** That is the one pre-registered alternative for
this knob; no further renderer variant is tried against the benchmark. The
winning renderer's mnimi and oracle rows *are* the sitting's rows, so the
Phase 0 headline carries a one-alternative selection on the renderer,
disclosed here and in the write-up; `naive_rag` and `no_memory` run under
the chosen renderer afterwards, and the n=500 final is free of the
selection because the renderer is frozen before Phase 1.

**The drift pair.** After the sitting, mnimi n=100 is re-run once with the
same pins; `N/100 changed` becomes the family's Tier 2 statement
("reproducible within N/100 measured drift") and the served
`system_fingerprint` classes are reported beside it. Nothing is re-selected
on the pair; the first run's rows stay the published ones.

**Budget for Phase 0 after this decision (estimate, batch where possible):**
presentation pair $3.5; the remaining arms of the sitting (naive_rag,
no_memory) + judge for all four ≈ $3.5; drift pair $1 — **≈ $8**, against
the $22.5 the plan carried before `full_history` was cut. Every submission
still prints its projection and refuses over the remaining cap.

## The JSON render format lands, undecided (2026-09-12)

**Decision:** the one renderer gains a second *format*. `render_turns` /
`render_records` take `fmt="text" | "json"`; `MemoryConfig.render_format`
(default `"text"`) is the knob `get_context` reads; the harness has
`--render-format` and hands one value to every context-bearing arm
(`build_system`), so `render_template_hash(fmt)` — now hashed per format —
is true for all of them. The text format's hash is byte-for-byte what the
published local artifacts carry (`9c03ddae5c33…`, asserted by a test), so
nothing already published moved. `run.render_format` in `results.json`
names the framing for a human; the hash in `pins.json` is the pin, and it
sits in `HARNESS_PARITY_FIELDS`, so `evals.stats` refuses to pair a text run
with a JSON run — exactly right, since the presentation pair is decided on
scores, not on a paired test.

**The shape.** Both formats render the same blocks: one block per timestamp
change, the verbatim turns inside it. Text emits `[Session date: <ts>]` and
`role: content` lines; JSON emits
`[{"session_date": <ts>, "turns": [{"role", "content"}, …]}, …]` via
`json.dumps(ensure_ascii=False, indent=2)`. Same dates, same roles, same
content, nothing added — the block grouping is one function both formats
call, so the two framings cannot disagree about where a session boundary
falls. The paper does not print its JSON schema (§5.5 names the idea and
cites Yin et al. 2023); this is mnimi's own, recorded here as
`RENDER_JSON_TEMPLATE`, the descriptor string that is hashed. The two-space
indent is a readability choice for the reader model and costs tokens (a
mnimi context of ~4.7k tokens grows by roughly a fifth); the pair measures
the whole package, not the indent.

**What is not decided.** Which format the era runs. That is the
presentation pair's job (pre-registration above: JSON is adopted iff
strictly higher than text on both mnimi and oracle at n=100; otherwise text
stays). Until the pair is read, the library default stays `"text"`; if JSON
wins, the default flips in one commit and FUTURE.md's item closes with the
numbers either way. The FUTURE.md trigger ("a planned full re-run of all
five arms") fired with the gpt-4o era, which is why the format lands now and
not mid-phase.

**Why a config field and not a second renderer.** SPEC's invariant is one
renderer for every arm, because date granularity and speaker labels were a
measured cross-arm confound. A second render *function* would be a second
place for that confound to grow back; a format flag on the one function keeps
parity a property of the code. It is the third `MemoryConfig` field and the
first that is not a threshold — a knob `get_context` reads, so it earns its
place under the "no field nothing reads" rule.

## `--verify-drift` lands: the N/n-changed count is a harness command (2026-09-12)

**Decision:** the divergence count Tier 2 actually claims is now produced by
the harness, not by hand. `evals/drift.py` compares two predict runs row by
row on `question_id` and reports how many `predicted` strings changed, the
UTF-8 byte offset of each first divergence, how many rows changed
`reader_prompt_tokens`, and the served `system_fingerprint` classes on each
side. Two entry points, one comparison: `--verify-drift <reference>` on the
predict stage (sync, batch, or a batch resume — it fires when the fresh
`predictions.jsonl` lands and leaves `drift.json` beside it), and
`python -m evals.drift <reference> <fresh>` for two existing artifacts.
FUTURE.md's trigger ("the next time a drift hunt starts") is the gpt-4o
family's drift pair (PLAN 0.7), so it lands before that pair is run.

**What a drift pair is.** The same configuration, twice. Both sides' pins
are compared key by key over the keys they share; any difference is a
refusal (`DriftPairError`), because a prediction that changes under changed
pins is a configuration difference and calling it drift would launder it.
Two shared pins are reported rather than refused: `artifact_schema` (a bump
that only adds a pin is not a configuration change) and `harness_git_sha`
(the published restart pair spans three commits and `/4 → /5`; a code change
that moves nothing is exactly what a drift pair may show, and one that moves
rows is named in the report so nobody quotes it as reader drift). Judging on
shared pins rather than on `pins_hash` is what lets the published pair be
checked: `tests/test_drift.py` now asserts
`results/published/mnimi__100q` vs `mnimi__100q_restart_2026-09-10` is
0/100 changed, 0/100 prompt-token changes, by the instrument — the same
figure the 2026-09-10 hand comparison reported.

**Not a pin.** `drift.json` is a diagnostic beside the artifact, like
`reader_resolved.json`; it never enters `pins_hash` and is not one of the
three promoted files. A refused pair still leaves the fresh run's own
predictions on disk.

**Also in this change:** the batch resume hint printed after
`--batch-no-wait` now repeats every non-default flag the pins hash depends
on. It omitted `--render-format`, so the hint for a JSON-renderer submission
would have walked the user into the resume's own pins-hash refusal (found
while submitting the presentation pair; the four pair submissions were
resumed with the flag spelled out by hand).

## The Batch API enqueued-token cap: a run is a sequence of sub-batches (2026-09-12)

**What happened.** The four presentation-pair submissions (oracle and mnimi
at n=100, text and JSON) all failed validation within minutes:
`token_limit_exceeded` — "Enqueued token limit reached for gpt-4o-2024-08-06
in organization … Limit: 90,000 enqueued tokens." Nothing ran and nothing
was charged (the batches report 0/0). The cap is the organization's usage
tier for the Batch API, per model; an n=100 arm is ~520–580k tokens, six
times over it, and the 2-question smoke batch of task 0.2 was the only size
that could ever have validated. The ledger had booked the four at their
projections ($4.45); those lines are superseded by `void` lines at $0.

**Decision.** The harness plans every batch run as **sub-batches under the
cap** and submits them one at a time, each after the previous reached a
terminal status: `evals/batch.py` `plan_chunks` (greedy, order-preserving;
per-request cost = the trim gate's chars/4 input estimate + `max_tokens`,
the same upper bound the cost gate uses), `ENQUEUED_TOKEN_LIMIT = 90_000`
(`--batch-enqueued-tokens` overrides it for one run and is read back from
the state file on resume), `batch_state.json` schema 2 with a `chunks` list
(schema-1 files upgrade in place), per-chunk `batch_chunk_NN.jsonl` beside
the full `batch_requests.jsonl`, and a chunk whose batch fails validation
without enqueueing anything (the cap, while another batch is still in
progress) is resubmitted up to three times. `--batch-no-wait` submits the
first chunk only; the resume drives the rest. `reader_resolved.json` keeps
the flat keys (ids joined, counts summed, worst status) plus a per-chunk
`batches` list and the cap.

**Not a pin.** Chunking changes nothing the reader sees: the request bodies
are the ones already on disk, every chunk is a batch of the same
configuration, and a chunked run and a single-batch run share a
`pins_hash`. The cap is a fact about this key, recorded like the ledger.

**What it costs.** Time, not money: at ~6.5k tokens per request a chunk is
~13 requests, an n=100 arm is 8 chunks in series, and only one arm can be in
flight per key. Small batches have returned in minutes so far; the plan's
"up to 24 h per submission" now reads per chunk, in the worst case. The
alternative — synchronous calls at standard rate — would have put the
programme at ~$57 against the $50 cap (the n=500 final alone doubles from
~$12 to ~$24), so chunking is the mechanism that keeps Phase 5 affordable,
not just Phase 0.

**Also:** the four dead run directories were deleted (they held only
requests and pins; the ledger keeps the record), and the pair was resubmitted
at the harness commit that carries this change, since the harness sha is a
resume pin and a parity field and the whole sitting must share one.

## The presentation pair is read: the text renderer stays (2026-09-12)

Pre-registered rule (above): JSON becomes the era's renderer only if strictly
higher than text on both mnimi and oracle at n=100. Result, judged by the
pinned judge, `provisional: []` on all four:

| arm | text | JSON | b (JSON only) | c (text only) | p (exact McNemar) | fed tokens text / JSON |
| --- | --- | --- | --- | --- | --- | --- |
| oracle | 90/100 | 88/100 | 2 | 4 | 0.69 | 4,992 / 5,508 |
| mnimi | 75/100 | 71/100 | 1 | 5 | 0.22 | 4,537 / 5,057 |

JSON is lower on both arms, so **the text renderer is the era's renderer**;
`MemoryConfig.render_format` keeps its `"text"` default and `RENDER_TEMPLATE`
(hash `9c03ddae5c33…`) stays the pinned format for every arm through Phase 5.
Neither difference is significant and none is claimed: what the rule decided
is which of two framings to freeze, not that one is better. Two observations
for the record, not for a claim: JSON costs ~10% more fed tokens for the same
content (the indentation and quoting), and the paper's own finding (§5.5, Fig.
6) was that JSON helps *with* Chain-of-Note on the paper's prompt — our
`mnimi-con-v1` is that prompt with two deviations, and on this slice the
effect did not appear. The JSON format stays in the library as an option;
FUTURE.md's item closes with these numbers. The JSON arms are published as
`results/published/{oracle,mnimi}__100q_gpt4o_json/` — the evidence for the
decision, not the headline.

## The gpt-4o-era number, and what this family can say about reproducibility (2026-09-12)

**The sitting.** Four arms at n=100, harness `dd73736` clean, one day, $5.04
of the $50 budget for everything (pair, sitting, drift pair, all judging):
`no_memory` 5, `oracle` 90, `naive_rag` 80, **`mnimi` 75**; `full_history`
cited from the paper at 64.0. Full provenance, the paired tests and every
caveat: `results/published/README.md` § "The gpt-4o-era number". The
pre-registration's primary — mnimi vs naive_rag — is **a null, as
pre-registered**: b=1, c=6, p=0.125. Secondaries: mnimi vs no_memory b=71,
c=1; mnimi vs oracle b=2, c=17, p_holm=0.0007. Oracle lands inside the
paper's 92.4 for this snapshot with Chain-of-Note; the reader move did what it
was meant to do — the bound is now 90, not 50, and the memory layer is what
decides the number.

**The direction of the primary, recorded as the pre-registration asked.** Six
discordant rows for naive_rag against one for mnimi, after the local family's
six against none; 2 of the six question ids are the same on both
families (1cea1afa, c6853660). The categories where
mnimi trails naive_rag are knowledge-update (11/16 vs 13/16) and
temporal-reasoning (10/17 vs 12/17) — reported for completeness, no claim at
these cell sizes. The v1 write side is a dedup screen and nothing else, and
twice now the screen has cost rows it did not earn back. This is R3's input
(PLAN 1.2: classify the rows with the retrieval probe, fix without touching
the threshold) and, per the pre-registration, is not read as a harness
asymmetry because it is not significant.

**Tier 2 for this family, measured.** The drift pair — mnimi predicted twice
under identical `pins_hash`, 65 minutes apart — changed **85/100 answers at
the byte level** (median first divergence at byte 154 of ~860-character
step-by-step answers; 13 rows diverge inside the first 50 bytes), with
`reader_prompt_tokens` identical on all 100 rows and three `system_fingerprint`
classes served across the sitting. The judged score held at 75 both times,
with 3 rows flipping each way — inside the judge's own measured flip rate.
`seed=0` and temperature 0 do not make gpt-4o deterministic, as OpenAI
documents; the family's Tier 2 statement is therefore **"score reproducible
within 6/100 row flips; predictions not reproducible as text"**, the mirror
image of the local family's 0/100. Every artifact of this family carries that
sentence wherever its score goes. The instrument that produced it is
`python -m evals.drift results/published/mnimi__100q_gpt4o
results/published/mnimi__100q_gpt4o_drift_2026-09-12` (the `--verify-drift`
flag wrote the same report beside the fresh run).

**What the era's first number changes in the plan.** 85 is now a real
target: oracle is 90 on this slice, naive_rag 80, and published systems on
this reader report 71–85. mnimi at 75 with a v1 write side is 10 rows short of
the Wilson-lower-bound form of the criterion at n=100 (92/100) and 5 behind
naive_rag on the same rows; the Phase 1 lifts (R3 first) and the extraction
era are what the plan spends on next. Phase 0 closes at v1.7.0 with one item
open: 0.8, pausing NVIDIA driver auto-updates, is a settings action on the
development machine and is recorded here when done.

## NVIDIA driver auto-updates paused; the driver is the next unpinned layer (2026-09-12)

**Decision (PLAN 0.8):** automatic driver updates are switched off in the
NVIDIA app on the development machine, with the driver at **595.95** (CUDA
13.2) — the value every published artifact records in `run.environment`.
The reader no longer runs on this GPU on the gpt-4o family, but the embedder
(`BAAI/bge-small-en-v1.5` through onnxruntime) does, and the extractor will
(Phase 2, in-process llama-cpp-python with CUDA). A driver move is therefore
a potential move in every vector and every retrieved set, on both families,
and it is not a pin: `driver_version` is captured as resolved, never
requested, because a driver cannot be asked for per run. The Ollama tray
updater already moved the reader build three times under identical pins
(2026-09-10); this is the same failure mode one layer down, closed the same
way — the updater is off, the value is recorded, and a change shows up in
`run.environment` before it shows up in a score.

What this does not do: it does not make the driver reproducible for anyone
else. A cross-machine reproduction runs on whatever driver that machine has
and compares by `python -m evals.drift`; the embedder's vectors are the thing
to check first if that pair diverges (ingest is deterministic given the ONNX
graph and the driver, and `reader_prompt_tokens` identical on every row is
the sign that retrieval did not move).

## Phase 1 pre-registration: the probe gates, the paired-sitting protocol (2026-09-12)

Recorded before any Phase 1 run. Baseline for the phase: `mnimi__100q_gpt4o`
(75/100) and `naive_rag__100q_gpt4o` (80/100), harness `dd73736`, the same
100 question ids throughout.

**Protocol for every experiment (R3, R4+R5, R6).**
1. The knob is a `MemoryConfig` field with a harness flag and a pin
   (`retrieval_pins()` + `build_pins`, artifact schema bumped), so baseline
   and variant are the same code at the same commit.
2. The retrieval probe (`python -m evals.probes.retrieval`) runs first, on
   the n=100 slice, for baseline and variant: deterministic, CPU, no API.
   The probe gate is stated per experiment below; a variant that fails it is
   not sent to the API.
3. One sitting per experiment: baseline arm and variant arm(s) predicted
   through the Batch API at one clean commit, judged, paired with
   `python -m evals.stats`. The baseline re-run doubles as a drift
   measurement against `mnimi__100q_gpt4o` (`python -m evals.drift`).
4. Adoption: the probe gate must pass AND the paired score must not lose
   (b ≥ c on variant-vs-baseline). No significance is required to adopt —
   n=100 cannot detect a 3-point gap — and none is claimed; the direction
   plus the probe's deterministic evidence is the decision. A variant that
   loses on the paired run is reverted regardless of the probe.
5. The family's reader drift (6/100 flips between identical runs, measured
   2026-09-12) is the noise floor on every paired result here; a delta
   inside it is reported as "within drift".

**R3 — dedup scope.** Three of the six rows naive_rag won on this family
(`1cea1afa`, `67e0d0f2`, `cc6d1ec1`) are three of the five rows where the
0.95 screen dropped an annotated evidence round (RETRIEVAL §4); the local
family's `dcfa8644` is a fourth. Hypothesis: a near-duplicate from a
*different session* is a repeat or an update, not a duplicate — the date is
information — so the cosine screen should apply within a session only.
Probe gate: the number of evidence rounds dropped by the cosine screen must
fall to 0 on the slice, ALL-evidence recall@10 must not fall, ANY@10 must
not fall. Prediction: cosine drops fall from 570 to a small within-session
residue; mnimi's retrieved sets become identical to naive_rag's on every
question without a within-session near-duplicate. If adopted,
`dedup_scope="session"` becomes the default and the SPEC's dedup section
gains a "v1 as built" note that cross-session dedup is deferred to the
extraction era (where `valid_time` makes it a supersede, not a drop).

**R5 — query instruction.** BGE v1.5's model card recommends the prefix
"Represent this sentence for searching relevant passages: " on the query
for short-query-to-long-passage retrieval; mnimi embeds the bare question.
Probe gate: ANY@10 and ALL@10 both ≥ baseline, and the rank of the best
evidence round improves on more questions than it worsens. Stored vectors
do not move (query side only), so no `memory_meta` change.

**R4 — chunking clipped rounds.** 41.95% of rounds exceed BGE's 512-token
window and are truncated at embed time (09-08 §C13); the evidence in the
tail of a long round is invisible to retrieval. Rounds over
`chunk_tokens` are embedded as overlapping windows, one record per window,
all windows sharing one `round_key` so the reader sees the round once.
Probe gate: ALL@10 must rise by ≥ 3 questions and ANY@10 must not fall;
the dedup drop rate is reported (the 0.95 threshold is not re-selected —
if it behaves differently on chunks, that is reported, not tuned). R4 and
R5 are probed separately (four probe runs: base, +R5, +R4, +both) and sent
to the API once, as the combination the probe admits; `naive_rag` is
re-run under the same knobs because ingestion granularity must match.

**R6 — `top_k=20`, the one pre-registered alternative.** The paper (§5.2)
shows GPT-4o still improving past 20k retrieved tokens where weak readers
do not. Probe: ANY@20 and ALL@20 are read off the existing k=50 search
(RETRIEVAL §1: 92/95 and 83/95 at k=20 vs 91 and 77 at k=10 on the old
configuration). Sitting: k=10 vs k=20 at the then-current configuration.
Adoption requires b > c AND Holm-significance of mnimi-vs-naive_rag not
worsening; otherwise k=10 stays. No other k is ever run.

**Error analysis (1.5)** runs after the sittings, on the era's mnimi
misses: each miss is classified retrieval-miss (no evidence round in the
top-k) or reading-miss (evidence present) from the probe, and the R7
(hybrid FTS5: exact-term misses) / R8 (bge-base) triggers are read off it.

## R3 gate amended before the R3 probe finished: cross-session losses (2026-09-12, 18:55)

The rebuilt probe's baseline run (Task 2) reproduced the July retrieval on
every count it could be checked on — ANY@k 50/77/91/92/94 of 95 at
k=1/5/10/20/50, 24,747 rounds, 16 exact and 570 cosine drops, and the same
five questions losing an evidence round to the cosine screen (`gpt4_31ff4165`,
`67e0d0f2`, `dcfa8644`, `1cea1afa`, `cc6d1ec1`) — and measured one thing July
could not: **which session the dropping neighbour came from.** Of the 570
cosine drops only 27 (on 24 questions) are cross-session; the rest are
within-session repeats. Of the five lost evidence rounds, four were dropped
against a *different* session and one, `67e0d0f2` (cosine 0.955), against
its own.

The pre-registered R3 gate said "evidence rounds dropped by the cosine screen
must fall to 0". Session scope leaves within-session drops untouched by
construction, so that line was written on a fact not yet measurable and
cannot be met by the mechanism the hypothesis names, whatever the probe says.
**Amended, before `probe_mnimi_r3.json` exists and before any API spend:**
the R3 gate is *evidence rounds lost to a cross-session drop must be 0*
(expected: the four), *ALL@10 and ANY@10 must not fall*, and within-session
losses are reported (expected: `67e0d0f2` alone). Disclosed plainly: because
the probe is deterministic, the baseline data already implies the outcome;
the amendment is recorded with its time so the sequence is auditable, and
it narrows what R3 claims rather than widening it — R3 is a fix for the
cross-session case, and `67e0d0f2` stays on the list of rows naive_rag wins
that the next phase still owes an explanation for.

**ALL@k is recorded, not matched.** The rebuilt probe reads ALL@10 74/95
(July: 77/95) and naive_rag ALL@10 77/95 (July: 79/95) because it
tags more evidence rounds on a few multi-evidence questions (`b46e15ed`: five
rounds carry `has_answer` turns; July's table listed three ranks) and July's
tagging script is lost. ANY@k, the drop counts and the lost rows are
identical, so the retrieval is reproduced; the ALL denominator is this
probe's from now on, and every Phase 1 ALL@k gate compares within it.

## R4 corrected before its sitting: `k` counts rounds, not windows (2026-09-12, 22:00)

The R4 probe (chunking at 510 tokens, overlap 64, on the R3 session scope)
read ANY@10 **86**/95 against 91 and ALL@10 **70** against 77 — a loss,
where the pre-registration expected a gain — with 34,489 stored records
against 24,195. The mechanism is in the stored count: a long round embedded
as several windows puts several records into the top-10, the reader sees
that round once (its windows share a `round_key`), and the other slots are
gone. `k` was being counted over windows while the reader, the renderer and
every recall number count rounds.

**Correction, before any R4 sitting and before the R4+R5 probe is read:**
`Store.search_rounds` — the read path over-fetches windows and keeps each
round's best-ranked window until `k` distinct rounds are in hand; with no
windowed records it is `Store.search` exactly, so every non-chunked
configuration (and every published artifact) is byte-for-byte unchanged.
`Memory.recall`, `naive_rag` and the probe's k=50 search all go through it.
The probe is re-run for R4 and R4+R5 under the corrected read path; the R4
gate (ALL@10 ≥ baseline + 3, ANY@10 not lower) is unchanged. This is a
design correction the probe caught for $0 — the reason the probe exists —
not a tuning step: no threshold, no `k`, no prompt moved.

**R5 read (same probe run, before the sitting):** on the R3 scope the BGE
query instruction leaves ANY@10 at 91 and ALL@10 at 77 (gate: not lower —
met), lifts ALL@20 85 → 86, ALL@50 89 → 91, ANY@1 50 → 51 and ALL@1 24 →
27, and moves the best evidence rank up on 13 questions and down on 11 (gate:
improved > worsened — met, by two). Admitted to the sitting as a marginal
positive; whether it earns a point is the sitting's question.

## R3 read: the dedup scope is the session (2026-09-12, 22:15)

**Sitting:** two mnimi arms at `b259a4a` (clean), 100 questions, the Batch
API in 7 sub-batches each, judged by the pinned judge, `provisional: []`:

| arm | score | Wilson 95% | fed tokens |
| --- | --- | --- | --- |
| `mnimi__100q_gpt4o_r3base` (store scope) | 77/100 | [67.8, 84.2] | 4,537 |
| `mnimi__100q_gpt4o_r3` (session scope) | 79/100 | [70.0, 85.8] | 4,546 |

Variant pair (uncorrected, b = the variant's wins): **b=5, c=3**, p=0.73.
Rule (Phase 1 pre-registration): probe gate passed (four cross-session
evidence losses recovered, none lost, ALL@10 74 → 77) and b ≥ c — **adopted**.
`MemoryConfig.dedup_scope` defaults to `"session"` from this commit; the
harness and probe flags default to the library's value.

**What the five wins were.** `1cea1afa` (knowledge-update) and `cc6d1ec1`
(temporal-reasoning) are two of the four evidence rounds the probe said
session scope recovers, and both were rows naive_rag won on the published
sitting — the mechanism paid where it was predicted to. The other three
(`b46e15ed`, `gpt4_c27434e8_abs`, `7405e8b1`) and the three losses
(`1c0ddc50`, `a9f6b44c`, `8077ef71`) sit inside the family's measured drift
(3 flips each way between identical runs), and nothing is claimed about
them. Two points at n=100 is not significance and is not reported as such;
the deterministic evidence is the probe's, the sitting says the change did
not lose.

**The baseline arm as a drift reading.** Identical pins to the published
`mnimi__100q_gpt4o` except the harness commit: 61/100 predictions changed
at the byte level, 0 prompt-token changes, score 77 against the published
75 — a second sample of the family's drift, consistent with the first
(85/100 changed, 0 score delta). The family's Tier 2 sentence stands.

**Not a threshold change.** 0.95 is untouched; what changed is which
neighbours the screen is allowed to look at. Cosine drops fell 570 → 536 and
the cross-session share of them 27 → 0; 34 more records are stored per
100-question corpus (24,161 → 24,195). The remaining within-session loss,
`67e0d0f2`, is still on the list the next phase owes.

**Reading against naive_rag** is deferred to the R4+R5 sitting, where
naive_rag is re-run at the same commit (the published naive_rag arm is at
`dd73736` and cannot be paired across commits).

## R4+R5 sitting pre-registered from the probes (2026-09-13, automated)

Corrected read path (`Store.search_rounds`), R3 scope throughout. Baseline =
the R3 probe (ANY@10 91/95, ALL@10 77/95). R4 gate: ALL@10 >= 80
and ANY@10 >= 91.

| probe | ANY@10 | ALL@10 | ANY@20 | ALL@20 | stored | cosine drops |
| --- | --- | --- | --- | --- | --- | --- |
| R5 | 91 | 77 | 92 | 86 | 24195 | 536 |
| R4 (corrected k) | 89 | 75 | 93 | 84 | 34489 | 840 |
| R4+R5 (corrected k) | 89 | 75 | 93 | 86 | 34489 | 840 |

**Admitted to the sitting: R5 alone (R4 failed its gate under the corrected k)** — flags `--query-instruction bge`. Arms at one clean
commit: mnimi baseline (library defaults), mnimi variant, naive_rag variant.
Rule: adopt iff b >= c on the variant pair; the mnimi-variant vs
naive_rag-variant pair is the primary re-measured under this configuration
(still a pre-registered null at v1). The line above was written by the gate
script from the probe files before submission.

## R6 pre-registered after the R4+R5 sitting (2026-09-13, automated)

R4+R5 variant pair: b=5, c=3 — **adopted** by the rule
(scores: baseline 79, variant 81, naive_rag under the
variant 82). The era's configuration for R6 is therefore
`--query-instruction bge`.

Probe reading for that configuration, off its k=50 search: ANY@10 91
→ ANY@20 92; ALL@10 77 → ALL@20 86 (of 95).
Sitting: two mnimi arms at one clean commit, `--top-k 10` and `--top-k 20`,
both with the flags above. Rule: k=20 is adopted only if b > c on the variant
pair (and it may not worsen mnimi vs naive_rag, read at the next naive_rag
re-run); otherwise k=10 stays. No other k is ever run. Written by the gate
script before submission.

## R5 read: the query instruction is on; R4 stays off (2026-09-13, 04:00)

**R4 at the probe, under the corrected `k`:** ANY@10 89/95, ALL@10 75/95
against the R3 baseline's 91 / 77 (gate: 80 / 91) — rejected without an API
call. Counting `k` over rounds removed the crowding (86 → 89) but chunking
still loses two evidence rounds at k=10: windows of long, irrelevant rounds
compete with whole evidence rounds, and the per-window cosine screen drops
more (840 vs 536). The knob stays in the library at `chunk_tokens=0`; the
long-round problem (42% of rounds clipped) is real and still open, and the
extraction era changes the embedded unit anyway — R4 is re-examined there,
not tuned here.

**The sitting** (three arms at `b666729`, `provisional: []`; the automated
gate admitted R5 alone):

| arm | score | Wilson 95% | fed tokens |
| --- | --- | --- | --- |
| `mnimi__100q_gpt4o_r45base` (session scope, bare question) | 79/100 | [70.0, 85.8] | 4,546 |
| `mnimi__100q_gpt4o_r45` (+ BGE query instruction) | 81/100 | [72.2, 87.5] | 4,515 |
| `naive_rag__100q_gpt4o_r45` (+ BGE query instruction) | 82/100 | [73.3, 88.3] | 4,534 |

Variant pair b=5, c=3, p=0.73 — **adopted**: `MemoryConfig.query_instruction`
defaults to `BGE_QUERY_INSTRUCTION` from this commit; the harness and probe
flags default to the library's value. Wins `852ce960`, `b5ef892d`,
`09ba9854`, `38146c39`, `8077ef71`; losses `b46e15ed`, `7405e8b1`,
`6f9b354f` — inside the family's drift, no claim on any row; the probe's
deterministic reading (equal at k=10, better at k=20/50, 13 up / 11 down on
the best evidence rank) is the evidence, the sitting says it did not lose.

**The primary, re-measured under the adopted configuration:** mnimi 81 vs
naive_rag 82, **b=2, c=3**, p=1.0 — the pre-registered null, and no longer
leaning: the published sitting read 1 / 6 on the same rows. naive_rag-only
rows: `67e0d0f2` (the within-session loss R3 could not touch), `a9f6b44c`,
`7405e8b1`; mnimi-only: `b5ef892d`, `09ba9854`. v1's write side now costs
nothing measurable against verbatim storage on this benchmark, which is the
most it can claim before extraction.

**Drift, third reading:** the baseline arm vs the published R3 arm — 65/100
changed at the byte level, 0 prompt-token changes, score 79 → 79.

## R6 read: k stays 10 (2026-09-13, 04:00)

Two mnimi arms at `0dcf09f` (session scope + BGE instruction), the one
pre-registered alternative:

| arm | score | Wilson 95% | fed tokens |
| --- | --- | --- | --- |
| `mnimi__100q_gpt4o_k10` | 82/100 | [73.3, 88.3] | 4,515 |
| `mnimi__100q_gpt4o_k20` | 81/100 | [72.2, 87.5] | 8,819 |

Variant pair **b=6, c=7**, p=1.0 — rule: adopt only if b > c — **k=10 stays**.
Probe reading for this configuration: ANY@10 91 → ANY@20 92, ALL@10 77 →
ALL@20 86 of 95 — nine more questions have all their evidence in context
at k=20, and the reader converts none of that into score while reading
twice the tokens. Consistent with the paper's Table 3 for capable readers
(top-10 ≈ top-20) and the end of the `k` question for this era: no other
value is ever run. The k=10 arm is a fourth drift reading against the R5
variant arm: 61/100 changed, 0 prompt-token changes, 81 → 82.

## Phase 1 error analysis: R7 at the margin, R8 does not fire (2026-09-13)

`python -m evals.probes.misses runs/probe_mnimi_r5.json
results/published/mnimi__100q_gpt4o_k10/results.json --oracle … --naive …`
on the era's final configuration (session scope, BGE instruction, k=10;
82/100):

| | wrong rows | oracle right on them | naive_rag right on them |
| --- | --- | --- | --- |
| retrieval-miss (no evidence round in the top-10) | 4 | 4 | 0 |
| reading-miss (evidence in the top-10, still wrong) | 14 | 8 | 1 |

By category: multi-session 0 / 6, temporal-reasoning 2 / 3, single-session-
preference 1 / 2, knowledge-update 0 / 2, single-session-user 1 / 1
(retrieval / reading). Six of the fourteen reading misses oracle also gets
wrong — the reader's own floor on this slice (`35a27287`, `a3838d2b`,
`3a704032`, `a2f3aa27`, `0a995998`, `09d032c9`).

**R7 (hybrid FTS5) — trigger met at the margin, not built.** Three of the
four retrieval misses carry the question's own words verbatim in an
evidence round that ranked 15–36: `af082822` (*nordstrom*, *friends and
family sale*, rank 28), `6f9b354f` (*repaint*, *bedroom*, *walls*, rank
15), `b46e15ed` (*charity*, *events*, ranks 25–40). Each evidence turn is
a long multi-topic round whose dense vector is dominated by other content —
the same long-round problem R4 was for. The pre-registered trigger was ≥ 3
exact-term misses; it is met, and the ceiling is four rows at n=100 against
fourteen reading misses. Ruling: R7 stays deferred, trigger recorded as
met, to be built only if the extraction era's error analysis shows the same
shape after facts are the embedded unit (which changes the retrieval
problem entirely). Cross-encoder rerank: not indicated — every reading miss
already has its evidence inside the top-10, so ordering is not the failure.

**R8 (bge-base) — does not fire:** retrieval misses (4) do not outnumber
reading misses (14).

**Where Phase 2 starts.** Fourteen of eighteen misses are reading misses,
eight of which oracle gets right — i.e. the reader can answer from the
evidence *sessions* but not from mnimi's ten verbatim rounds. That is the
extraction thesis in one number: the rounds carry the evidence and the
reader does not find it in them. multi-session (6 reading misses, 62.5%)
and temporal-reasoning (3, 70.6%) are where it is lost.

## Phase 1 closes (2026-09-13, v1.8.0)

Adopted: `dedup_scope="session"` (R3), `query_instruction=BGE_QUERY_INSTRUCTION`
(R5). Rejected: chunking (R4, at the probe), k=20 (R6, at the sitting).
Instruments: `evals/probes/{retrieval,aggregate,misses}`, `--verify-drift`,
`Store.search_rounds`, variant pairs in `evals.stats`. **mnimi on the
gpt-4o family: 75 (v1, 2026-09-12) → 79 (R3) → 81/82 (R5, two sittings);
naive_rag under the same configuration 82; primary b=2, c=3.** API spend
for the phase $5.69; programme total $10.75 of $50. Every
sitting `provisional: []`, one commit per sitting, every artifact in
`results/published/`. The 85 criterion is read at n=500 in Phase 5, not
here; at n=100 the adopted configuration's Wilson lower bound is 73.3.

## Phase 2 pre-registration: the extraction era's design, gates and sitting (2026-09-13)

Recorded before any extractor runs on the slice. Baseline for the phase: the
adopted v1.8.0 configuration — `mnimi__100q_gpt4o_k10` (82/100),
`naive_rag__100q_gpt4o_r45` (82/100), primary b=2, c=3 — and the probe file
`runs/probe_mnimi_r5.json` (ANY@10 91/95, ALL@10 77/95, 24,195 stored, 16
exact + 536 cosine drops, one evidence round lost: `67e0d0f2`). The
task-level plan is `mnimi docs/PHASE2.md`; its numbers doc is
`mnimi docs/PHASE2-RESULTS.md`.

**Where the misses are, read by hand** (`mnimi docs/PHASE2-RESULTS.md` §0, all
18 wrong rows of the k10 arm). Ten rows are partial-retrieval misses: the
evidence is a one-clause aside inside a long round about something else
("… Can you suggest tips? By the way, I've been doing guided breathing
sessions with my Fitbit …"), the round's vector is the assistant's answer,
and the round ranks 11–40 (`gpt4_31ff4165`, `b46e15ed`, `af082822`,
`45dc21b6`, `gpt4_d6585ce8`, `gpt4_2ba83207`, `75832dbd`, `3a704032`,
`0a995998`, `6f9b354f`). Two rows lost an evidence round to the
within-session cosine screen because the session is a run of near-template
rounds whose *facts* differ (`67e0d0f2`: 8 edX vs 12 Coursera courses;
`a3838d2b`: six dated charity events). Seven rows need a relative date
resolved against the session date ("yesterday", "today", "last month", "two
weeks ago", "the week before last", "recently"). Four rows are reader-bound
(evidence at rank ≤ 4, oracle also wrong or judge-borderline: `35a27287`,
`15745da0`, `a2f3aa27`, `09d032c9`). That is the design: fact records as
extra retrieval keys on the round (LongMemEval's measured key expansion),
per-kind dedup so a fact survives its round's near-duplicate, a deterministic
date resolver, and a reader-visible facts header on the round.

**Design (PHASE2.md D1–D12), in one paragraph.** Hybrid store: every round
keeps its v1 record, byte-identical embed text; each extracted fact is a
second record on the same `round_key`, embedded as `raw\nfact` under a new
`fact_embed_template_hash`; dedup screens run per kind and per session at
the untouched 0.95; retrieval collapses to rounds and `k=10` rounds for every
arm; the reader sees a round as its turns plus a `facts:` header (primary)
or as its facts alone (the one pre-registered alternative), a pinned render
unit that is a system-level pin, not a parity field. Extractor:
`Qwen/Qwen3-1.7B-GGUF` @ `90862c4b9d2787eaed51d12237eafdfe7c5f6077`,
`Qwen3-1.7B-Q8_0.gguf` (the only quant in the official repo; there is no
official "Qwen3-1.7B-Instruct" — SPEC's name resolves to this post-trained
hybrid with thinking disabled through the chat template's empty think
block), llama-cpp-python built from source with CUDA 13.2, greedy
(`temperature=0, top_k=1`), `seed=0`, `n_ctx=4096`, `n_batch=512`,
`n_ubatch=512`, `n_threads=8`, `n_gpu_layers=99` (a disclosed deviation from
SPEC's CPU-only pin, on the reader precedent: GPU is deterministic with the
batch pinned), one sequence, a cold prefill per round (`reset()` before every
call — the reader's prompt-cache lesson), input capped at 1,536 tokens.
Output schema, prose first: `content, raw, when, subject, predicate, object,
salience`; `when` is a verbatim time mention and a deterministic resolver
anchored on `ts` produces `valid_time` (SPEC's model-emitted ISO date is not
asked of a 1.7B model that is never shown the date). Pre-filter: turn-level
lexicon + rules, never "question → drop" (16 of the 18 evidence turns above
are questions carrying the evidence as an aside). Cache: one SQLite file per
extractor configuration, keyed by the round's turns. Nine new `memory_meta`
keys; pins schema /8. Fallback E1 after one four-hour day: an Ollama-served
extractor on 0.32.13 with `OLLAMA_NUM_PARALLEL=1` (parallel slots make a
round's logits depend on its batch-mates), disclosed. The extractor is
injected as `Memory(db_path, embedder, config, *, extractor=None)` — one
defaulted keyword on the locked constructor; `None` is the v1 write path.
All twelve confirmed by the maintainer on 2026-09-13.

**Protocol.**
1. Nothing about the prompt, the lexicon or the grammar is developed on the
   slice's evidence rounds: the prompt's development set is 40 rounds from
   the 400 questions outside the slice; the byte-stability set is 50 rounds
   of the slice's corpus stratified by length; the pre-filter's false-drop
   check reads the slice's evidence rounds and must be 0 (a correctness
   check, disclosed).
2. Every knob is a `MemoryConfig` field, a harness flag and a pin, so the
   baseline arm (`--extractor none`, `--render-unit turns`) is the same code
   at the same commit as the extraction arms.
3. The corpus pass runs once, through the retrieval probe, into the cache;
   the sitting's ingest is cache hits.
4. Work happens on `feature/extraction` (worktree `../mnimi-wt`), one commit
   per plan task, fast-forwarded onto `main` at the phase close so every
   artifact's `harness_git_sha` stays on `main`'s history.

**Gate 4-i (probe, no API, PHASE2 Task 7).** `python -m evals.probes.retrieval
--system mnimi --extractor qwen3 --limit 100`: **ANY@10 ≥ 89/95 (≥ 93%) and
ALL@10 ≥ 77/95**, otherwise extraction stops here and is re-scoped. Reported
beside the gate: drop rate by kind and by screen, the `[]` rate, the
truncation rate, facts per round, stored counts; and two predictions,
falsifiable before any spend — (P1) `67e0d0f2` and `a3838d2b` no longer
lose an evidence round to dedup; (P2) of the ten partial-retrieval rows, at
least four have every evidence round inside the top-10.

**Gate 4-ii (n=20 dev prefix, batch, ≈ $0.25, Task 8).** The `round+facts`
arm on the first 20 stratified questions (the 0.95 dev slice): its score must
be within 3 of the k10 arm's score on the same 20 rows, and its fed tokens
are reported against the k10 arm's. This is the early-wrongness gate, not a
pairing; a fail stops the sitting.

**Gate 4-iii (the sitting, n=100, one clean commit, Task 9).** Four arms
through the Batch API, one at a time: mnimi `--extractor none` (baseline;
`--verify-drift results/published/mnimi__100q_gpt4o_k10`, and its
`reader_prompt_tokens` must be identical on 100/100 rows — the refactor
moved no retrieval), mnimi `--extractor qwen3 --render-unit round+facts`,
mnimi `--extractor qwen3 --render-unit facts`, `naive_rag`. Rules:
- Extraction is adopted (library default `render_unit="round+facts"` and the
  harness's mnimi default `--extractor qwen3`) iff gate 4-i passed AND b ≥ c
  for `round+facts` vs baseline. A paired loss reverts the default; the code
  stays.
- The render unit `facts` replaces `round+facts` iff b > c on that pair.
- **Primary of the extraction era:** the adopted mnimi arm vs `naive_rag`,
  exact McNemar, alpha 0.05, one test. It is no longer a pre-registered
  null: the expectation is a positive direction. At n=100 only a gap of
  ~8 points is detectable, so the n=100 reading is a working number and the
  verdict is Phase 5's n=500. The fed-token table for all four arms is part
  of the artifact. Secondaries (no_memory, oracle) are the published
  family arms at another commit — reported descriptively, not paired.
- The family's drift (6/100 flips) is the noise floor: a delta inside it is
  "within drift".

**Budget.** PLAN carried $2.2; this sitting is four arms (≈ $3.3 with
judges) plus gate 4-ii (≈ $0.25): **≈ $3.7, ≈ $4.7 with one re-run** —
$39.25 remains, Phase 5 needs $14.25.

## Extractor runtime: llama-cpp-python 0.3.35 with CUDA 13.2, measured (2026-09-13)

**Decision (PLAN 2.1, PHASE2 Task 2):** E2 stands. The extractor runs
in-process through `llama-cpp-python 0.3.35`, built from source on the run
machine against CUDA 13.2 (`V13.2.86`, host compiler MSVC 19.50.35729 from
VS 2026 Build Tools 18, CMake 4 + Ninja from the VS component) with
`CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF"`,
`FORCE_CMAKE=1`, `CMAKE_GENERATOR=Ninja`, installed with
`pip install --user --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.35`
into the Python 3.14 user site (the run environment). The E1 fallback was
not needed; the build took 13.5 minutes. The `[extract]` extra pins that
version; the string `llama-cpp-python 0.3.35; <CMAKE_ARGS>; cuda 13.2` is
the `extractor_runtime` row of `memory_meta` and of the pins.

**Two Windows facts the recipe depends on.** (1) Long paths are disabled
on this machine (`LongPathsEnabled=0`) and the sdist nests
`vendor/llama.cpp` deeply enough that extraction under
`%LOCALAPPDATA%\Temp\pip-install-*` fails at ~260 characters; the build
runs with `TMP=TEMP=D:\t` and `SKBUILD_BUILD_DIR=D:\t\b`. (2) CUDA 13 keeps
its runtime DLLs in `CUDA\v13.2\bin\x64`, and the binding loads `llama.dll`
with the legacy search order, so that directory must be on `PATH` (the
toolkit installer adds it to the machine PATH; a shell older than the
install has to add it by hand — every harness command in this phase is run
that way). Smoke log lines kept for the record: `ggml_cuda_init: found 1
CUDA devices`, `load_tensors: offloaded 29/29 layers to GPU`,
`llama_context: flash_attn = enabled`, `n_batch = 512`, `n_ubatch = 512`.

**The grammar sampler was the cost, and the decode loop is now
llama-server's.** On one 653-token round: prefill 0.20 s, greedy decode of
137 tokens without a grammar 2.87 s, and under *any* grammar — the pinned
schema's 5,964-character GBNF or a 1,074-character one without length
bounds — 10.3–11.3 s. llama.cpp's grammar sampler evaluates every one of
Qwen3's 151,936 vocabulary entries against the grammar at every step, ~55 ms
per token, and the size of the grammar does not matter. `QwenLlamaExtractor`
therefore decodes the way llama-server does: take the greedy token from the
raw logits, verify that one candidate against the grammar sampler, and only
on a rejection rescan the whole vocabulary under the grammar and take the
best valid token. This picks exactly the token the grammar-then-greedy chain
picks (the global argmax, when valid, is the valid argmax; otherwise the
rescan *is* the chain) — checked byte for byte on the smoke round, three
runs — at 13.5 ms per token. `DECODE` carries
`grammar_mode = "greedy-verify-rescan"`, so the mechanism is part of the
decode hash (`ba1a81ab349d…`); the earlier chain-based hash `d25ec146…` was
never used on the slice.

**Measured cost (Task 2 Step 6, `python -m evals.probes.extractor_bench
--rounds 50`, 50 rounds of the slice stratified over ten length deciles,
ids in `runs/extract_bench50.log`):**

| quantity | value |
| --- | --- |
| seconds per round p50 / p90 / mean | 2.56 / 5.21 / **2.71** |
| prompt tokens mean / completion tokens mean | 1,023 / 192 (p50 190, p90 436, max 844) |
| throughput | 448 tokens/s overall |
| facts per round | 1.80 (0 facts on 18/50, 2 on 16/50, 8 on 1/50) |
| outputs that hit `max_tokens` | 0 |
| inputs cut to the 1,536-token cap | 0 |
| **projected corpus pass** | **18.7 h** for 24,747 rounds (≈ 18.6 h after the 0.29 % pre-filter skip) — gate 2.1 (≤ 24 h): **PASS** |

**One parser correction found by the bench.** Two of the fifty outputs
(`06878be2/a67c1862_4/5`, `a2f3aa27/61d3fec4_4/4`) came back as `[]`
although the model had stopped cleanly at 239 and 229 tokens: their `raw`
excerpts of multi-line assistant turns carried real newlines, which this
binding's JSON grammar admits (`char ::= [^"\\] | "\\" (...)` — no
control-character exclusion) and which `json.loads` in strict mode rejects.
The grammar defines the language the model emits, so `schema.parse_output`
now parses with `strict=False`; re-parsed, the fifty outputs are 0/50 bad and
90 facts. The cache stores bytes, so a hit re-reads through the corrected
parser without a model call. Not a pin: the prompt and grammar are
unchanged.

**Deviation from SPEC, restated:** the decode runs with `n_gpu_layers=99`
(SPEC's pin said CPU-only). The reader precedent holds — deterministic with
the batch pinned — and the byte-stability run of Task 3 is the evidence for
this extractor; the CPU path would cost roughly an order of magnitude more
per round and was not measured.

## Extractor prompt frozen: `qwen3-fact-v4`, developed outside the slice (2026-09-13)

**Decision (PHASE2 Task 3 Step 5):** the extraction prompt is
`mnimi.extract.prompt.SYSTEM_PROMPT` version `qwen3-fact-v4`;
`extractor_prompt_hash` (prompt + chat template + schema + the GBNF the
runtime samples under) = `41c8ea2c3bb4901aa923e734e2e37d0b7d8ea29aef5a7fc473f1e5f8fead0993`. From this line on it is
pinned: an edit is a new era (new hash, new cache file, re-ingest), never an
in-place change under a run.

**How it was developed — and on what.** Forty rounds sampled (seed 0,
stratified over ten length deciles) from the 400 LongMemEval-S questions
*outside* the n=100 slice (`runs/extract_dev{,2,3,4}.jsonl`); the slice's
evidence rounds were never read during this step (pre-registration protocol
1). Each version was read by hand, whole, against the kinds of facts
PHASE2-RESULTS §0 asks for: asides with dates, counts and names; plans and
experiences; the assistant's specific recommendations; nothing for
chit-chat.

| version | facts | user facts | assistant facts | junk-shaped¹ | dated (`when`) | `[]` rounds | s/round | user facts on 7 plan/experience rounds² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (the plan's draft) | 98 | 46 | 51 | 33 | 2 | 6 | 3.70 | 15 |
| v2 (exclusion list, "or nothing") | 49 | 24 | 25 | 15 | 6 | 15 | 1.87 | 0 |
| v3 (permissive user rule + v2's date rule) | 65 | 36 | 23 | 15 | 11 | 17 | 2.43 | 6 |
| **v4 (v3 + a thanks-plus-plan example)** | **94** | **59** | 25 | 27 | **15** | 7 | 3.39 | **17** |

¹ facts whose content starts "The user asked/thanks…", "The assistant
acknowledges/is glad/encourages/provided information/explained…" — a regex
over the content, the same for every version. ² `36b9f61e`, `71315a70`,
`a3332713`, `982b5123/359a2a0d`, `59524333`, `6aeb4375/8de5b7cf`,
`gpt4_483dd43c`: user turns stating plans or experiences beside a
thank-you or a question, which v2 and v3 returned `[]` for in 0.3 s.

What each step taught. v1 captured the user's asides well but left `when`
empty when the date sat inside the sentence ("May 14th", "in 2017", "this
week") and produced a fact for every pleasantry. v2 fixed the noise by
naming what not to extract and lost half the user facts — a 1.7B model
over-applies a negative list, and "state the specific recommendation
instead, or nothing" became "nothing" for whole rounds. v3 restored the
permissive user-turn rule and kept the date rule (dated facts 2 → 11); four
plan/experience rounds still returned `[]` instantly because the only
`[]` example is a thank-you turn, and those rounds thank before they plan.
v4 adds one example of a thanks-plus-plan turn that yields facts; the four
rounds come back with their plans and counts, dated facts reach 15, user
facts exceed v1's, and the pleasantry noise stays below v1's. Junk-shaped
facts (27) are mostly "The user is asking for…" and "The assistant provided
tips on…" wrappers; they are extra keys on a round the reader sees anyway
and are left to the sitting to price.

**One deterministic guard came out of it.** v3 and v4 each produced two
facts whose `when` does not occur in the round ("next week" echoed from an
example; "last week" attached to a light bulb the user "finally" replaced).
`mnimi.extract.resolver.verbatim_mention` keeps a mention only if it occurs
in the round's text (case- and whitespace-insensitive); `round_pieces` drops
an invented mention and keeps the fact, so no `valid_time` can come from a
date the user never stated. The `[]` and encyclopedic-assistant behaviours
(a Malayalam explainer still yields six assistant facts under every
version) are accepted: D1's round record keeps every round retrievable, and
the facts are additional keys.

**Not tuned on the benchmark.** No evidence round of the slice was read; the
four versions were judged on the metrics above and on the hand reading of
the forty rounds. Gate 4-i (Task 7) is the first time the prompt meets the
slice.

## Byte-stability read: 50/50 across a shuffled restart; the corpus pass starts (2026-09-13, 16:45)

**Gate 2.2 (PHASE2 Task 3 Step 6):** `python -m evals.probes.extractor_bench
--rounds 50 --stability` — the 50 length-stratified rounds of the slice
extracted twice, each time in a fresh process (a fresh model load, a cold
CUDA graph cache), once in order and once shuffled (seed 1), under the frozen
prompt (`41c8ea2c3bb4…`) and the pinned decode (`ba1a81ab349d…`):
**50/50 raw outputs byte-identical.** A round's output depends on neither the
round before it (cold prefill, one sequence) nor its position — which is
what the cache key `sha256(turns)` assumes and what SPEC's "byte-stable
output across two runs" criterion asks. Determinism at `n_gpu_layers=99` is
therefore measured for this extractor too, as it was for the reader; the
CPU-only pin stays a disclosed deviation, not a fallback in use.

**Cost re-measured under the frozen prompt.** The same 50 rounds: mean
3.36 s and 3.14 s per round in the two passes (p50 2.7, p90 6.8–7.5),
prompt tokens mean 1,588 (the v4 instruction and its four examples are
~560 tokens longer than the draft), completion tokens mean 202, facts per
round 2.12, `[]` on 16 %, 0 outputs at the token budget, 0 inputs at the
cap — **projected 21.6–23.1 h** for the 24,747 rounds, inside gate 2.1's
24 h with less room than the draft's 18.7 h. Accepted: the prompt bought
user-fact recall and dates (DECISIONS "Extractor prompt frozen"), and the
pass runs once into the cache.

**The corpus pass (Task 7 = gate 4-i) started 2026-09-13 16:45**, detached
from the session: `python -m evals.probes.retrieval --system mnimi
--extractor qwen3 --limit 100 --out runs/probe_mnimi_extract.json` at
commit `e1805e5`, cache `.cache/extract/<pins hash>.sqlite`. Its reading
is the next entry.

## Gate 4-i read: the extraction arm's retrieval — ANY@10 93/95, ALL@10 83/95, PASS (2026-09-14, 20:01)

**The corpus pass finished** — `python -m evals.probes.retrieval --system mnimi
--extractor qwen3 --limit 100 --out runs/probe_mnimi_extract.json`, code at
`e1805e5` (the two commits since are docs), 2026-09-13 16:45 → 2026-09-14
20:01, exit 0. Every round of the slice's 100 histories went through the
pre-filter, the pinned Qwen3-1.7B extractor on the GPU, the resolver and the
per-kind screens, and every model output is in the cache
(`.cache/extract/e15153f8….sqlite`, 23,302 distinct rounds, 33.6 MB). No API
call was made; Phase 2 has spent $0 so far.

**The gate (pre-registered in "Phase 2 pre-registration"):** on the 95
evidence-bearing questions, against the no-extractor probe
(`runs/probe_mnimi_p2base.json`, identical to Phase 1's R5 read):

| | extract | p2base | bar | |
| --- | --- | --- | --- | --- |
| ANY@10 (some evidence round in the top-10) | **93/95** | 91/95 | ≥ 89/95 | PASS |
| ALL@10 (every evidence round in the top-10) | **83/95** | 77/95 | ≥ 77/95 | PASS |

Nine rows changed status: ANY gained 3, lost 1; ALL gained 7, lost 1.
Zero evidence rounds were lost to a screen (0 cross-session drops). Per
category (extract vs p2base, ALL@10): knowledge-update 14 vs 13,
multi-session 11 vs 8, single-session-assistant 17 vs 17,
single-session-preference 13 vs 12, single-session-user 16 vs 15,
temporal-reasoning 12 vs 12; ANY@10 moves only on single-session-user
(16 vs 15) and temporal-reasoning (15 vs 14).

**P1 holds.** `67e0d0f2` (ranks 7, absent → 14, 1) and `a3838d2b`
(2, 3, absent, absent, 25, 6 → 3, 5, 9, 14, 16, 7) lost no evidence round;
the fact records pulled their absent rounds into the top-50 and, for three of
the four, into the top-10 or its edge.

**P2 holds: 5 of the 10 partial-retrieval rows are complete** (bar ≥ 4):
`af082822` (28 → 1), `45dc21b6` (1, 13 → 2, 1), `gpt4_2ba83207`
(16, 3, 7, 1 → 4, 1, 5, 2), `3a704032` (20, 10 → 2, 5), `6f9b354f` (15 → 1).
Still partial: `gpt4_31ff4165` (3 of 6 in the top-10, was 2), `b46e15ed`
(3 of 5, was 0), `gpt4_d6585ce8` (3 of 5, unchanged), `0a995998` (1 of 3,
was 1). `75832dbd`'s single evidence round is outside the top-50 on both
sides — its facts did not bridge the question's wording either.

**The one loss:** `gpt4_4929293b` (temporal-reasoning, "a week ago"), rank
8 → 11, pushed out by other rounds' fact records. The k10 arm answered it
wrong with the round in view, so the gate pays one ANY@10 point and no
answer; it is now a retrieval miss in the misses join.

**Preview, not a score.** Joined over the k10 arm's verdicts: of its 18
wrong rows, 16 now have evidence in the top-10 (reading misses) and 2 are
retrieval misses (`gpt4_4929293b`, `75832dbd`). Seven of the nine moved rows
were wrong at k10; five of those are now complete, and the oracle answered
four of the five right (`3a704032` is wrong even for the oracle). Retrieval
moved where the hand-reading (PHASE2-RESULTS §0) said it would; whether the
reader converts it is gate 4-ii's and 4-iii's question.

**Extraction counters.** Rounds 24,747; sent to the model 24,676
(pre-filter skipped 71, 0.29 %); `[]` outputs 2,687 (10.9 %); truncated
outputs 87 (0.35 %, 83 of them unparseable and fact-less); truncated inputs
83 (0.34 %); facts per round sent: mean 2.40, p50 2, p90 5, max 8; facts
stored 54,202 (2.20 per round sent) beside 24,195 round records — 78,397
records in the hundred stores. Drops: round/exact 16, round/cosine 536,
fact/exact 379, fact/cosine 4,460 — the fact screens run at 0.95 unchanged
and collapse restatements within a session, never an evidence round.

**Time.** Wall 27.26 h (98,139 s), 3.98 s per round sent overall. The 87
uncontended questions ran at a median 3.24 s/round (13.3 min per question);
13 questions (9–10, 12–22) ran at 4.9–21.8 s/round while a game and a
browser shared the GPU, 5.0 h of excess. At the uncontended rate the pass
projects to 22.2 h — inside gate 2.1's 24 h as pre-registered; the extra
five hours belong to the machine's evening, not to the extractor.

**Decision: PASS → Task 8** (gate 4-ii, the n=20 dev prefix through the
Batch API, ≈ $0.25), then the four-arm sitting (gate 4-iii). The extraction
code stays opt-in (`--extractor qwen3`, `render_unit="turns"` default)
until 4-iii decides.

## Gate 4-ii read: the n=20 dev prefix on the extraction arm, 18/20 = the k10 arm's 18/20 — PASS (2026-09-14, 20:31)

`python -m evals --system mnimi --limit 20 --stage all --reader-transport openai
--batch --batch-poll-seconds 60 --extractor qwen3 --render-unit round+facts
--run-dir runs/mnimi__20q_gpt4o_extract` at `49198e9`, tree clean,
`provisional: []`, pins_hash `b71b173d…`; every round came from the cache
(no model call), two sub-batches (13 + 7 requests) under the 90k cap, 16.9 min
wall, $0.1726 actual against a $0.2619 projection (reader $0.1538, judge
$0.0188). Phase 2 spend so far: $0.17; programme total $10.93 of $50.

**Score 18/20, the k10 arm's score on the same 20 rows; the gate was within
3.** The verdicts are identical row for row — the same two wrong
(`gpt4_31ff4165`, multi-session, 3 of 6 evidence rounds in the top-10 now
against 2 before; `35a27287`, single-session-preference, wrong for the
oracle too) — so b = 0, c = 0 on the prefix: nothing broke and nothing moved,
which is what a smoke test at n=20 can say. **Fed tokens** mean 5,272
against the k10 arm's 4,639 (+14 %; max 6,660 vs 5,925; 0 truncated): the
facts block under each retrieved round costs about 630 tokens per context at
k=10. The five-arm question of whether the reader converts the retrieval
gain (gate 4-i: seven wrong-at-k10 rows moved, five to complete) is the
n=100 sitting's — none of those rows is in the prefix.

**Decision: PASS → Task 9**, the four-arm sitting at one clean commit
(p2base with `--verify-drift` against the published k10 arm, extract
`round+facts`, extract `facts`, `naive_rag`), pre-registered rules
unchanged: extraction adopted iff b ≥ c on `extract` vs `p2base`; `facts`
replaces `round+facts` iff b > c on that pair; the primary is the adopted
arm vs `naive_rag`, exact McNemar α 0.05.

## Gate 4-iii read: extraction adopted — `round+facts` 84 vs 80 (b=7, c=3); mnimi 84 vs naive_rag 79 (b=7, c=2) (2026-09-15, 00:32)

**The sitting.** Four arms at `82aa8b5`, tree clean on every launch, artifact
schema /8, `provisional: []` on all four, the same 100 stratified questions,
`gpt-4o-2024-08-06` through the Batch API in 7–8 sub-batches per arm, judged
with `longmemeval-paper-v3`, paired by `python -m evals.stats`. The extraction
arms replayed every round from the corpus pass's cache (no model call).

| arm | run dir | score | Wilson 95 % | mean fed tokens (max) |
| --- | --- | --- | --- | --- |
| baseline, `--extractor none --render-unit turns` | `mnimi__100q_gpt4o_p2base` | 80/100 | [71.1, 86.7] | 4,515 (6,792) |
| **extraction, `round+facts`** | `mnimi__100q_gpt4o_extract` | **84/100** | [75.6, 89.9] | 5,298 (7,732) |
| extraction, `facts` | `mnimi__100q_gpt4o_extract_facts` | 78/100 | [68.9, 85.0] | 1,701 (2,977) |
| `naive_rag` | `naive_rag__100q_gpt4o_p2` | 79/100 | [70.0, 85.8] | 4,534 (6,792) |

**The rules, applied (pre-registration of 2026-09-13):**

1. `round+facts` vs baseline: **b=7, c=3** (p=0.34) → b ≥ c → **extraction is
   adopted**: `MemoryConfig.render_unit` defaults to `"round+facts"` and the
   harness's mnimi arm extracts unless `--extractor none` is given
   (`evals.__main__.default_extractor`; `build_system` keeps `None` = no
   extractor for explicit callers and tests).
2. `facts` vs `round+facts`: **b=2, c=8** (p=0.11) → `round+facts` stays. The
   eight losses are all single-session rows (`95228167`, `fca762bc`,
   `0e5e2d1a`, `58bf7951`, `561fabcd`, `e9327a54`, `0a34ad58`, `1568498a`):
   facts alone drop the verbatim turns those questions are answered from,
   at a third of the tokens.
3. **Primary of the extraction era — the adopted arm vs `naive_rag`: b=7,
   c=2, exact McNemar p=0.18, not significant at 0.05.** mnimi 84 vs 79 is
   the first sitting in the programme where the primary leans mnimi's way
   (Phase 0: b=1, c=6; Phase 1: b=2, c=3). It is a working number at n=100;
   the verdict is Phase 5's n=500.

Descriptive, not paired for a decision: baseline vs `naive_rag` b=3, c=2
(80 vs 79 — the v1 write side is a wash against verbatim storage, as
Phase 1 left it); `facts` vs `naive_rag` b=9, c=10.

**Where the four points came from.** The seven rows `round+facts` wins over
the baseline: `57f827a0`, `852ce960`, `af082822`, `45dc21b6`, `2e6d26dc`,
`3a704032`, `6f9b354f` — five of them are rows gate 4-i moved into ALL@10
(`af082822`, `45dc21b6`, `2e6d26dc`, `3a704032`, `6f9b354f`), four of those
the pre-registered P2 rows. Against the k10 arm's 18 wrong rows: four are now
right (`af082822`, `45dc21b6`, `3a704032`, `6f9b354f`), `gpt4_2ba83207` stays
wrong with all its evidence in view, and 13 stay wrong. The three losses
(`09ba9854`, `75f70248`, `7405e8b1`) had no retrieval change at 4-i: reader
flips under a context 17 % longer. Per category (baseline / `round+facts` /
`facts` / `naive_rag`): knowledge-update 13/15/15/13, multi-session
10/10/11/8, single-session-assistant 17/17/12/17, single-session-preference
13/13/11/14, single-session-user 15/16/15/15, temporal-reasoning 12/13/14/12.

**The fed-token table** (mean `reader_prompt_tokens`, baseline / `round+facts`
/ `facts`): knowledge-update 4,941 / 5,656 / 1,742; multi-session 4,862 /
5,512 / 1,665; single-session-assistant 3,360 / 4,064 / 1,931;
single-session-preference 4,877 / 5,490 / 1,496; single-session-user 4,571 /
5,530 / 1,710; temporal-reasoning 4,526 / 5,567 / 1,664. The facts header
costs +783 tokens per context (+17 %); no context truncated in any arm.

**Drift, the fifth reading of the family.** The baseline arm against the
published k10 arm (identical configuration, the `/7 → /8` schema bump and
five commits between): **97/100 predictions changed at the byte level,
prompt tokens changed on 0/100 rows, score 82 → 80** — retrieval untouched by
Task 6, as the sitting's first check required, and the text not reproducible,
as this family never is (85, 61, 65, 61, 97 changed across five readings;
scores within 2). Fingerprints on the extraction arm: `fp_15cc60b404` 44,
`fp_f923e16c69` 56.

**Deviations, disclosed.**

- The launcher stopped after arm 1 and was restarted for arms 2–4 at the
  same commit: `--verify-drift` was refused — "pins differ on
  ['extractor_model']" — because the /7 artifact spells a missing extractor
  `None` and the /8 arm writes the literal `"none"`. No file changed between
  the two launches (`git status` clean at both; one sha in all four
  `pins.json`). The pre-registered check was read directly from the two
  prediction files (0/100 prompt-token changes) before arm 2 was submitted.
  After the sitting `evals.drift.pins_differences` learned the two spellings
  of an absent `extractor_*` pin (`ABSENT`, with a test); the fixed tool's
  report (`drift.json` beside the published baseline) agrees with the direct
  reading.
- The refusal returned before the run's accounting: the baseline arm's
  `reader_resolved` block was never written and its **$0.68 of reader spend
  never reached the ledger**. The spend was booked by hand the same night
  from the seven batch output files (451,528 prompt + 22,382 completion
  tokens, $0.6763 at the batch rate, 100/100 responses, 0 failed) and the
  harness now accounts before it verifies and books a filled entry on any
  refusal (`_predict_sync`, `_predict_batch`, `main`; a test).
- `render_unit="round+facts"` changes the JSON render *shape* even without
  facts (each round becomes an item carrying `facts: []` and its turns); the
  text shape is byte-identical without facts. A test pins both.

**Cost.** The sitting $2.81 (readers $0.6763 + $0.7746 + $0.3274 + $0.6787,
judges $0.3526); with gate 4-ii, Phase 2 spent **$2.98** against the $3.3
carried; programme $13.74 of $50.

## Phase 2 closes: the SPEC deviations in one place, and what the era leaves behind (2026-09-15, v1.9.0)

**Deviations from SPEC, each disclosed where it was decided and collected here:**

1. **Hybrid store, not facts-only** (D1). Every round keeps its v1 record and
   facts are second records on the same `round_key`; retrieval collapses to
   rounds. LongMemEval's key-expansion setting, chosen for a 1.7B extractor's
   miss rate; the facts-only *render* (`facts`) was the one pre-registered
   alternative and measured worse (78 vs 84).
2. **`when` + a deterministic resolver instead of a model-emitted
   `valid_time`.** The model quotes a time mention verbatim (dropped if the
   round does not contain it); `mnimi.extract.resolver` anchors it on `ts`.
   No LLM decides a date.
3. **GPU offload** (`n_gpu_layers=99`, CUDA 13.2) against the CPU-only pin:
   measured 50/50 byte-stable across a shuffled restart in fresh processes;
   the CPU build is the same code with different pins and its determinism
   is unmeasured.
4. **No question rule in the pre-filter.** A question can carry a fact
   ("should I take the 7 pm train to Utrecht again?"); the six rules keep
   every question, and the false-drop count on the slice's evidence rounds
   is 0.
5. **Over-long rounds are capped, not dropped**: a round past the 1,536-token
   input cap is cut at a token boundary and flagged (`truncated_input`,
   0.34 % of the slice); its round record is stored whole.
6. **Field meanings**: `content` keeps its embed-text meaning for both kinds
   (a fact's is `raw` + newline + `fact`) and `fact` carries what SPEC calls
   a fact's `content`; `MemoryRecord` gained `kind`, `fact`, `raw`, the
   triple, `valid_time`, `time_mention`.
7. **`negation_lexicon_hash = "none"`** in `memory_meta` until Phase 3 writes
   the lexicon; the row exists so the guard's shape is final.
8. **Harness fixes after the sitting** ("Gate 4-iii read"): the drift tool's
   two spellings of an absent extractor; accounting before verification.

**What the era measured.** Corpus pass 27.3 h (22.2 h at the uncontended
rate) into a 33.6 MB cache; extractor 2.40 facts per round, 10.9 % empty,
0.35 % truncated; gate 4-i ANY@10 93/95 (91), ALL@10 83/95 (77), 0 evidence
lost; gate 4-ii 18/20 = k10; gate 4-iii 80 / 84 / 78 / 79 — extraction
adopted, `round+facts`, primary b=7, c=2. Phase 2 spent $2.98 (programme
$13.74 of $50) and ≈ 25 hands-on hours against ≈ 45 planned.

**What it leaves for Phase 3.** The 16 remaining mnimi misses are reading
misses with the evidence in the top-10 (oracle also fails 6 of them); the
adopted arm loses three baseline rows to reader flips under a 17 % longer
context — the fed-token cost of the facts header is the first thing decay
and ranking (Phase 4) can trim. Phase 3 (conflict, the negation and
value-substitution screens) starts from a store that now holds triples and
valid times to conflict on. v1.9.0.

## Phase 3 pre-registration: conflict handling and the deterministic screens (2026-09-15)

Recorded before any Phase 3 code ran on the slice. Baseline for the phase: the
adopted v1.9.0 configuration and Phase 2's probe `runs/probe_mnimi_extract.json`
(commit `e1805e5`): ANY@10 93/95, ALL@10 83/95, 0 evidence rounds lost,
78,397 records stored, drops `round/exact 16`, `round/cosine 536`,
`fact/exact 379`, `fact/cosine 4,460`. The task-level plan is
`mnimi docs/PHASE3.md` (D1–D12 in full, Tasks 1–5); its numbers doc will be
`mnimi docs/PHASE3-RESULTS.md`. Phase 3 spends $0 of API.

**What was measured at design time, disclosed.** The extraction cache's
corpus-wide tables were read (23,302 rounds, 55,841 facts; subject, predicate
and object frequencies, marker counts inside fact texts, object lengths) and
the pair rules were simulated per store on the slice; no evidence round and
no question was read, no threshold was selected, nothing was scored. Three
facts from that reading shaped the design: 99.8 % of facts carry a triple;
the predicates are overwhelmingly speech-act verbs (`explained` 4,221,
`provided` 3,594, `mentioned` 2,248) and 37 % of objects are clauses of ten
tokens or more; and 5,705 `(subject, predicate)` groups per store hold two or
more objects with 33,963 facts in them (`user|has` alone 1,788: a cat *and* a
sister). A bare same-pair rule would therefore have superseded tens of
thousands of true facts.

**Design (PHASE3.md D1–D12), in brief.**
- D1 Fact records only conflict; a round is evidence, never superseded.
- D2 Candidates: (a) the value screen runs store-wide over every *active*
  earlier fact with the same normalized `pair_key`, through an index, because
  the session-scoped cosine gate never sees the cross-session
  knowledge-update case; (b) the negation screen runs over cosine-pass pairs
  within `dedup_scope` and over (a)'s candidates. A same-pair pair with
  different value-sized objects (≤ 6 tokens) is *routed away from the merge*;
  it is *superseded* only under three rules — negation (same object, opposite
  polarity), functional (a frozen predicate group: residence, employer,
  occupation, origin, vehicle, partner, name, weight, school, age, height,
  phone, email, birthday — positive, different values), numeric (both objects
  carry a number, the numbers differ, the residues match: "2 of Emma's
  recipes" → "3 of Emma's recipes"). The assistant's facts never conflict.
  The leaning (every same-pair different-object fact conflicts) is overturned
  on the 33,963-fact measurement above. Design-time projection on the slice:
  16 functional + 35 numeric + 0 negation supersessions across the 100
  stores (by category: knowledge-update 1/13/0, multi-session 3/8/0,
  single-session-user 3/8/0, single-session-assistant 4/1/0,
  single-session-preference 2/1/0, temporal-reasoning 2/4/0) — LongMemEval's
  history is built non-conflicting, so gate 3-i is a non-regression guard and
  gate 3-ii is the falsification.
- D3 Normalization is code, exact match after a fixed rewrite (lower-case,
  apostrophes and punctuation out, contractions expanded, first-person forms
  and "the user" → `user`, articles and auxiliaries stripped, markers removed
  and counted, antonym and functional forms canonicalized, number words →
  digits); no lemmatization, no fuzzy matching; inflections are enumerated.
  Versioned by a NEW `memory_meta` row `conflict_rules_hash` (not folded into
  `resolver_version` — different artifact, different lifetime). A v1.9 store
  is refused at open; every store is rebuilt from the cache.
- D4 The negation lexicon (`mnimi/conflict/lexicon.py`): 19 contractions, 20
  markers (`not`, `no`, `never`, `no longer`, `used to`, the cessation verbs as
  prefixes; `anymore` deliberately not a marker), ten antonym groups with
  every inflection (`like` folds love/enjoy with dislike/hate; `own` folds
  has/bought/adopted with sold/gave away/rehomed; pass/fail, win/lose, drop
  left out), fourteen functional groups (`uses`, `using`, `has` are NOT
  functional — measured). Polarity on the predicate and object tokens,
  `polarity_of_text(content)` as the fallback for a null triple.
  `negation_lexicon_hash = 330604b5772e…` replaces the literal `"none"`;
  `conflict_rules_hash = 7d19c48828c8…`. Both frozen at Block 1's commit and
  pinned by `tests/test_conflict.py`.
- D5 The entropy gate: token-level Shannon entropy in bits of the normalized
  fact text; `min(H(incoming), H(neighbour)) < dedup_entropy_gate` (SPEC's
  2.0) keeps a cosine-pass fact pair apart, after the two screens abstain;
  facts only. Token-level because a sentence's character entropy sits near
  4 bits whatever it says (Graphiti's character-level gate is on names).
- D6 One ordering: `(effective_time, session date, session ts, trust, id)`,
  greater wins; `effective_time` = `valid_time` when set, else the session
  date — SPEC step 4 for the pure cases, and the mixed dated-vs-standing case
  by the same key (a standing fact is an assertion current as of its session;
  "dated beats standing" is rejected because a 2015-dated fact would beat a
  2023 assertion). Loser `salience = 0`; winner `supersedes` = the last loser's
  id; `superseded {old_id}: {rule} {pair_key}: {old} -> {new}` at INFO on the
  `mnimi.memory` logger (the same channel as `filtered: {rule}`).
- D7 Runs inside `add()` per fact; `consolidate()` is the same decision as an
  idempotent full pass over the pair index (no-op after `add()`), still not
  called by the harness (Phase 4.3). A null-triple negation pair found through
  cosine is superseded by `add()` only.
- D8 Read path untouched: no salience-0 exclusion, no ranking, no decay. The
  probe can move only through keeps (fewer fact drops, never more, never a
  round).
- D9 Gate 3-i below. D10 Gate 3-ii below.
- D11 `dedup_entropy_gate = 2.0` lands (SPEC field); `conflict_resolution =
  True` lands as the one switch for the stage (`False` = the v1.9 write path,
  the gate 3-ii baseline and the "behind a flag" fallback); no threshold
  changes; no knob for the lexicon.
- D12 `memory_meta`: `negation_lexicon_hash` live, `conflict_rules_hash` new
  (sixteen rows; fifteen at v1.9). Pins schema /9 (`dedup_entropy_gate`, `conflict_resolution`,
  the two hashes; mnimi declares them, naive_rag does not;
  `HARNESS_PARITY_FIELDS` unchanged). Language: the probe is Tier 2 given the
  pins and the cache; the demo set is deterministic and CI-run.

**Gate 3-i (the slice guard, no API, PHASE3 Task 2).** After the screens land:
`python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100
--out runs/probe_mnimi_p3.json` from the cache, read against
`runs/probe_mnimi_extract.json`. PASS iff **ANY@10 ≥ 93/95, ALL@10 ≥ 83/95,
evidence lost = 0, `round/exact` = 16 and `round/cosine` = 536 exactly,
`fact/exact` = 379 exactly, `fact/cosine` ≤ 4,460, stored ≥ 78,397 and stored +
drops = 83,788**, AND the no-extractor probe `runs/probe_mnimi_p3base.json` is
identical to `runs/probe_mnimi_p2base.json` on ranks, top-50, drops and stored
for 100/100 questions. Reported beside it: keeps by screen, per-category
ANY/ALL; in Task 3, the probe re-run must be identical to Task 2's on 100/100
(D8), with conflicts and supersessions by rule and category against the
projection above, and prediction **P1**: the store of `45dc21b6` records a
numeric supersession whose winner's object carries `3`. Tripwires (reported,
not gated): `fact/cosine` < 3,345 → twenty kept pairs read by hand before
Task 3; superseded > 500 → the rules re-read before Task 4.

**Gate 3-ii (the demo set, the falsification gate, PHASE3 Task 4).**
`python -m evals.probes.conflict_demo --seed 0`: 100 seeded pairs — value-change
30 (15 functional, 15 count), negation 30 (15 marker, 15 antonym), dated-update
20 (10 in session order, 10 with the later-`valid_time` fact ingested first),
paraphrase 10 and unrelated 10 as controls — through a model-free
`ScriptedExtractor`, one `Memory` per pair, `HashingEmbedder`. A conflict pair
is correct iff exactly one active fact carries its pair key with the expected
value (or polarity), the loser has `salience = 0` and the winner's `supersedes`
points at it, after `add()` and again after `consolidate()`; paraphrase is
correct iff nothing is superseded and every active fact carries the expected
value; unrelated iff two facts stay active. Baseline: the same set with
`conflict_resolution=False` (the v1.9 write path). **PASS iff mnimi ≥ 72/80 on
the conflict families and ≥ 27/30, ≥ 27/30, ≥ 18/20 within them, controls
20/20, both readings, and strictly above the baseline on conflicts.**
Expected baseline: 0/80 on conflicts (two active facts fail the "exactly one
active" reading by construction; the brief's "near 50 %" guess is superseded
by this strict metric, which is what supersession exists to make true) and
20/20 on controls. An optional pass through the real extractor
(`--extractor qwen3`) is reported, never the gate.

**Adoption rule.** The screens and supersession ship as the library default
(`conflict_resolution=True`, harness following) iff gate 3-i AND gate 3-ii
pass. On a 3-i loss the code stays behind `conflict_resolution=False` as the
default and Phase 3 is re-scoped in PLAN.md (Tasks 3–4 still run,
informatively); on a 3-ii loss the same, with the failure ids recorded. No
number is tuned to pass either gate.

**Prohibited in this phase.** No LLM anywhere new; no threshold selection (0.95
and k=10 untouched; `dedup_entropy_gate` at SPEC's default); no edit to the
lexicon, the normalization tables or any `memory_meta` row under a run (an edit
is a version bump, a new hash, a re-ingest, a dated entry); no read-path change
(no salience-0 exclusion, no ranking, no decay); no change to `naive_rag`, the
renderer, `k`, the reader prompt, `EMBED_TEMPLATE` or `FACT_EMBED_TEMPLATE`;
`evals/systems/mnimi.py` keeps not calling `consolidate()`.

## Gate 3-i read: the screens keep 326 fact pairs apart and move no evidence round — PASS (2026-09-15)

**The instrument.** `python -m evals.probes.retrieval --system mnimi --extractor qwen3
--limit 100 --out runs/probe_mnimi_p3.json` from the worktree at the Task 2 tree (committed as
the Task 2 commit), every round replayed from the Phase 2 cache (`misses: 0` on 100/100
questions; the model was loaded, never called), 84 min wall; read with
`evals.probes.aggregate` and `evals.probes.identity` against `runs/probe_mnimi_extract.json`
(Phase 2, `e1805e5`). Beside it, the no-extractor probe `runs/probe_mnimi_p3base.json` against
Phase 2's `probe_mnimi_p2base.json`. Full tables: `mnimi docs/PHASE3-RESULTS.md` § Task 2.

**The gate, against the pre-registration:**

| | pre-registered | Phase 2 | Task 2 | |
| --- | --- | --- | --- | --- |
| ANY@10 / ALL@10 | ≥ 93/95, ≥ 83/95 | 93, 83 | **93, 83** | PASS |
| evidence rounds lost | 0 | 0 | **0** | PASS |
| round/exact, round/cosine | 16, 536 exactly | 16, 536 | **16, 536** | PASS |
| fact/cosine | ≤ 4,460 | 4,460 | **4,134** | PASS |
| stored, stored + drops | ≥ 78,397; 83,788 | 78,397 | **78,710; 83,788** | PASS |
| p3base = p2base | 100/100 | — | **100/100** | PASS |
| fact/exact | = 379 | 379 | **392** | pre-registration error, below |

Per-category ANY/ALL cells are Phase 2's to the unit; evidence ranks are identical on 99/100
questions (`gpt4_0b2f1d21`: one evidence round rank 3 → 4, still top-10). Keeps by screen:
negation 150, value 170, low-entropy 6 — 326, exactly Phase 2's fact/cosine minus Task 2's.
No tripwire (fact/cosine ≥ 3,345).

**One pre-registration error, corrected in the open.** "`fact/exact` = 379 exactly" assumed the
exact screen sees the same inputs; it does not — a kept fact joins the exact-key set, so 13
later restatements of kept facts moved from the cosine screen to the exact screen. Fact drops
total 4,526 vs 4,839, the 313 difference being the stored delta to the record. The invariant
meant (nothing dropped that v1.9 kept; records conserved) holds; the literal is wrong and
stays wrong in the pre-registration text above, with this entry as its reading.

**Decision: PASS → Task 3** (supersede). `MemoryConfig.conflict_resolution` stays `True` as the
library default pending gate 3-ii. What the screens cost the reader: 313 more fact records in
the hundred stores (+0.4 %), none of which displaced an evidence round from the top-10.

## Supersede lands: 46 facts superseded on the slice, no rank moved; P1 not held (2026-09-15)

**What landed (PHASE3 Task 3, PLAN 3.2).** `mnimi.conflict.supersede`: `conflict_between` (the
three rules of D2 — negation on one object, functional on a frozen predicate group, numeric on a
shared residue; the assistant's facts never conflict) and `beats` (D6's one ordering:
`valid_time` or the session date, then the session date, the raw session timestamp, the user's
span over the assistant's, the id). `Store.active_facts_by_pair` / `facts_with_pair_key` /
`supersede` (loser `salience = 0`, winner `supersedes` = the last loser's id). `Memory.add` runs
`_resolve_conflicts` after every stored fact against its active same-pair facts store-wide, plus
the cosine-kept negation neighbour when the pair has no key to meet on; `consolidate()` is the
same decision as an idempotent full pass over the pair index (tests: twice = once; after
`add()` = no-op; off means off). `evals/systems/mnimi.py` still does not call it. Every
supersession logs `superseded {old_id}: {rule} {pair_key}: {old} -> {new}` at INFO on
`mnimi.memory`. Tests 329 → 344.

**The probe (Step 8), at the Task 3 commit.** `runs/probe_mnimi_p3s.json` from the cache
(`misses: 0` on 100/100, 47 min) is **identical to Task 2's `probe_mnimi_p3.json` on 100/100
questions** over evidence ranks, top-50 sessions, drops and stored — D8 as pre-registered:
supersession stores the loser and moves no rank. Conflicts found: functional 16, numeric 30,
negation 0; **46 facts superseded** (projection 16 / 35 / 0), by category knowledge-update 14,
multi-session 7, single-session-user 11, single-session-assistant 5, single-session-preference 4,
temporal-reasoning 5. Tripwire (> 500) not tripped. Full list, read by hand, in
`mnimi docs/PHASE3-RESULTS.md` § Task 3: about half the numeric supersessions are enumerations
(`answered: question 249 -> 250 -> …`) and nine of the functional ones are containment
refinements (`tokyo -> roppongi`) — the exact-match rules' known false-positive classes,
invisible in this phase and Phase 4's to price.

**P1 not held.** No supersession on `45dc21b6`: the earlier round's extraction carries no
"tried two recipes" triple (the model emitted the recipes as something the user has "been loving",
no count), so the later `tried out | 3 of Emma's recipes` fact has nothing to conflict with. An
extraction limit; recorded as the failed prediction it is.

**Two corrections during the task, disclosed.** The cosine-negation path was first applied to
every kept neighbour rather than only to null-triple pairs (D7) — narrowed on the first run's
question 1; and it let one keyed assistant fact through (`assistant|read: yes -> no`) because the
null-triple side hid the subject — D2 says the assistant never conflicts, so the subject is now
read from either side; the probe was re-run at the committed code and the two runs' supersession
lines differ by exactly that line (`runs/probe_mnimi_p3s_run1.*` kept). → Task 4, gate 3-ii.

## Gate 3-ii read: the demo set — mnimi 80/80 conflicts + 20/20 controls vs exact-dedup 0/80 + 20/20 — PASS (2026-09-16)

**The instrument (PHASE3 Task 4, PLAN 3.3).** `evals/probes/conflict_demo.py`: `generate(seed=0)`
builds the 100 pre-registered pairs from frozen template tables through `random.Random(0)` —
value-change 30 (15 `functional`: residence 3, employer 3, occupation 3, vehicle 2, partner 2,
school 2; 15 `count`: countables 7, "tried n of X's recipes" 4, "about n followers" 4), negation
30 (15 `marker`: `no longer` ×6, `can't … anymore` 3, `not … anymore` 3, `never` 3; 15
`antonym`: love/hate, started/stopped, joined/left, bought/sold, trust/distrust, 3 each),
dated-update 20 (10 `in-order`, 10 `reversed` — the later-`valid_time` fact ingested first from
the earlier session, "I moved to Chicago on March 3rd" in February against "Back in 2016 I lived
in Portland" in September), paraphrase 10 and unrelated 10 as controls. Each pair is two rounds
(a user turn, a short assistant turn) in LongMemEval-shaped sessions — an early one (Jan–Mar
2023) and a late one (Jun–Sep 2023), the same session for the paraphrase controls — with one
pre-authored `ExtractedFact` per round handed back by `ScriptedExtractor`, so the write path
(pre-filter, resolver, both screens, the pair index, supersession, `consolidate()`) runs end to
end with no model. One fresh `Memory` per pair, `MemoryConfig()` defaults with only
`conflict_resolution` toggled, `HashingEmbedder`. Every user turn is unique, so one script serves
the set; `generate(0) == generate(0)` and the family sizes are asserted by
`tests/test_conflict_demo.py`, whose gate test IS the pre-registered criterion and prints the
table (suite 344 → 348). Command: `PYTHONPATH=src python -m evals.probes.conflict_demo --seed 0
--out runs/conflict_demo.json` from the worktree at the Task 4 tree (committed as the Task 4
commit, parent `20a0f4f`); exit 0 is the gate, the JSON the artifact.

**The gate, against the pre-registration (D10):**

| family | pairs | mnimi after `add()` / after `consolidate()` | exact-dedup (v1.9 path) | pre-registered | |
| --- | --- | --- | --- | --- | --- |
| value-change | 30 | **30 / 30** | 0 / 0 | ≥ 27 | PASS |
| negation | 30 | **30 / 30** | 0 / 0 | ≥ 27 | PASS |
| dated-update | 20 | **20 / 20** | 0 / 0 | ≥ 18 | PASS |
| = conflicts | 80 | **80 / 80** | 0 / 0 | ≥ 72, strictly above baseline | PASS |
| paraphrase | 10 | 10 / 10 | 10 / 10 | 10 | PASS |
| unrelated | 10 | 10 / 10 | 10 / 10 | 10 | PASS |
| = controls | 20 | **20 / 20** | 20 / 20 | 20 | PASS |

Failure ids: none. By rule, the mnimi run found and settled exactly the conflicts the set was
built to hold — functional 35 (15 value-change + 20 dated), numeric 15, negation 30 — 80
supersessions, every winner's `supersedes` pointing at its loser, 0 on the controls (paraphrase
and unrelated pairs are candidates on the index — 40 candidate reads — and `conflict_between`
abstains on every one). The baseline stores both facts of every conflict pair active
(`2 active facts on the pair key, expected exactly one` on all 80), as pre-registered: the
"exactly one active" reading is what supersession exists to make true, and the v1.9 path never
touches `salience`. `consolidate()` after `add()` changed nothing on any pair (idempotent, D7).

**What the hashing pass did not exercise, disclosed.** Under `HashingEmbedder` no fact pair
reached the 0.95 cosine gate — `pairs_screened = 0` on all 100 pairs — so the screens of Task 2
(negation, value, entropy) were not on the path here: every conflict went through the pair
index (D2a) and every control was kept apart by the gate, not by a screen. The same set under
the real embedder (`--embedder bge`, `runs/conflict_demo_bge.json`; descriptive, the `[embed]`
extra): the table is identical cell for cell (80/80 + 20/20 vs 0/80 + 20/20), and 8 of the 10
same-session paraphrase pairs did trip the gate — `pairs_screened 8`, all 8 read
`merge/duplicate` by `screen_pair` (same pair key, same object, same polarity), the other 2 sat
below 0.95 and stayed as two active facts. Both are the "correct" reading for a paraphrase. The
cross-session conflict pairs never reach the gate under either embedder (session scope, R3),
which is exactly the case D2's index exists for.

**The descriptive pass through the real extractor (never the gate).** `--embedder bge
--extractor qwen3 --out runs/conflict_demo_qwen3.json` (CUDA on PATH, `ggml_cuda_init: found 1
CUDA devices`; the demo's own cache `.cache/extract/conflict_demo_e15153f8….sqlite`, never the
corpus file — 200 misses on the mnimi run, 200 hits on the baseline run; 0 truncated outputs, 0
pre-filter skips). The same 200 natural-language rounds, the model's own triples instead of the
authored ones, the same reading against the authored pair key and value. Per family
(conflict-able = both rounds yield a fact on one non-null pair key; correct = D10's reading):

| family / subfamily | pairs | conflict-able | on the authored key | correct (add = consolidate) |
| --- | --- | --- | --- | --- |
| value-change / functional | 15 | 12 | 11 | 11 |
| value-change / count | 15 | 10 | 10 | 10 |
| negation / marker | 15 | 8 | 7 | 7 |
| negation / antonym | 15 | 2 | 2 | 2 |
| dated-update / in-order | 10 | 4 | 4 | 4 |
| dated-update / reversed | 10 | 8 | 8 | 8 |
| **= conflicts** | **80** | **44** | **42** | **42** |
| paraphrase | 10 | 2 | 2 | 8 |
| unrelated | 10 | 1 | 1 | 1 |

Read: **every pair the model made conflict-able on the authored key resolved correctly (42/42)**,
and the two conflict-able pairs it did not are the rules abstaining as designed — both facts of
`value-change/functional/10` came out as `user | owns | Honda Civic` / `Kia Soul` (`own` is not a
functional group, so two cars are two facts), and both of `negation/marker/05` came out with the
assistant as subject (`assistant | noted | the user can eat spicy food`), which D2 says never
conflicts. **0 supersessions on the 20 controls.** The other 38 conflict misses and the 11
control misses are all upstream of the screens: the 1.7B extractor returned `[]` on 57 of the
200 rounds (28.5 % — a bare one-sentence round with a one-line acknowledgement, against 10.9 %
on the corpus), put 4 more into the assistant's mouth, and keyed a few elsewhere (`user | says |
can eat red meat`; "I drive a Nissan Leaf" → `owns`, "I'm dating Priya" → no triple). The
antonym family is where it hurts most (2/15: the model seldom emits both sides of love/hate,
started/stopped as triples on one key), the numeric rule where it hurts least (10/15). The
dated pairs the model did key resolved on `valid_time` in both orders (reversed 8/8, in-order
4/4 of the keyed ones: `moved to | Chicago | 2022-03-03` beat `lived in | Portland | 2016` from a
later session). This is the honest ceiling of screens + extractor together on this set, and it
is an extraction ceiling: PHASE3-RESULTS § Task 4 keeps the model's triples per pair. It is not
the gate, was never pre-registered as one, and changes no decision.

**Decision: PASS.** Both gates of the pre-registration have passed (3-i on 2026-09-15, 3-ii
here); by the adoption rule the screens and supersession ship as the library default
(`MemoryConfig.conflict_resolution = True`, the harness following) — nothing changes in this
commit, the default has stood since Task 2 pending this read. No lexicon, normalization or
threshold was touched between the pre-registration and this read. What the demo does and does
not claim: it is deterministic and CI-run (seeded, asserted byte-identical per seed); it
falsifies the mechanism on authored triples, not the extractor's — the descriptive pass above
is the honest ceiling of the two together — and the read path still renders a superseded fact
(D8), so nothing here moves a benchmark number until Phase 4 reads `salience`. → Task 5, close
Phase 3 at v1.10.0.

## Phase 3 closes: the SPEC deviations in one place, and what the phase leaves for Phase 4 (2026-09-16, v1.10.0)

**Deviations from SPEC, each disclosed where it was decided and collected here** (PHASE3 D1–D12;
"Phase 3 pre-registration", "Gate 3-i read", "Supersede lands", "Gate 3-ii read"):

1. **Supersession has a predicate condition** (D2, CHANGELOG #17). SPEC step 4's "a new fact
   contradicting an existing one" is narrowed to three rules — `negation` (one object, opposite
   polarity), `functional` (a frozen predicate group, two positive values), `numeric` (different
   numbers on one residue) — because the literal same-pair-different-object rule would have zeroed
   33,963 true facts on the cache (5,705 multi-valued groups; `user|has` 1,788). A same-pair pair
   with two value-sized objects is still routed away from the merge; it is superseded only under a
   rule. The assistant's facts never conflict.
2. **One effective time, not a class priority** (D6). `valid_time` when set, else the session
   date; then the session date, the raw `ts`, the user's span over the assistant's, the id. SPEC's
   three cases fall out of it and the mixed dated-vs-standing case is decided by the same key (a
   2015-dated fact does not beat a 2023 assertion).
3. **`supersedes` is one integer pointing at the *last* loser** the winner defeated (SPEC's shape;
   a chain of updates leaves each intermediate loser at salience 0 with its own pointer).
4. **`consolidate()` reads the pair index only** (D7). A cosine-pass negation pair whose triples
   are null has no key to meet on; `add()` supersedes it, the pass cannot re-derive it (no vectors
   in the pass). Idempotent by construction; a no-op after `add()`; still not called by
   `evals/systems/mnimi.py` — Phase 4.3 wires it as the explicit decision.
5. **`conflict_resolution` is a config field SPEC did not foresee** (D11): one switch for the
   screens, the entropy gate and supersession; `False` is the v1.9 write path byte for byte (a
   test compares stored contents). `dedup_entropy_gate` lands at SPEC's 2.0, untuned, token-level
   (D5 — a sentence's character entropy sits near 4 bits whatever it says).
6. **Candidates come from the store-wide `pair_key` index, not from cosine-gate-pass pairs**
   (D2a). SPEC scopes screens 3–4 to cosine-pass pairs; since R3 the gate is session-scoped and
   the knowledge-update case ("Boston" in session 3, "Seattle" in session 9) never reaches it.
   The negation screen keeps SPEC's scope for the in-session half and adds the index half.
7. **Two guard rows, sixteen in all** (D12): `negation_lexicon_hash` live (`330604b5772e…`),
   `conflict_rules_hash` new (`7d19c48828c8…`, its own lifetime — not folded into
   `resolver_version`). The "thirteen rows" quoted at v1.9 was a miscount of fifteen. Pins schema
   /9 (`dedup_entropy_gate`, `conflict_resolution`, the two hashes; mnimi declares them, naive_rag
   does not). A v1.9 store is refused at open.
8. **The read path is untouched** (D8): no salience-0 exclusion, no ranking, no decay. A
   superseded fact still ranks and renders. This is what made the Task 3 identity check exact
   (100/100) and is why no benchmark number moved in this phase and none is claimed.
9. **One pre-registration error, corrected in the open**: `fact/exact = 379 exactly` read 392 —
   a kept fact joins the exact-key set, so 13 later restatements moved from the cosine screen to
   the exact screen; records conserved at 83,788.
10. **Two corrections during Task 3, disclosed**: the cosine-negation path narrowed to
    null-triple pairs (D7 as written), and the assistant exclusion applied through that path
    (one `assistant|read: yes -> no` line in the first run, gone in the committed run).
11. **P1 not held**: no numeric supersession on `45dc21b6` — the earlier round's extraction has
    no count triple. An extraction limit, recorded as the failed prediction it is.

**What the phase measured** (all at $0; `mnimi docs/PHASE3-RESULTS.md`). Gate 3-i: ANY@10 93/95,
ALL@10 83/95, 0 evidence lost, round drops 16/536 identical, fact/cosine 4,460 → 4,134 (326 keeps:
negation 150, value 170, low-entropy 6), stored 78,397 → 78,710, ranks identical 99/100, the
no-extractor probe identical to Phase 2's on 100/100. Supersede: the probe identical to Task 2's
on 100/100; 46 facts superseded on the slice (functional 16, numeric 30, negation 0; projection
16 / 35 / 0), about half the numeric ones enumerations and nine functional ones containment
refinements. Gate 3-ii: mnimi 80/80 conflicts (30/30, 30/30, 20/20) + 20/20 controls after `add()`
and after `consolidate()` vs the v1.9 path's 0/80 + 20/20; BGE identical; the real-extractor pass
42/42 of the pairs the model keyed on the authored pair, 0 control supersessions, 57/200 rounds
`[]`. Tests 272 → 348, ruff clean, CI reproduced on 3.11 and 3.12 with the `[dev]` extra alone.
Hands-on ≈ 19.5 h against 25 planned; ≈ 3.3 h of unattended probe time; closed 58 days ahead of the
Nov 13 line.

**What it leaves for Phase 4.** (a) The salience-0 exclusion has a price to read before it ships:
the 46 slice supersessions include the exact-match rules' two false-positive classes —
`tokyo -> roppongi` containment refinements on `lives in` (nine) and `answered: question 249 ->
250 -> …` enumerations on a numeric residue (fifteen) — invisible while the read path ignores
`salience`, and the first thing an active-record filter would hide from the reader. (b) Ranking
(`salience_weights`, `ScoredRecord`), decay with its floor and `last_accessed` are unbuilt; the
fed-token cost of the facts header (+783 tokens per context, Phase 2) is still the lever they can
trim. (c) `consolidate()` is wired into the harness only by an explicit edit (4.3). (d) An
extraction limit the demo exposed, outside this phase's scope: the 1.7B extractor returns `[]` on
28.5 % of one-sentence rounds (10.9 % on the corpus) and seldom emits both sides of an antonym
pair as triples — the screens can only resolve what the model keys. (e) The lexicon and the
normalization tables stay frozen; any edit is a version bump and a re-ingest. v1.10.0.

**Commit hashes after the merge (2026-09-16).** `main` had gained `fe70e76` (a FUTURE.md
section) after the branch was cut, so `feature/conflict` was rebased onto it and `main`
fast-forwarded to `0a165d9`, both pushed. The rebase rewrote the six Phase 3 hashes. Each pair
has the same subject and the same patch-id, and each pair's trees differ only by
`docs/FUTURE.md`: `e936ff8` → `9a324c7`, `bba655e` → `d2f905e`, `6918e60` → `c9863d7`, `3ffb959` → `20a0f4f`, `2f48d59` → `0de8971`, `8fbb018` → `0a165d9`. The probe and demo runs of "Gate 3-i read", "Supersede
lands" and "Gate 3-ii read" ran at the old hashes, on the same code; the entries in this file and
the private docs now cite the new ones.

## Phase 4 pre-registration: decay, ranking and the read side (2026-09-16)

Recorded before any Phase 4 code ran on the slice. Baseline for the phase: v1.10.0 on `main`
(`0a165d9`; `c5d46c9` records the Phase 3 hash map) — the read path does not read `salience`.
The gpt-4o family's adopted number is Phase 2's `mnimi__100q_gpt4o_extract` (84/100, `82aa8b5`).
Probe references: Phase 3's `runs/probe_mnimi_p3s.json` (the supersede probe, `20a0f4f`: ANY@10
93/95, ALL@10 83/95, 0 evidence lost, 78,710 stored, 46 facts superseded) and
`runs/probe_mnimi_p3base.json` (the no-extractor probe, `c9863d7`). The task-level plan is
`mnimi docs/PHASE4.md` (D1–D12, Tasks 1–10); its numbers doc is `mnimi docs/PHASE4-RESULTS.md`.
Phase 4 spends ≈ $2.6 of API.

**What was measured at design time, disclosed.** Two readings, neither of which ranked, scored or
read an evidence round, a question or an answer. (1) The slice's session dates (dataset
timestamps only): 0 unparseable; a question's history spans 10 days at the median, 53 at p90,
155 at most; the question date falls 0 days after the latest session at the median, 30 at most;
at SPEC's 30-day half-life a round's decay factor is 0.891 at the median and 0.616 at p90, and
1.6 % of rounds reach the 0.15 floor. (2) The extraction cache's `salience` values (55,841
facts): user facts 30,730 of 30,750 at 1.0; assistant facts 21,734 of 24,530 at 0.5 and 918 at
0.25 — the extractor's salience is a speaker flag in effect, so SPEC's salience multiplier halves
most of the assistant's facts before any decay. A throwaway reference implementation in a scratch
copy of `c5d46c9`, outside the repository, ran the plan's tests (391 passing, ruff clean) so the
plan's test code is checked; it is not committed and is not the implementation.

**Design (PHASE4.md D1–D12), in brief.**
- D1 `now_logical` = the user's `created_at` with the greatest `created_at_key` (ties: the
  greater string), `None` when none is dated; `logical_days` = whole days between anchor dates,
  0 when undated or negative. Never the wall clock, never the question date.
- D2 `last_accessed`: a column, `created_at` at insert, `now_logical` on every record `recall()`
  returns (one UPDATE after the relevance drop). Under the eval protocol it cannot move a
  number; shipped because SPEC lists it and decay reads it.
- D3 Decay in `consolidate()`, after the conflict pass, persisted, rounds and facts:
  `salience = max(min(decay_floor, initial_salience), initial_salience *
  0.5 ** (days(now_logical, last_accessed) / decay_half_life_days))` for every record with
  salience > 0. `initial_salience` is a new column (the inserted value, never updated), so a pass
  is a pure function of stored fields (twice = once) and an access restores. Salience 0 is never
  decayed. `decayed {id}: {days} days since last access, salience {old} -> {new}` at INFO on
  `mnimi.memory`. No decay arithmetic on the read path.
- D4 `ranking`: `"similarity"` = `Store.search_rounds` untouched (the landing default);
  `"score"` = SPEC's `(w_sim·relevance + w_rec·recency)·salience` with `recency =
  0.5 ** (days(now_logical, created_at) / decay_half_life_days)`, a round scored by its best
  record, the exact top-k over rounds by score (`rank_rounds`: batches of k, 4k, … until
  `max(w_sim·c_last + w_rec, 0) < score_k`). SPEC's "top-k by vector, then attach scores" is read
  as selection by score: read literally, decay could never change what reaches the reader.
  Invariant: default weights and a uniform salience of 1.0 give `search_rounds`' result exactly.
- D5 `active_only` (landing default `False`): the KNN and the renderer's `facts:` header skip
  salience-0 records; rounds always stay. Priced by gate 4-ii before it ships.
- D6 `ScoredRecord(record, relevance, recency, salience, score)`; `recall() ->
  list[ScoredRecord]`; `get_context` unwraps (rendered bytes unchanged); the probe reads
  `Memory._rank`, the same function without the write-back.
- D7 `recall_min_relevance = 0.0`, applied after the top-k only when > 0 (0.0 is off by
  definition); never set in a run.
- D8 `MnimiSystem(consolidate=)`: one `consolidate()` per store after the last session, before
  the question; flag `--consolidate`, off until the sitting, pinned. The day the benchmark can
  change.
- D9 Pins schema /10 (`active_only`, `ranking`, `salience_weights`, `recall_min_relevance`,
  `decay_half_life_days`, `decay_floor`, `consolidate`, `decay_rules_hash`; mnimi only;
  `HARNESS_PARITY_FIELDS` unchanged). `memory_meta` gains a seventeenth row, `decay_rules_hash =
  d4a0bcf07330…`, over the frozen `mnimi.decay.DECAY_RULES` (pinned by a test); a v1.10 store is
  refused at open.
- D10, D11 below. D12: v1.11.0; the probe is Tier 2 given the pins and the cache; the sitting is
  "score reproducible within 6/100 flips; text not reproducible".

**Gate 4-i (identity under defaults, no API, PHASE4 Task 7).** From the worktree at the Task 7
commit, the cache replaying with `misses: 0`: (a) `python -m evals.probes.retrieval --system
mnimi --extractor qwen3 --limit 100 --out runs/probe_mnimi_p4i.json` is identical to
`runs/probe_mnimi_p3s.json` on evidence ranks, top-50 sessions, drops and stored for **100/100**
questions, with 46 supersessions; (b) the same with `--extractor none --ranking score
--active-only on --out runs/probe_mnimi_p4ibase.json` is identical to
`runs/probe_mnimi_p3base.json` on **100/100**. A difference is a bug in Tasks 2–7, fixed before
gate 4-ii and disclosed; this entry is not amended.

**Gate 4-ii (the exclusion priced, no API, Task 8).** `--extractor qwen3 --active-only on
--ranking similarity --out runs/probe_mnimi_p4x.json` against `p4i`: PASS iff **ANY@10 ≥ 93/95,
ALL@10 ≥ 83/95, and 0 evidence rounds leave the top-10**; validity: `misses: 0`, 46
supersessions, 0 stale facts under the top-10 rounds. Reported beside it: the 46 supersessions of
`runs/probe_mnimi_p3s.log` classified by hand — refinement, enumeration, coexisting (the three
false-positive classes) or update, the false positives reported as a range — expected ≥ 29
lines (Phase 3 named 13 refinement and 16 enumeration lines); per question, superseded
facts against stale facts under its top-10 rounds at `p4i`; the top-10 evidence rounds carried
by a superseded fact at `p4i`. Reported, not gated: `p4s` (`--ranking score`, `active_only` per
the gate) against `p4x`, and `p4d` (plus `--consolidate on`) against `p4s`. Predictions: **P1**
`p4x` identical to `p4i` on ≥ 95/100; **P2** `p4s` identical to `p4x` on ≤ 50/100; **P3** `p4d`
identical to `p4s` on ≤ 50/100.

**Gate 4-iii (the sitting: n=100, gpt-4o family, one clean commit — Task 8's; Task 9).** Three
mnimi arms through the Batch API with `--extractor qwen3 --render-unit round+facts`, one arm in
flight at a time: **A** `--active-only off --ranking similarity --consolidate off`
(`--verify-drift results/published/mnimi__100q_gpt4o_extract`, descriptive: Phase 3's kept facts
may change some prompts); **B** `--active-only on --ranking score --consolidate off`
(`--active-only off` if gate 4-ii failed); **C** B's flags with `--consolidate on`. Rules (b = the
second arm's wins in `python -m evals.stats <first> <second>`):
- `MemoryConfig.ranking` defaults to `"score"` iff b ≥ c on A → B.
- `evals.knobs.MNIMI_DEFAULT_CONSOLIDATE` becomes `True` iff `"score"` is adopted AND b ≥ c on
  B → C.
- `MemoryConfig.active_only` defaults to `True` iff gate 4-ii passed.
- SPEC's decay-on/off ablation is B → C: X = score(C) − score(B), reported with b, c, p and
  whether it lies inside the family's 6/100-flip drift, whatever its sign. A → C is descriptive.
- A loss leaves the v1.10 default; code, flags and tests stay. The mnimi vs naive_rag primary is
  not re-paired (Phase 5's n=500 reads it).

**Budget.** PLAN carried $1 for one arm. The sitting is three arms at ≈ $0.87 each with judges
(the `round+facts` arm: reader $0.7746, judge ≈ $0.09) — **≈ $2.6**, ≈ $4.0 projected upper
bound — against $36.26 remaining; Phase 5 keeps its $14.25.

**Prohibited in this phase.** No LLM anywhere new; no clock; no value of `decay_half_life_days`,
`decay_floor`, `salience_weights` or `recall_min_relevance` other than SPEC's in any run, and no
selection among values; no change to 0.95, `k=10`, `EMBED_TEMPLATE`, `FACT_EMBED_TEMPLATE`, the
extractor, the lexicon, the normalization tables, the reader prompt, the renderer's templates or
`naive_rag`; no edit to `DECAY_RULES` after Task 2's commit; the extractor's salience is priced,
not overridden.

## Gate 4-i read: identity under defaults — 100/100 and 100/100 — PASS (2026-09-17)

**The instrument.** Two retrieval probes from the worktree at the Task 7 tree (committed as the
Task 7 commit; the probes ran on the tree that became it, before this entry was written), read
with `evals.probes.identity` and `evals.probes.aggregate`. (a) `python -m evals.probes.retrieval
--system mnimi --extractor qwen3 --limit 100 --out runs/probe_mnimi_p4i.json` — every Phase 4
flag at its default, so `ranking="similarity"`, `active_only=False`, `--consolidate off` — 81 min,
every round replayed from the Phase 2 cache (`'misses': 0` on 100/100 questions, 49,352 hits; the
model was loaded on the GPU and never called; the cache holds 23,302 rows before and after).
(b) the same with `--extractor none --ranking score --active-only on --out
runs/probe_mnimi_p4ibase.json` — 55 min, concurrent with (a). Full tables, the preflight output
and both aggregates verbatim: `mnimi docs/PHASE4-RESULTS.md` § Task 7.

**The gate, against the pre-registration:**

| | pre-registered | reading | |
| --- | --- | --- | --- |
| (a) `p4i` = `probe_mnimi_p3s.json` on evidence ranks, top-50 sessions, drops, stored | 100/100 | **100/100** | PASS |
| (a) facts superseded | 46 | **46** (functional 16, numeric 30, negation 0) | PASS |
| (a) cache misses | 0 | **0 on 100/100** | PASS |
| (a) ANY@10 / ALL@10 | 93/95, 83/95 | **93/95, 83/95** | PASS |
| (b) `p4ibase` = `probe_mnimi_p3base.json` | 100/100 | **100/100** | PASS |

Stored, drops, keeps and the extraction counters are Phase 3's to the unit (78,710 stored; 16 /
536 / 392 / 4,134; keeps 150 / 170 / 6; facts stored 54,515). Clause (b) is D4's invariant on real
BGE stores rather than on hashing fixtures: default weights and a uniform salience of 1.0 make
`rank_rounds` return `Store.search_rounds`' rounds in its order, record for record, with
`active_only` on for good measure — no row moved.

**The numbers Task 8 inherits** (the `p4i` aggregate's `read path:` line, the v1.10 read path):
**5** salience-0 facts sit under the top-10 rounds and would be rendered, **1** top-10 evidence
round is carried by a superseded record — `10e09553` evidence #0, the largemouth-bass question
whose answer *is* the superseded value ("7 largemouth bass", superseded by the 7/22 trip's "9") —
and **3** are carried by a down-weighted record (the extractor's 0.5 salience on an assistant
fact). Records decayed: 0, `--consolidate` being off. Gate 4-ii prices exactly this.

**Decision: PASS → Task 8.** Nothing in Tasks 2–7 moved a retrieval row, so the Phase 4 defaults
are the v1.10 read path on the slice as pre-registered; the landing defaults stay
(`ranking="similarity"`, `active_only=False`, `MNIMI_DEFAULT_CONSOLIDATE=False`) until gates 4-ii
and 4-iii decide them.

## Gate 4-ii read: the exclusion priced — ANY@10 93/95, ALL@10 83/95, 0 evidence rounds left — PASS (2026-09-17)

**The instrument.** `python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100
--active-only on --ranking similarity --out runs/probe_mnimi_p4x.json` (58 min, cache replayed with
`'misses': 0` on 100/100, 23,302 rows before and after), read with `evals.probes.aggregate` and
`evals.probes.identity` against gate 4-i's `runs/probe_mnimi_p4i.json`. Beside it, reported and not
gated: `p4s` (`--ranking score`, 46 min) and `p4d` (the same plus `--consolidate on`, 46 min). All
three at HEAD = the Task 7 commit `77f4b97` plus the fix `0ccdaa9` below; $0. Full tables, the
classified 46 and all three aggregates verbatim: `mnimi docs/PHASE4-RESULTS.md` § Task 8.

**The gate, against the pre-registration:**

| | pre-registered | `p4i` | `p4x` | |
| --- | --- | --- | --- | --- |
| ANY@10 | ≥ 93/95 | 93/95 | **93/95** | PASS |
| ALL@10 | ≥ 83/95 | 83/95 | **83/95** | PASS |
| evidence rounds leaving the top-10 | 0 | — | **0** | PASS |
| validity: misses / supersessions / stale facts under the top-10 | 0 / 46 / — | 0 / 46 / 5 | **0 / 46 / 0** | ok |

The exclusion's whole cost on the slice is one row: `10e09553`'s first evidence round falls from
rank 1 to rank 6 — still retrieved, now carried by the round's own record instead of the superseded
"7 largemouth bass" fact that the question actually asks for. Four other rows differ only outside
the top-10 (first divergence at ranks 13, 34, 35, 38). Five stale facts stop being rendered.

**The 46 supersessions, classified by hand under D5's rubric** (each reason quotes its log line;
the source turn decides where the line cannot): refinement **11**, enumeration **17**, coexisting
**13**, coexisting? **2**, update **3**. False positives as a range: **low 28** (refinement +
enumeration), **high 43** (plus coexisting and the undecided rows). The pre-registration expected
≥ 29 lines: **held on the high reading, not on the low one** — two `midtown -> downtown san
francisco` rows leave refinement (the source shows a stay against a base) and `segment 3 ->
segment 4` joins enumeration (Phase 4's own rubric example; Phase 3 filed it as an update). Only
three of 46 lines are the "value changed" case the rule is for. Two extraction failure modes
account for much of the rest: dropped hyphens in ranges (`6-7 hours` → `67`) and third-party
numbers attributed to the user (two personas' ages and incomes, a word problem's minutes).

**The two reported probes.** `p4s` (SPEC's score, no decay) reorders most top-50 lists — 33/100
identical — but moves no evidence round across the top-10 boundary and leaves ANY@10 / ALL@10 at
93 / 83; down-weighted carriers fall 3 → 0, because a round's own record outranks its 0.5-salience
facts. `p4d` (score + `--consolidate on`, 60,545 records decayed) is the first configuration in
this phase to move recall: **ANY@10 93 → 89, ALL@10 83 → 75**, 13 evidence rounds leaving the
top-10 and 2 entering, the loss concentrated in knowledge-update (ALL@10 14 → 8) while
temporal-reasoning gains a row. Decay at SPEC's values is therefore expected to cost the sitting's
arm C, not to help it — recorded here before the arms run.

**Predictions:** P1 held (`p4x` = `p4i` on 95/100, bar ≥ 95), P2 held (`p4s` = `p4x` on 33/100,
bar ≤ 50), P3 held (`p4d` = `p4s` on 13/100, bar ≤ 50).

**One crash and its fix, disclosed.** The first `p4d` run raised `sqlite3.OperationalError: k value
in knn query too large, provided 6400 and the limit is 4096` on question 1. `Store._knn` requested
`max(k * 8, k)` rows without clamping to sqlite-vec's 4,096 limit, so any caller wanting more than
512 rows raised; `rank_rounds`' exact-top-k loop reaches 800 once decayed saliences lower the k-th
score (the bound assumes salience 1.0). Latent since v1 — `search_rounds` grows the same way.
Fixed at the source in `0ccdaa9` (`min(max(k * 8, k), MAX_KNN_ROWS)`, `MAX_KNN_ROWS = 4096`) with a
regression test; tests 391 → 392. The clamp binds only above 512 rows and every earlier probe call
was below it (they completed), so no reading taken before the fix moved and gate 4-i stands. Task 8
was specified as "no code": the sitting's commit therefore carries `src/mnimi/store.py` and
`tests/test_store.py` beside this entry, and nothing else.

**Decision: PASS.** Arms B and C run `--active-only on`, and `MemoryConfig.active_only` defaults to
`True` in Task 9's adoption commit. No lexicon, rule, threshold or other default was touched in
response to any number above.

## Gate 4-iii read: ranking adopted (`score`, b=2 c=1), decay not wired (b=2 c=8, X=-6), active_only on (2026-09-18)

**The sitting.** Three mnimi arms at the Task 8 commit `25cde7f` (clean; schema
`mnimi-eval-artifact/10`; `provisional: []` on all three), `gpt-4o-2024-08-06` through the Batch
API in 8 / 8 / 9 sub-batches, one arm in flight at a time, 2026-09-17 22:49 → 2026-09-18 01:57,
every round replayed from the corpus cache (23,302 rows before and after each arm). All three
carried `--extractor qwen3 --render-unit round+facts`; no decay, weight or relevance flag was
passed, so SPEC's values ran. Full tables: `mnimi docs/PHASE4-RESULTS.md` § Task 9.

| arm | flags | score | Wilson 95 % | fed tokens mean |
| --- | --- | --- | --- | --- |
| A `p4base` | the v1.10 read path | 86/100 | [77.9, 91.5] | 5,301 |
| **B `p4rank`** | `--active-only on --ranking score` | **87/100** | [79.0, 92.2] | 5,302 |
| C `p4decay` | B plus `--consolidate on` | 81/100 | [72.2, 87.5] | 5,307 |

| pair (b = the second arm's wins) | b | c | p | rule → outcome |
| --- | --- | --- | --- | --- |
| A → B | 2 | 1 | 1.0000 | `MemoryConfig.ranking` → `"score"` iff b ≥ c → **adopted** |
| B → C | 2 | 8 | 0.1094 | `MNIMI_DEFAULT_CONSOLIDATE` → True iff adopted AND b ≥ c → **did not fire; stays False** |
| A → C | 2 | 7 | 0.1797 | descriptive |

**SPEC's decay-on/off ablation:** X = score(C) − score(B) = **−6 points on n=100** (b=2, c=8,
p=0.1094), **outside** the family's drift band (identical-pin re-runs moved 0 and 2 points with
6/100 flips). Decay at SPEC's 30-day half-life and 0.15 floor costs this benchmark accuracy, and
the Task 8 probe said so first (ANY@10 93 → 89, ALL@10 83 → 75, 13 evidence rounds out of the
top-10): four of the eight rows C loses — `e66b632c`, `06db6396`, `0a34ad58`, `10e09553` — are
among those 13. LongMemEval's evidence is often the oldest round in a haystack, and decay
multiplies exactly that round's score down. The loss sits in knowledge-update (15 → 12) and
single-session-preference (14 → 11); temporal-reasoning is flat at 14.

**Adopted, strictly by the pre-registered rules:** `ranking = "score"`, `active_only = True`
(gate 4-ii), `MNIMI_DEFAULT_CONSOLIDATE = False` (unchanged). The code, the flags and the tests for
decay all stay — decay is built, measured and switched off, which is what the ablation was for.
Exactly the four named default assertions changed; suite 392, ruff clean, CI reproduced.

**Drift, arm A against the published `mnimi__100q_gpt4o_extract`** (v1.9, `82aa8b5`): 91/100
predictions changed at the byte level, prompt tokens changed on 9/100 rows, score 84 → 86 — the
nine are Phase 3's kept facts changing the rendered context, the rest is this family's text drift.
Say of this family "score reproducible within 6/100 flips; text not reproducible"; never
"byte-identical".

**Cost:** readers $2.3316 + judges $0.2287 = **$2.5603** (pre-registered ≈ $2.6; projected upper
bound $3.5075). Ledger: $16.2953 of $50, $33.7047 remaining. The three arms are published under
`results/published/mnimi__100q_gpt4o_{p4base,p4rank,p4decay}/`.

## Phase 4 closes: the SPEC deviations in one place, and what the phase leaves for Phase 5 (2026-09-18, v1.11.0)

**Deviations from SPEC, each disclosed where it was decided and collected here** (PHASE4 D1–D12;
"Phase 4 pre-registration", "Gate 4-i read", "Gate 4-ii read", "Gate 4-iii read"):

1. **The retriever selects the top-k by score, not by vector then score** (D4, CHANGELOG #18).
   §Retriever's "KNN top-k → attach component scores" would rank only what the vector already
   chose, and since `get_context` re-sorts by time, salience and decay could never change what the
   reader sees — the required decay ablation would have measured nothing. `rank_rounds` is exact
   (growing batches, a sound stop bound) and equals `Store.search_rounds` under uniform salience:
   tested in CI and measured 100/100 on the slice's real stores (gate 4-i(b)).
2. **`initial_salience`, a column SPEC's table does not list** (D3). Decay recomputes `salience`
   from it, so the pass is a pure function of stored fields — `consolidate()` twice is
   `consolidate()` once, bit for bit — and an access restores the full value at the next pass. A
   watermark column could do neither.
3. **Decay never raises a salience and both kinds decay** (D3): `max(min(decay_floor,
   initial_salience), …)` keeps a record inserted below the floor where it is, and rounds decay
   with their facts (otherwise an old round would keep ranking on its undecayed round record while
   only its facts sank).
4. **The decay log line carries the record id** and Phase 3's ASCII arrow: `decayed {id}: {days}
   days since last access, salience {old} -> {new}` (D3).
5. **`recency` shares `decay_half_life_days`** — one time constant, no new knob (D4).
6. **`ranking` and `active_only` are switches SPEC did not foresee** (D4, D5, D10), adopted at
   `"score"` and `True` by the pre-registered rules; `"similarity"` / `False` remain the v1.10 read
   path, byte for byte, and two tests pin it.
7. **The `last_accessed` write-back covers each returned round's representative record** (D2), not
   every fact the renderer shows under it — SPEC's "returned hits" read literally. Under the eval
   protocol it cannot move a number (one query per fresh store) and it is shipped because SPEC
   lists the field and decay reads it.
8. **`recall_min_relevance = 0.0` means off by definition** (D7) — the drop is skipped rather than
   applied as "cosine >= 0", and it is applied after the top-k, with nothing filling the gap. Never
   set in a run.
9. **The extractor's salience is SPEC's multiplier start, priced rather than overridden** (D4). On
   the cache it is a speaker flag (user facts 30,730/30,750 at 1.0; assistant facts 21,734/24,530
   at 0.5), so `"score"` halves most assistant facts before any decay. What that cost: nothing at
   the reader — arm B (87) beat arm A (86) — and on the probe it moved one evidence round inside
   the top-10 (`38146c39`, rank 5 → 10) while removing all three down-weighted top-10 carriers.
10. **`consolidate()` reaches the harness behind a pinned switch** (D8), `--consolidate`, default
    off; `MnimiSystem` marks a store pending on `add()` and consolidates once before the question.
11. **Seventeen guard rows** (`decay_rules_hash`, D9) and **pins schema /10** (eight mnimi-only
    keys). A v1.10 store is refused at open by name.
12. **The salience-0 exclusion moved from PLAN row 5.1 into this phase** (D5) and was priced before
    it shipped: gate 4-ii, 0 evidence rounds out of the top-10.
13. **v1.11.0, not v2.0.0** (D12): `recall()` converged on SPEC's locked return type
    (`list[ScoredRecord]`) rather than breaking it — v1.10's `list[MemoryRecord]` was the disclosed
    divergence.

**What the phase measured.** Gate 4-i: both probes identical to Phase 3's on **100/100** questions
(46 supersessions, `misses: 0`, ANY@10 93/95, ALL@10 83/95) — Tasks 2–7 moved no retrieval row.
Gate 4-ii: the exclusion costs the slice one evidence round five ranks (`10e09553`, 1 → 6, still
retrieved) and hides five stale facts; ANY@10 and ALL@10 unchanged, **0** evidence rounds lost, so
`active_only` ships on. The 46 supersessions classified by hand: refinement 11, enumeration 17,
coexisting 13, undecided 2, update 3 — **28 to 43 of 46 are false positives** of the exact-match
rules, two extraction failure modes behind much of it (dropped hyphens in ranges, third-party
values attributed to the user). Reported probes: the score ranking leaves ANY@10 / ALL@10 at 93/83
while reordering most top-50 lists; decay drops them to **89/75** with 13 evidence rounds out of
the top-10. The sitting (three arms, one clean commit `25cde7f`, $2.5603): **86 / 87 / 81**, A → B
b=2 c=1 (score adopted), B → C b=2 c=8 (**decay at −6 points**, not wired), A → C b=2 c=7. Tests
348 → 392, ruff clean, CI reproduced on 3.11 and 3.12 with the `[dev]` extra alone before every
commit. Hands-on ≈ 4.5 h against ≈ 25 planned, plus ≈ 7 h of unattended probe and batch time;
$2.56 of $50, $33.70 remaining.

**One bug found and fixed inside the phase, disclosed** ("Gate 4-ii read"): `Store._knn` never
clamped its `k * 8` over-fetch to sqlite-vec's 4,096-row limit, so any caller wanting more than 512
rows raised. `rank_rounds` reaches that depth once decayed saliences lower the k-th score; the
limit was latent since v1 (`search_rounds` grows the same way). Fixed at the source with a
regression test (`0ccdaa9`), the crashed run kept; the clamp binds only above 512 rows, so no
earlier reading moved.

**What it leaves for Phase 5.** (a) `export()` — the last of the five locked methods; (b)
`context_token_budget` and `raw` in the rendered block — the fed-token lever, still untouched at
~5,300 tokens a question; (c) concurrency (WAL, one write queue, `consolidate` serialized with
`add`, FUTURE.md § Concurrency); (d) **the n=500 sitting, which SPEC §Benchmark contract requires
to report decay-on vs decay-off**: with decay now off by default that means an extra decay-on arm
beside the four, ≈ $3.5 by batch at n=500, and PLAN §4.1 is re-projected for it; (e) the source
provenance pointer (`conversation_id/turn_id/role`), still Phase E; (f) two open questions this
phase raised rather than settled — whether a shorter half-life or a recency weight > 0 would turn
decay from a cost into a gain (both are FUTURE items with triggers, and neither may be selected on
LongMemEval), and whether the exact-match conflict rules should learn containment, which the 46
say costs 28–43 false supersessions today. v1.11.0.
