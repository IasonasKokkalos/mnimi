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
