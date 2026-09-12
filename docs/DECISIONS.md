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
