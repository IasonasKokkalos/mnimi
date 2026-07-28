# FUTURE SPECS AND IDEAS

Deferred / unvalidated work for mnimi. Nothing here is v1 scope. Each entry
tags the SPEC.md section and line(s) that govern or reference it, so the two
documents stay in sync. Section names are given alongside line numbers because
line numbers rot on the next SPEC edit and section names don't.

**Refs last verified against SPEC.md on 2026-07-28.** If you renumber SPEC,
re-verify or drop the numbers and keep the section names.

Legend:
- **[SPEC-deferred]** — SPEC explicitly pushes this here with reasoning.
- **[SPEC-open]** — SPEC lists it as undecided with a stated leaning; it is not
  yet a deferral, and the leaning is not a lock.
- **[FUTURE-origin]** — originated in this file; SPEC has no conflicting lock.
- **[already-enforced]** — listed here historically but is in fact a live v1
  constraint; kept only as a pointer, not real future work.
- **[trigger]** — the condition under which it gets reconsidered.

---

## Retrieval quality (revisit only on W3 evidence)

* **Hybrid FTS5 + reciprocal-rank fusion.** FTS5 lives inside SQLite, so this
  adds no dependency. Helps exact-term queries (IDs, names, numbers) where
  dense retrieval misses. BM25-alone measured well below dense on LongMemEval
  (Table 9); the hybrid's effect is unmeasured. **[SPEC-deferred: §Deferred /
  Unvalidated, lines 713-718]** · **[trigger]** W3 error analysis shows
  exact-term retrieval misses.

* **In-process cross-encoder reranker.** Runs *after* the cosine KNN narrows
  candidates — a reranker over the top-k, never a replacement for the
  embedding pass (too heavy to run across the whole DB). Feasibility is
  source-proven: OMEGA ships cross-encoder/ms-marco-MiniLM-L-6-v2 via ONNX
  (int8 ≈ 571 MB disk / ~650 MB RSS, lazy-loaded). Cost: a second model
  download in the install story. **[SPEC-deferred: §Deferred / Unvalidated,
  lines 719-723]** · **[FUTURE-origin: cross-encoder note]** · **[trigger]** W3
  shows retrieval-*ordering* failures (not recall failures).

* **Time-aware query expansion.** +6.8–11.3% temporal-reasoning recall in the
  paper (Table 4) — but only with a strong LLM extracting time ranges; weak
  models hallucinate ranges and prune wrongly. Also puts an LLM on the read
  path, which the current design forbids. **[SPEC-deferred: §Deferred /
  Unvalidated, lines 730-735]** · **[trigger]** W3 shows TR *retrieval* (not
  reading) is the bottleneck — and even then, try a deterministic date-range
  parser first.

* **Recency weight > 0 in ranking.** `salience_weights` defaults to
  `{similarity: 1.0, recency: 0.0}` because the paper's winning config ranks
  by dense similarity alone (§5.1). A positive recency weight risks burying
  old evidence (information-extraction questions). **[SPEC-deferred: §Deferred
  / Unvalidated, lines 736-737; config at §MemoryConfig, lines 166-170]** ·
  **[trigger]** W3 tuning evidence only.

* **`recall_min_relevance` > 0.** The read-side relevance floor ships at 0.0
  (off) because BGE absolute similarity values are unreliable and retrieval
  recall gates ~90% of correct answers (Fig 14) — a fixed floor risks dropping
  true evidence. OMEGA ships this on at 0.60, untested on LongMemEval-style
  recall. **[SPEC-deferred: §Deferred / Unvalidated, lines 738-739; config at
  §MemoryConfig, lines 156-165]** · **[trigger]** W3 shows distractor-driven
  reading failures.

## Extraction (revisit only if extraction is the measured bottleneck)

**Constraint every entry below inherits:** the extraction output schema orders
free-prose fields (`content`, `raw`) *before* constrained slots
(`subject`/`predicate`/`object`, times), because premature serialization
degrades capacity-limited models. Any future schema change keeps that ordering
and re-hashes `extractor_prompt_hash` — the grammar is part of the pin.
**[already-enforced: §CHANGELOG #12, lines 37-43; §Stage 2 output contract,
lines 280-301]**

* **Extractor LoRA fine-tune.** The measured lever for small-model extraction
  quality: fine-tuned small models can match much larger teachers on
  extraction, and off-the-shelf triple extractors perform poorly untuned
  (PMC12065832; arXiv 2507.13827). Effort cost, not infra cost — no new
  runtime dependency. **[SPEC-deferred: §Deferred / Unvalidated, lines 724-729;
  risk stated at §Stage 2, lines 330-335]** · **[trigger]** W3 error analysis
  shows extraction (not retrieval or reading) is the bottleneck.

* **Deterministic (spaCy / rule) extraction fallback.** An alternative to the
  pinned local LLM for the extraction stage. **[SPEC-deferred: §Extraction →
  Out of scope, lines 389-392; §Deferred / Unvalidated, lines 752-753]** ·
  **[trigger]** LongMemEval number shows extraction is the bottleneck.

* **Pluggable / API-backed extractors.** Swap the hardcoded Qwen3-1.7B for a
  configurable or API model. Rejected in v1 because an API model can change
  under a benchmark run and silently invalidate the number. **Scope of the
  rejection, stated precisely:** it is a *benchmark* prohibition, not a library
  one — an API extractor may become a shipped option, but never the one a
  published number was produced under. **[SPEC-deferred: §Extraction → Out of
  scope, line 389; §Stage 2 (Model), lines 319-321]**

* **Per-record-type extraction schemas.** Different extraction contracts per
  fact type. **[SPEC-deferred: §Extraction → Out of scope, lines 389-390]**

* **Multi-turn coreference windows** beyond the single conversation.
  **[SPEC-deferred: §Extraction → Out of scope, lines 390-391]**

## Decay / salience / lifecycle

* **`pinned` records.** Decay-exempt, conflict-winning identity facts. Removed
  from the v1 schema because no LongMemEval question type depends on
  decay-immunity within the eval horizon, and extraction-inferred pinning is
  an unmeasured judgment handed to a 1.7B model. **[SPEC-deferred: §CHANGELOG
  #6, lines 73-76; §Deferred / Unvalidated, lines 747-748]**

* **Per-type decay rates / permanent types.** OMEGA's `_DECAY_LAMBDAS` (λ=0 for
  constraints/preferences) is effectively `pinned`-by-type; it reintroduces
  the same unmeasured type judgment. Bundled with `pinned` above.
  **[SPEC-deferred: §Deferred / Unvalidated, lines 743-746]**

* **Prospective-fact expiry.** The legitimate residue of the removed
  `valid_time` step-function decay: facts that describe a *future* validity
  window could expire once that window closes. Hard constraint carried over
  from the removal: **past events never expire** (zeroing them forfeits the
  27% temporal-reasoning slice). **[SPEC-deferred: §CHANGELOG #3, lines 60-64;
  §Deferred / Unvalidated, lines 749-751]**

* **Optional purge / GC.** Hard-delete old records for users who don't need
  audit history. Off by default — deletion breaks `export()`'s point-in-time
  guarantee. **Correction from original FUTURE phrasing:** the trigger cannot
  be "below salience floor," because SPEC now clamps decay at
  `decay_floor = 0.15` and reserves salience 0 for superseded records only —
  no *active* record ever falls below the floor. Rewritten trigger: hard-delete
  **superseded** records (salience 0) older than N days. **[FUTURE-origin,
  reconciled against: §MemoryConfig decay_floor, lines 148-150; §MemoryRecord
  note, lines 196-198; §Human-readable memory, lines 689-701]**

## Access-frequency signals

* **Access-frequency ranking / `access_count`.** A per-record access counter
  feeding rank or decay. Provably inert under the LongMemEval protocol
  (reset-per-instance, one query per question — the count is a constant zero at
  query time), and carries a feedback-loop risk in production
  (frequently-recalled → ranks higher → recalled more). Shipped by both OMEGA
  and Hypabase; rejected here on protocol grounds. **[SPEC-deferred: §CHANGELOG
  #7, lines 77-82; §Ranking, lines 517-522; §Deferred / Unvalidated, lines
  740-742]**

## Embedder flexibility (all deferred together)

* **`MemoryConfig.embedder` — swappable embedder at init time only**, never
  mid-corpus (a corpus embedded under one model can't be queried with another;
  the guard raises on mismatch). **Scope clarification (2026-07-28):** the
  `Embedder` *protocol* and the `embedder` *constructor argument* are live
  today — they are in the locked public signature (§Public API, line 109) and in
  `src/mnimi/embeddings.py`. What is deferred is (a) an embedder field on
  `MemoryConfig` and (b) any embedder choice for the *eval run*, which is
  hardcoded. Do not read this entry as "the library has no pluggable embedder."
  **[SPEC-deferred: §Embedding config, lines 562-564; §Embedding config → Out
  of scope, lines 624-626; guard at §Reproducibility guard, lines 361-364]** ·
  **[FUTURE-origin]**

* **Custom embedder model in config init**, **Matryoshka dim truncation**,
  **hot-swap / re-embed workflow**, **API-backed embedder options.** The whole
  pluggable-embedder surface. v1 hardcodes `BAAI/bge-small-en-v1.5` @ 384-dim
  with a pinned HF revision. **[SPEC-deferred: §Embedding config → Out of
  scope, lines 624-626; §Deferred / Unvalidated, lines 752-753]** ·
  **[FUTURE-origin]**

## Architecture questions SPEC leaves open

Both are *undecided with a stated leaning*, not deferrals. The leaning is the
current position, not a lock — reopening either needs a reason, not a vote.

* **Tiered model (core / recall / archival, Letta-style).** Lean *no* for v1:
  the differentiator is write-path policy, not tier structure. A tier system
  would also add a second retrieval surface to hold identical across every
  baseline, which the benchmark contract makes expensive. **[SPEC-open: §Open
  questions, lines 703-709]** · **[trigger]** context-budget pressure that
  ranking and decay demonstrably cannot solve.

* **Self-editing memory (agent edits its own store via tools).** Lean *no*:
  mnimi manages memory *for* the agent, and an agent-writable store puts an
  LLM back inside the bookkeeping loop that SPEC deliberately keeps
  deterministic. **[SPEC-open: §Open questions, lines 703-709]** · related
  lock: §Extraction, lines 337-347.

## Storage / scale / sync (post-SQLite)

* **Graph DB backend for ~1M entities per user.** At personal scale
  brute-force KNN over sqlite-vec is correct; a graph backend is a
  large-scale-only concern. **Hard constraint, already live:** do not hardcode
  sqlite-vec calls into write-path logic, so the store stays swappable.
  **[already-enforced: §MemoryConfig, lines 131-132; §Embedding normalization,
  lines 552-559]** · **[FUTURE-origin: 1M-entity target]**

* **ANN near-duplicate caveat (post-SQLite).** When switching off sqlite-vec's
  brute-force KNN to an approximate index (ANN), a true near-duplicate can
  occasionally miss the candidate set entirely, silently degrading merge
  quality. Worth a comment/test now, real fix later. Note that v1's
  brute-force choice is deliberate and reproducibility-preserving.
  **[FUTURE-origin]** · related SPEC context: §Dedup strategy cosine gate,
  lines 437-439.

* **Multi-device sync.** A raw SQLite file synced across devices risks
  corruption; sync must be a separate layer, not a property of the file.
  **[FUTURE-origin]** · related SPEC context: §Concurrency, lines 628-632.

* **Multi-agent / multi-user concurrent writes + queueing.** Several agents
  (or users) writing at once, with a queueing/sync strategy. v1 serializes all
  writes through a single connection/queue and does not race `consolidate()`
  against `add()`. **[FUTURE-origin]** · related SPEC context: §Concurrency,
  lines 628-632.

## Replay / ablation infrastructure

* **Replayable write path (op-log).** Persist an ordered op-log so a user's
  history can be re-run deterministically under different config (new decay
  half-life, thresholds, etc.) and diffed against the original — enables
  ablations without live re-benchmarking. Complements the logical-clock and
  hash-pin guarantees that already make a run reproducible. **[FUTURE-origin]**
  · related SPEC context: §Logical time, lines 212-220; §Reproducibility guard,
  lines 349-385; decay-on/off ablation at §Benchmark contract, lines 682-687.

## Pattern recognition

* **`memory.recognition` — pattern-recognition memory.** Detect recurring
  patterns across a user's history (beyond fact storage/retrieval). No SPEC
  surface yet; genuinely new capability, not a deferral of existing scope.
  **[FUTURE-origin]** · adjacent to the "learn user patterns, not just store
  facts" gap noted across competitors. **Constraint if ever built:** it does
  not get a public method — §Public API is five methods and locked.

## Packaging / integration

* **MCP wrapper.** Expose mnimi as an MCP server for Claude Code / Cursor /
  etc. Kept out of the library core deliberately — mnimi is import-as-library
  (`pip install → import → wrap`), and library-not-server is a positioning
  point. A wrapper is additive and would not change the core. **[FUTURE-origin]**
  · related SPEC context: the library-vs-server distinction at §Dedup strategy
  prior-art delta, lines 473-476.

## Reproducibility Tiers

### Tier 3 — Containerized reference environment (queued, NOT forced)

**[SPEC-deferred: §Reproducibility Tiers, line 994 ("SEE TIER 3 IN FUTURE.md");
strengthens Tier 2 at lines 818-995]**

**Goal:** make regeneration bit-identical across *machines*, not just across
restarts of one machine, by pinning the entire execution environment rather than
the request options.

**Sketch:** a container image pinning, by digest, the Ollama / llama.cpp build,
the CUDA runtime and driver, and the daemon environment
(`OLLAMA_FLASH_ATTENTION`, `LLAMA_ARG_CACHE_RAM`, `OLLAMA_KV_CACHE_TYPE`) — with
the model blob pulled by digest, not by tag. The image digest is published in the
pins header so a Tier-3 run is identifiable as such. Note the pinning target has
moved since the first sketch of this idea: at the pinned full GPU offload the
variables that decide reduction order are the GPU, driver, CUDA version and
daemon build — not the CPU's AVX2/AVX-512 dispatch, which only governs the
`--num-gpu 0` path.

#### Trigger status: nothing currently forces this

Two candidate forcing conditions were investigated and **both closed without
needing a container**:

* **Flash attention** was a real variable — FA=0 vs FA=1 changed 2/2 probe
  predictions with cache state held constant — and, left unset, resolves to
  `auto`, which is a property of the host GPU. That is exactly the shape of
  problem that forces a reference environment. It did not, because FA proved
  **controllable in-process**: it is pinned via `OLLAMA_FLASH_ATTENTION` and
  asserted at preflight against the daemon's *resolved* value.
* **The prompt cache** produced bistable output that survived daemon restarts,
  and was the last unexplained drift. It also proved **fixable in-process**, via
  `LLAMA_ARG_CACHE_RAM=0` plus a per-question cache-bust prefix. Measured error
  bar after the fix: 0/20 predictions changed on both published systems.

So Tier 3 is **queued for cross-machine reproduction, not a fix for anything
currently broken**. Reading this entry as urgent would misrepresent the state of
the harness: single-machine reproducibility is measured and closed at 0/20, and
Tier 1 already gives hardware-independent verification of the *number*, which is
the claim that carries the credibility. Tier 3 only strengthens Tier 2, which is
the weaker claim and the one that already ships with an honest tolerance
statement.

**Cost, stated so the trade stays visible:** image maintenance, a second CI path,
a GPU-enabled container runtime on every machine that wants to reproduce, and a
model-blob distribution story. Real, and currently unjustified.

**Build this if any of these fire:**
- A measured cross-hardware divergence is large enough to change a published
  conclusion, not just a decimal.
- A third party attempts regeneration, diverges, and cannot diagnose why from
  the `run.environment` block.
- A competitor comparison hinges on a delta of the same order of magnitude as
  cross-hardware noise.
- A new drift source is found that is *not* controllable per-request or
  per-daemon-env — the condition that both closed investigations above failed to
  meet.

Until one fires, the honest tolerance statement in Tier 2 is the correct
position. **[FUTURE-origin: trigger set]**

### `--verify-drift` — regeneration check against committed predictions

**[FUTURE-origin]** · related SPEC context: §Tier 2 verdict-cache drift detector,
lines 985-992; Tier 1 artifacts at lines 764-817.

A harness flag that re-runs `--stage predict` and diffs the fresh predictions
against a committed `predictions.jsonl`, reporting a divergence count (N/20
changed) and the first byte offset of each divergence, rather than only a score
delta.

**Why it is worth building.** The drift investigation was won by an accident of
instrumentation: the judge verdict cache keys on `(question_id,
sha256(predicted))`, so an unchanged re-run that *missed* the cache proved the
reader had moved. That signal is currently a side effect of a cost optimization,
observable only as a `judge_cache_misses` count in the run block, and it is
mute about *where* the outputs diverged. `--verify-drift` promotes it to a
first-class reproducibility instrument: an explicit command whose output is a
divergence count, which is the number Tier 2 actually claims.

It is also the mechanism that would produce the cross-hardware delta Tier 2
promises to publish "once available", and the diagnostic a third party would run
before filing the divergence report that could trigger Tier 3.

**[trigger]** Before the first published cross-machine reproduction attempt, or
the next time a drift hunt starts — whichever comes first. Not blocking any
current number.

---

## Prohibited — do not file these here

These look like deferrals and are not. Each is a lock with a measured or
structural reason; wanting one back is not evidence. They are listed so a future
reader stops rather than opening a FUTURE entry for them.

* **Per-category reader prompts.** One reader prompt across all question types,
  for every system under test including competitors. Category-tuned prompts are
  a documented comparability leak (5–15 point swings). **[already-enforced:
  §CHANGELOG #13, lines 44-48; §Benchmark contract harness pins, lines 652-659]**

* **Wall-clock time in decay, recency, or ordering.** All time is logical
  (`now_logical` = max `system_time` in the store). Wall-clock makes the same DB
  score differently on different days — the exact reproducibility failure the
  hash guards cannot catch. **[already-enforced: §CHANGELOG #8, lines 83-90;
  §Logical time, lines 212-220]**

* **An LLM in the merge / conflict / decay loop, or anywhere on the read path.**
  Extraction is the single permitted LLM, and it is upstream of the
  deterministic pipeline. External evidence for the lock: Mem0's LLM-routed
  ADD/UPDATE/DELETE measured emitting malformed routing ~25% of the time.
  **[already-enforced: §Extraction, lines 337-347]** · this is also why
  "self-editing memory" above leans *no*.

* **`valid_time` as an expiry (step-function decay to 0).** Zeroes every past
  dated event and forfeits the 27% temporal-reasoning slice. The legitimate
  residue is prospective-fact expiry, which is filed above with the
  past-events-never-expire constraint. **[already-enforced: §CHANGELOG #3,
  lines 60-64]**

## Cross-references retired from earlier drafts

These were listed as future work in prior notes but are resolved or superseded:

* *"pluggable embedders / Matryoshka / API embedders — already in FUTURE.md,
  unchanged"* — SPEC lines 752-753 point here; consolidated under **Embedder
  flexibility** above.
* *"cross-encoder reranker"* — SPEC previously double-listed it; consolidated
  under **Retrieval quality → In-process cross-encoder reranker** above.
