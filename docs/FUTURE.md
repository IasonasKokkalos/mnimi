# FUTURE SPECS AND IDEAS

Deferred / unvalidated work for mnimi. Nothing here is v1 scope. Each entry
tags the SPEC.md line(s) that govern or reference it, so the two documents stay
in sync. Line numbers refer to the current SPEC.md.

Legend:
- **[SPEC-deferred]** — SPEC explicitly pushes this here with reasoning.
- **[FUTURE-origin]** — originated in this file; SPEC has no conflicting lock.
- **[already-enforced]** — listed here historically but is in fact a live v1
  constraint; kept only as a pointer, not real future work.
- **[trigger]** — the condition under which it gets reconsidered.

---

## Retrieval quality (revisit only on W3 evidence)

* **Hybrid FTS5 + reciprocal-rank fusion.** FTS5 lives inside SQLite, so this
  adds no dependency. Helps exact-term queries (IDs, names, numbers) where
  dense retrieval misses. BM25-alone measured well below dense on LongMemEval
  (Table 9); the hybrid's effect is unmeasured. **[SPEC-deferred: lines
  662-667]** · **[trigger]** W3 error analysis shows exact-term retrieval
  misses.

* **In-process cross-encoder reranker.** Runs *after* the cosine KNN narrows
  candidates — a reranker over the top-k, never a replacement for the
  embedding pass (too heavy to run across the whole DB). Feasibility is
  source-proven: OMEGA ships cross-encoder/ms-marco-MiniLM-L-6-v2 via ONNX
  (int8 ≈ 571 MB disk / ~650 MB RSS, lazy-loaded). Cost: a second model
  download in the install story. **[SPEC-deferred: lines 668-672]** ·
  **[FUTURE-origin: cross-encoder note]** · **[trigger]** W3 shows
  retrieval-*ordering* failures (not recall failures).

* **Time-aware query expansion.** +6.8–11.3% temporal-reasoning recall in the
  paper (Table 4) — but only with a strong LLM extracting time ranges; weak
  models hallucinate ranges and prune wrongly. Also puts an LLM on the read
  path, which the current design forbids. **[SPEC-deferred: lines 679-684]** ·
  **[trigger]** W3 shows TR *retrieval* (not reading) is the bottleneck — and
  even then, try a deterministic date-range parser first.

* **Recency weight > 0 in ranking.** `salience_weights` defaults to
  `{similarity: 1.0, recency: 0.0}` because the paper's winning config ranks
  by dense similarity alone (§5.1). A positive recency weight risks burying
  old evidence (information-extraction questions). **[SPEC-deferred: lines
  685-686, config at 147-151]** · **[trigger]** W3 tuning evidence only.

* **`recall_min_relevance` > 0.** The read-side relevance floor ships at 0.0
  (off) because BGE absolute similarity values are unreliable and retrieval
  recall gates ~90% of correct answers (Fig 14) — a fixed floor risks dropping
  true evidence. OMEGA ships this on at 0.60, untested on LongMemEval-style
  recall. **[SPEC-deferred: lines 687-688, config at 137-146]** · **[trigger]**
  W3 shows distractor-driven reading failures.

## Extraction (revisit only if extraction is the measured bottleneck)

* **Extractor LoRA fine-tune.** The measured lever for small-model extraction
  quality: fine-tuned small models can match much larger teachers on
  extraction, and off-the-shelf triple extractors perform poorly untuned
  (PMC12065832; arXiv 2507.13827). Effort cost, not infra cost — no new
  runtime dependency. **[SPEC-deferred: lines 673-678, risk stated at
  311-316]** · **[trigger]** W3 error analysis shows extraction (not retrieval
  or reading) is the bottleneck.

* **Deterministic (spaCy / rule) extraction fallback.** An alternative to the
  pinned local LLM for the extraction stage. **[SPEC-deferred: lines 370-373,
  701-702]** · **[trigger]** LongMemEval number shows extraction is the
  bottleneck.

* **Pluggable / API-backed extractors.** Swap the hardcoded Qwen3-1.7B for a
  configurable or API model. Rejected in v1 because an API model can change
  under a benchmark run and silently invalidate the number. **[SPEC-deferred:
  lines 370, 300-302]**

* **Per-record-type extraction schemas.** Different extraction contracts per
  fact type. **[SPEC-deferred: line 370-371]**

* **Multi-turn coreference windows** beyond the single conversation.
  **[SPEC-deferred: line 371-372]**

## Decay / salience / lifecycle

* **`pinned` records.** Decay-exempt, conflict-winning identity facts. Removed
  from the v1 schema because no LongMemEval question type depends on
  decay-immunity within the eval horizon, and extraction-inferred pinning is
  an unmeasured judgment handed to a 1.7B model. **[SPEC-deferred: lines 54-57
  (CHANGELOG #6), 696-697]**

* **Per-type decay rates / permanent types.** OMEGA's `_DECAY_LAMBDAS` (λ=0 for
  constraints/preferences) is effectively `pinned`-by-type; it reintroduces
  the same unmeasured type judgment. Bundled with `pinned` above.
  **[SPEC-deferred: lines 692-695]**

* **Prospective-fact expiry.** The legitimate residue of the removed
  `valid_time` step-function decay: facts that describe a *future* validity
  window could expire once that window closes. Hard constraint carried over
  from the removal: **past events never expire** (zeroing them forfeits the
  27% temporal-reasoning slice). **[SPEC-deferred: lines 41-45 (CHANGELOG #3),
  698-700]**

* **Optional purge / GC.** Hard-delete old records for users who don't need
  audit history. Off by default — deletion breaks `export()`'s point-in-time
  guarantee. **Correction from original FUTURE phrasing:** the trigger cannot
  be "below salience floor," because SPEC now clamps decay at
  `decay_floor = 0.15` and reserves salience 0 for superseded records only —
  no *active* record ever falls below the floor. Rewritten trigger: hard-delete
  **superseded** records (salience 0) older than N days. **[FUTURE-origin,
  reconciled against: lines 129-131 (decay_floor), 177-179 (salience 0 =
  superseded only), 638-650 (export point-in-time guarantee)]**

## Access-frequency signals

* **Access-frequency ranking / `access_count`.** A per-record access counter
  feeding rank or decay. Provably inert under the LongMemEval protocol
  (reset-per-instance, one query per question — the count is a constant zero at
  query time), and carries a feedback-loop risk in production
  (frequently-recalled → ranks higher → recalled more). Shipped by both OMEGA
  and Hypabase; rejected here on protocol grounds. **[SPEC-deferred: lines
  58-63 (CHANGELOG #7), 498-503, 689-691]**

## Embedder flexibility (all deferred together)

* **`MemoryConfig.embedder` — swappable embedder at init time only**, never
  mid-corpus (a corpus embedded under one model can't be queried with another;
  the guard raises on mismatch). **[SPEC-deferred: lines 544-545, 583-585;
  guard at 342-345]** · **[FUTURE-origin]**

* **Custom embedder model in config init**, **Matryoshka dim truncation**,
  **hot-swap / re-embed workflow**, **API-backed embedder options.** The whole
  pluggable-embedder surface. v1 hardcodes `BAAI/bge-small-en-v1.5` @ 384-dim
  with a pinned HF revision. **[SPEC-deferred: lines 583-585 (Out of scope →
  FUTURE.md), 701-702]** · **[FUTURE-origin]**

## Storage / scale / sync (post-SQLite)

* **Graph DB backend for ~1M entities per user.** At personal scale
  brute-force KNN over sqlite-vec is correct; a graph backend is a
  large-scale-only concern. **Hard constraint, already live:** do not hardcode
  sqlite-vec calls into write-path logic, so the store stays swappable.
  **[already-enforced: lines 112-113 (MemoryConfig — never hardcoded in
  write-path logic), 533-540 (normalize/store boundary discipline)]** ·
  **[FUTURE-origin: 1M-entity target]**

* **ANN near-duplicate caveat (post-SQLite).** When switching off sqlite-vec's
  brute-force KNN to an approximate index (ANN), a true near-duplicate can
  occasionally miss the candidate set entirely, silently degrading merge
  quality. Worth a comment/test now, real fix later. Note that v1's
  brute-force choice is deliberate and reproducibility-preserving.
  **[FUTURE-origin]** · related SPEC context: dedup cosine gate at lines
  418-420.

* **Multi-device sync.** A raw SQLite file synced across devices risks
  corruption; sync must be a separate layer, not a property of the file.
  **[FUTURE-origin]** · related SPEC context: single-writer/WAL concurrency at
  lines 587-591.

* **Multi-agent / multi-user concurrent writes + queueing.** Several agents
  (or users) writing at once, with a queueing/sync strategy. v1 serializes all
  writes through a single connection/queue and does not race `consolidate()`
  against `add()`. **[FUTURE-origin]** · related SPEC context: Concurrency at
  lines 587-591.

## Replay / ablation infrastructure

* **Replayable write path (op-log).** Persist an ordered op-log so a user's
  history can be re-run deterministically under different config (new decay
  half-life, thresholds, etc.) and diffed against the original — enables
  ablations without live re-benchmarking. Complements the logical-clock and
  hash-pin guarantees that already make a run reproducible. **[FUTURE-origin]**
  · related SPEC context: logical time at lines 193-201, reproducibility guard
  at lines 330-366, decay-on/off ablation at lines 631-636.

## Pattern recognition

* **`memory.recognition` — pattern-recognition memory.** Detect recurring
  patterns across a user's history (beyond fact storage/retrieval). No SPEC
  surface yet; genuinely new capability, not a deferral of existing scope.
  **[FUTURE-origin]** · adjacent to the "learn user patterns, not just store
  facts" gap noted across competitors.

## Packaging / integration

* **MCP wrapper.** Expose mnimi as an MCP server for Claude Code / Cursor /
  etc. Kept out of the library core deliberately — mnimi is import-as-library
  (`pip install → import → wrap`), and library-not-server is a positioning
  point. A wrapper is additive and would not change the core. **[FUTURE-origin]**
  · related SPEC context: the library-vs-server distinction at lines 454-457.

---

## Cross-references retired from earlier drafts

These were listed as future work in prior notes but are resolved or superseded:

* *"pluggable embedders / Matryoshka / API embedders — already in FUTURE.md,
  unchanged"* — SPEC line 701-702 points here; consolidated under **Embedder
  flexibility** above.
* *"cross-encoder reranker"* — SPEC previously double-listed it; consolidated
  under **Retrieval quality → In-process cross-encoder reranker** above.