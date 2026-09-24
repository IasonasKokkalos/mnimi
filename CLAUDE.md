# CLAUDE.md — mnimi

Project-level instructions for working in this repo. Read before changing code.
`docs/SPEC.md` is the contract; this file is the short version plus the rules
that are easy to break silently. On any conflict, SPEC wins — and fix this file.

## What this is

An embeddable, local-first agent-memory library. Thin public API, deep write-side
policy. Zero infra: one SQLite file via `sqlite-vec`, no server, no external
services. The library must import and run with only its two core deps.

## Stack

- **Language:** Python (>=3.10), src-layout (`src/mnimi/`).
- **Storage:** SQLite + `sqlite-vec` for vector search. One file on disk.
- **Core deps:** `sqlite-vec`, `numpy`. Nothing else. Adding a third is a
  decision, not a convenience — justify it in `docs/DECISIONS.md`.
- **Embedder extra** (`[embed]`): `onnxruntime`, `tokenizers`,
  `huggingface-hub` — the real BGE embedder. Optional so the core stays two
  deps; `HashingEmbedder` (numpy-only) is the default import path and CI.
- **Eval-only deps** (`[eval]` extra): `huggingface-hub` (dataset download),
  `ollama` (reader transport), `openai` (judge), `python-dotenv`. Never imported
  by the library. The Anthropic client is gone — reader is local, judge is
  OpenAI (`docs/DECISIONS.md`).

## API contract (locked — keep these signatures)

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder,
                 config: MemoryConfig = MemoryConfig()): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # -> list[ScoredRecord]
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
    def export(self, user_id: str) -> str                   # human-readable dump
```

Five methods. Do not widen the surface. Depth goes behind `add` and
`consolidate`, not into new public methods.

The contract also lives in the **message shape**: each message dict is
`{"role": "user"|"assistant", "content": str, "ts": ISO}`. `ts` is the session
timestamp, becomes the record's `system_time`, and is the anchor for relative-date
resolution. Both roles are in extraction scope.

## Locked decisions

- **Language = Python.**
- **Storage = sqlite-vec** (local-first, zero-infra). Not a hosted vector DB.
- **All time is logical.** `now_logical` = max `system_time` in the user's store.
  No library code path reads wall-clock for scoring, decay, recency, or ordering.
- **Exactly one LLM in the library, at extraction.** Merge / conflict / decay /
  both dedup screens stay deterministic. No LLM on the read path.
- **Extractor (built 2026-09-13; adopted 2026-09-15 at gate 4-iii, 84 vs 80, b=7, c=3):**
  `Qwen/Qwen3-1.7B-GGUF` @ `90862c4b9d2787eaed51d12237eafdfe7c5f6077`,
  `Qwen3-1.7B-Q8_0.gguf` — there is no official "Instruct" repo; the post-trained
  hybrid runs with thinking off through the template's empty think block — via
  llama-cpp-python 0.3.35 built with CUDA 13.2, grammar-constrained
  (greedy-verify-rescan), `temperature=0`, `top_k=1`, `seed=0`, `n_ctx=4096`,
  `n_batch=512`, `n_threads=8`, `n_gpu_layers=99` (a disclosed deviation from
  SPEC's CPU pin; 50/50 byte-stable across a shuffled restart), prompt
  `qwen3-fact-v4` (hash `41c8ea2c3bb4…`), 1,536-token input cap. Local, pinned,
  never API. Injected as `Memory(..., extractor=...)`; the harness's mnimi
  arm runs it unless `--extractor none` (`evals.__main__.default_extractor`);
  the model sees each round once (on-disk cache, 23,302 rounds for the slice).
  Every round keeps its v1 record; facts are extra records on the round (D1).
  The CUDA runtime DLLs live in `CUDA\v13.2\bin\x64`, which must be on PATH.
- **Embedder:** `BAAI/bge-small-en-v1.5` @ 384-dim, HF revision pinned to
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (machine-resolved, not
  maintainer-vetted — see SPEC §Embedder as built). Vectors unit-normalized
  inside `embeddings.py`; vec0 uses `distance_metric=L2`.
- **Embed/render split (v1.3.0).** Two templates, two hashes, two lifetimes.
  `EMBED_TEMPLATE` builds the embedded string and the dedup key — **frozen**,
  hashed into `memory_meta`; an edit moves every vector and voids the evidence
  the dedup threshold was selected on. `RENDER_TEMPLATE` builds what a reader
  sees — free to evolve, hashed into the harness pins as `render_template_hash`
  so a format change is loud there instead. Never collapse the two.
- **Embedded string, scoped by era:** extraction-era = `f"{raw}\n{content}"`;
  v1 (pre-extraction — no `raw` exists yet) = `[Session date: {date}] {text}`,
  role as metadata only, never in the vector (a role prefix drags all
  similarities together and degrades the cosine dedup threshold). Reopens when
  extraction lands — SPEC CHANGELOG #15. **Built 2026-09-13:** fact records
  embed `FACT_EMBED_TEMPLATE = "${raw}\n${fact}"` (the role-prefixed span, then
  the fact) under their own `memory_meta` row, `fact_embed_template_hash`; round
  records keep `EMBED_TEMPLATE` byte for byte, so no v1 vector moved.
- **Ranking:** `score = (w_sim·relevance + w_rec·recency) · salience`, defaults
  `{similarity: 1.0, recency: 0.0}`. Salience is a multiplier, not a term. No
  `access_count` — inert under the eval protocol. **Built in Phase 4 and adopted**
  (`MemoryConfig.ranking = "score"`): the k rounds are *selected* by score over an
  exact candidate pool, not scored after a vector top-k (SPEC CHANGELOG #18) — read
  literally, the spec'd order would make decay unable to change what the reader sees.
  With uniform salience it equals `Store.search_rounds` record for record (tested;
  gate 4-i(b) on 100 real stores).
- **Salience 0 means superseded, and only that**, and `Store.insert` refuses a
  salience outside [0, 1]. Decay clamps at `decay_floor` (0.15): it down-ranks,
  never excludes. The read path reads salience since v1.11.0 — as the ranking
  multiplier and, under `active_only`, as the active-record filter.
- **Conflict (Phase 3, built 2026-09-15/16; gates 3-i and 3-ii passed, `mnimi docs/PHASE3.md` D1–D12):**
  only fact records conflict; a round is evidence and is never superseded (D1).
  The negation lexicon (`mnimi/conflict/lexicon.py`: 19 contractions, 20
  markers, ten antonym groups) and the normalization tables plus fourteen
  functional predicate groups (`normalize.py`) are **frozen and hashed** —
  `negation_lexicon_hash = 330604b5772e…`, `conflict_rules_hash = 7d19c48828c8…`,
  both `memory_meta` rows, both pinned by tests; an edit is a version bump, a
  new hash, a re-ingest and a dated DECISIONS entry, never in place and never to
  make a gate pass. Normalization is exact match after a fixed rewrite — no
  lemmatization, no fuzzy matching (D3). Two facts on one `pair_key` conflict
  **only under a rule** — `negation` (one object, opposite polarity),
  `functional` (a frozen group, two positive values), `numeric` (different
  numbers on one residue) — otherwise they are two facts (D2, CHANGELOG #17: a
  bare same-pair rule would have zeroed 33,963 true facts on the cache); the
  assistant's facts never conflict. One ordering: effective time (`valid_time`
  else the session date), session date, raw `ts`, the user's span over the
  assistant's, id — greater wins (D6); loser `salience = 0`, winner
  `supersedes` = the last loser's id, `superseded {old_id}: {rule} {pair_key}:
  {old} -> {new}` at INFO on `mnimi.memory`. `conflict_resolution = True` is
  the one switch (screens, entropy gate, supersession); `False` is the v1.9
  write path byte for byte. **The read path is untouched (D8):** a superseded
  fact still ranks and renders until Phase 4 reads `salience`, so nothing in
  Phase 3 moved a benchmark number and none is claimed.
- **Decay and the read side (Phase 4, built 2026-09-17/18; gates 4-i, 4-ii, 4-iii,
  `mnimi docs/PHASE4.md` D1–D12).** All time stays logical: `now_logical` is the
  user's latest dated session timestamp (`mnimi.decay`), never a clock and never the
  question's date. Decay lives in `consolidate()` after the conflict pass, over every
  record with `salience > 0` of both kinds: `salience = max(min(decay_floor,
  initial_salience), initial_salience · 0.5 ** (days(now_logical, last_accessed) /
  decay_half_life_days))`, one transaction, one `decayed {id}: …` line at INFO. The
  rules are frozen and hashed — `decay_rules_hash = d4a0bcf07330…`, the seventeenth
  `memory_meta` row — and `initial_salience` (the inserted value, never updated) is
  what makes the pass idempotent and an access restorative. Adopted defaults:
  `ranking="score"` (gate 4-iii: 87 vs 86, b=2, c=1) and `active_only=True` (gate
  4-ii: 0 evidence rounds left the top-10). **Decay itself is built, measured and
  off**: wired into the harness behind `--consolidate` only, because the sitting read
  it at **−6 points** (81 vs 87, b=2, c=8) — SPEC's required decay-on/off ablation,
  reported as the negative result it is. `recall()` returns `list[ScoredRecord]`;
  `recall_min_relevance` ships at 0.0 (off by definition) and is never set in a run.
- **Phase 6 — the reachable 54 (built 2026-09-22/24; `mnimi docs/PHASE6.md` D1–D11,
  gates 6-i…6-iv; closed at v2.12.0).** Three levers against the 54 questions oracle answers
  and the published `mnimi__500q_gpt4o` (422) does not, each pre-registered, each $0 at the
  probe, each an n=500 arm paired against the *published* artifact and adopted iff b ≥ c.
  **L1, the time-aware term (adopted):** the question date reaches mnimi as the documented,
  stripped query prefix `"[Current date: <ts>]\n<question>"` (D2 — the question's `ts` is
  logical time too; the bare question is what gets embedded), `mnimi/temporal.py` parses the
  question's own relative-date expression under a frozen, hashed grammar (`temporal_rules_hash`,
  a harness pin; ±3-day slack) and SPEC's score gains `time_weight · time_match`;
  `MemoryConfig.time_weight` is its **own field** (not a `salience_weights` key — disclosed),
  0.05 fixed before the probe and **the default since v2.12.0**; `resolver_version = "v2"` rides
  with it (D4: future-marked mentions resolve forward, `last weekend`; a v1.12 store is refused).
  Gate 6-i PASS (identity 448/448 on the unparsed rows, 4 of 29 completed); arm 1 **429 vs 422,
  b=15, c=8**. **L2, the cross-encoder rerank (built, off):** `ranking="rerank"`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2` @ `233902d2…`, `rerank_pool=50`, admitted by the
  maintainer's ruling as a pair classifier; gate 6-ii **FAIL** (ANY@10 455, ALL@10 414 vs the
  bar 459 / 430, 29 rows lose evidence — a reorder that swaps 18 completions for 19 losses); no
  arm. **L3, `render_unit="turns"` with the extractor on (adopted, the default since v2.12.0):**
  facts stored and retrieved, none rendered; arm 2 **424 vs 422, b=22, c=20** — 10 of the 11
  header-implicated reading misses right without the header, temporal-reasoning −5 (D6's L3′,
  a `round+dated-facts` unit, is filed). **The headline, L1 + L3: 426/500 (85.2 %), Wilson
  [81.8, 88.0], b=18, c=14 against the published 422** (p = 0.60; descriptively three below L1
  alone, b=14, c=17). **The 85 criterion is NOT met** (441; lower bound 81.8). Of the 54, 26 are
  right in some arm, 7 in all, **28 in none** (18 partial-retrieval rows — the multi-evidence
  residue); headroom after: 74 wrong, oracle right on 49. None of P1–P5 held in full; no rule
  moved. Three arms $12.23 of the $13 cap; all `provisional: []`, manifests complete, paired by
  `evals.stats` (which since v2.9.0 *reports* a schema or commit difference instead of refusing
  it, as `evals.drift` does); `--verify-drift` cannot be used against the published arm (the
  pins differ in `resolver_version`). **Reader and judge are one snapshot** — the paper's own
  pairing, disclosed: a constant leniency cancels in every paired row and shifts absolute
  scores by an unsigned amount on top of the 6/100 flip rate; the maintainer chose to disclose
  and keep it (2026-09-23). Language: every arm is "the n=500 reading of configuration X,
  b=… c=… against the published 422", never a replication, never a significant gain.
- **Benchmark = LongMemEval** (`longmemeval_s`, ~500 questions). The harness in
  `evals/` is the source of truth.
- **Baselines — five systems, roles marked.** `no_memory` (the floor);
  `full_history` (a truncated-context baseline, **not** a ceiling — at the
  pinned 32K it truncates and the reader sees only a fraction of each history,
  so it measures what naive context-stuffing buys, not what is achievable);
  `oracle` (the **evidence-availability bound**, the paper's §5.5 choice — it
  bounds what evidence reaches the reader, not how well it is presented, so a
  focused retriever handing fewer, cleaner tokens can match or exceed it and
  being at or above oracle is not prima facie a bug; **on the gpt-4o family
  `full_history` is cited from the paper — Fig. 3b, GPT-4o with Chain-of-Note,
  64.0% — never run: the harness refuses it on the openai transport, decided
  2026-09-12**); `naive_rag` (the paper's
  strong K=V baseline, ingestion granularity identical to mnimi's — and
  **pre-registered as a NULL at v1**: v1's only write-side delta is a near-inert
  dedup screen on a benchmark constructed without conflicting facts, so the
  beat-it-by-a-margin criterion attaches to the extraction era). Competitor runs
  (OMEGA first) go through the same harness.
- **Harness pins — two reader families (pins schema /6), never paired against
  each other.** *Local family* (`--reader-transport ollama`, the default):
  reader = local Ollama `qwen2.5:1.5b-instruct-q4_0`
  (`num_gpu=99`, `num_batch=512`, `num_thread=8`, `top_k=1`, `seed=0`, temp 0,
  `num_ctx=32768`) **plus three daemon-level pins that cannot be sent per request**
  (`OLLAMA_FLASH_ATTENTION=1`, `LLAMA_ARG_CACHE_RAM=0`, and the server build
  `READER_TRANSPORT_VERSION = "0.32.13"` — checked against `GET /api/version`
  before any model load; a build change moved 20/20 predictions on
  byte-identical prompts, so any other build is refused). *gpt-4o family*
  (`--reader-transport openai`, decided 2026-09-11): reader =
  `gpt-4o-2024-08-06` over the OpenAI API — the paper's reader and judge
  snapshot — temperature 0, `seed=0`, `max_tokens=800`, `num_ctx=128000`
  (the model's window; `full_history` is not run on this family — its
  128K sitting is ~$15 for a row the paper reports, so the 64.0% is cited
  and the harness refuses the arm); the six Ollama-only pins are `None`; the served
  `system_fingerprint` is recorded per call in `reader_resolved.json` and
  `run.environment`, never pinned, so this family is reproducible only
  within measured drift; every run projects its cost from the real request
  bodies before the first call and refuses above the remaining budget (the
  API budget was $50 for the programme (Phases 0-5, $33.85 spent); on
  2026-09-22 the maintainer made $50 more available, so `API_BUDGET_USD`
  is 83.85 (the spend at that moment plus fifty; the ledger is never reset), `mnimi docs/PLAN.md`;
  `evals/pricing.py`); Batch API runs are sequences of sub-batches under the
  org's 90,000 enqueued-token cap (`evals/batch.py`, 2026-09-12 — a whole
  n=100 arm fails validation otherwise; not a pin). Both families: reader prompt =
  `mnimi-con-v1` (LongMemEval Fig 13 with exactly two deviations — one
  abstention sentence, the `${cache_bust}` prefix — so it is *not* the paper's
  prompt and is never named as one; `json-con-paper-v1` stays reserved for a
  byte-exact reproduction); `render_template_hash` pins the shared context
  format; judge = `gpt-4o-2024-08-06`, prompt `longmemeval-paper-v3`, temp 0,
  `max_tokens=10`. Reader is pinned in `evals/`, never in `memory_meta` — it is
  a harness property, not a store one.
- **One reader prompt for every question type**, for every system under test.
  Per-category answer-prompt tuning is prohibited; it is a comparability leak.

## Invariants that silently invalidate a number

Break one of these and the benchmark still runs — it just stops meaning anything.

- **`memory_meta` guard.** Embedder name/revision/dim, embed-template hash,
  extractor model/quant/runtime/decode-hash/prompt-hash, negation-lexicon hash,
  conflict-rules hash, prefilter-lexicon hash. Written once at DB creation, all
  checked on load, mismatch raises. (**Seventeen rows** since Phase 4, 2026-09-17
  — the four embedder rows, the R4 chunk pair, the five extractor rows or the
  literal `"none"`, `negation_lexicon_hash` live at `330604b5772e…`,
  `conflict_rules_hash` `7d19c48828c8…`, `prefilter_lexicon_hash`,
  `fact_embed_template_hash`, `resolver_version`, and Phase 4's `decay_rules_hash`
  `d4a0bcf07330…` over the nine frozen lines of `mnimi.decay.DECAY_RULES`; the v1.9
  header's "thirteen" was a miscount of fifteen; a v1.8, v1.9 or v1.10 store is
  refused.) Editing a lexicon or prompt is a versioned migration plus re-ingest
  — never an in-place edit under a run.
- **Normalize at the boundary.** Any insert path that bypasses `embeddings.py`
  breaks ranking correctness silently.
- **The screens only ever route a pair away from a merge.** They never drop a
  record the v1.9 path kept, and rounds go through the v1.9 screens byte for
  byte; `conflict_resolution=False` IS the v1.9 write path (a test compares
  stored contents). Gate 3-i's reading of this: 326 keeps, 0 new drops, no
  round moved, records conserved at 83,788.
- **Config, not constants.** Every threshold reads from `MemoryConfig`. No
  hardcoded threshold, no hardcoded `k=5` in write- or read-path logic.
- **Ingestion granularity is per-round**, and `naive_rag` must match mnimi's
  exactly. `k` is identical across all systems.
- **One renderer.** Every context-bearing arm renders through
  `mnimi.memory.render_turns` / `render_records`. Per-arm item granularity
  differs (mnimi renders rounds, `full_history` renders sessions); the format
  does not. Date granularity and speaker labels were a measured cross-arm
  confound before this, not cosmetics. The renderer has two *formats*
  (v1.6.0, `--render-format text|json`, `MemoryConfig.render_format`) that
  frame the same blocks; the harness hands one value to every arm and pins
  `render_template_hash(fmt)`, so a text run and a JSON run never share a
  pins hash and never pair. Which format the gpt-4o era runs is the
  pre-registered presentation pair's call (DECISIONS 2026-09-12) — do not
  flip the default on taste.
- **Sampling is stratified.** The dataset is category-clustered, so a file-order
  `--limit` slice is single-category and not comparable across systems.
- **Run documentation (the rule of 2026-09-22, DECISIONS "The run-documentation rule
  lands").** Every run also writes `runs/<run_id>/manifest.json` (purpose, claim, rule commit,
  commit + clean tree, versions, the arm's configuration, reader requested/served, judge +
  replay count, dataset + question-id digest, environment, cost, timings) and a row in the
  append-only registry `runs/INDEX.md` (aborted runs included; only `status` changes);
  `predictions.jsonl` rows carry `retrieved_ids` and the judge's `verdicts` (the first never
  overwritten — a re-grade is `judge_replay_N.json`); `summary.json` holds the counts and
  intervals. Say PROVISIONAL before a dirty-tree run; never overwrite a run directory
  (`--overwrite-run-dir` only after asking); `--claim` needs `--rule-commit`. Paired
  comparisons come from rows (`python -m evals.stats` saves them under `analyses/`), never
  from summary scores; a quoted number cites its run_id; cited numbers (`full_history` 64.0)
  live in their own table. Promotion is `python -m evals.publish <run_dir>` — clean tree and a
  complete manifest, or it refuses. At the end of a run read the manifest line: missing fields
  make the run `incomplete`.
- **Run artifacts.** Every run writes `pins.json` / `predictions.jsonl` /
  `results.json` under `runs/<system>__<Nq>/`. `runs/` is gitignored scratch; a
  number that gets quoted has its three files copied to
  `results/published/<system>__<Nq>/` and committed. A deviated decode pin, a
  dirty harness tree, a reader prompt other than `mnimi-con-v1`, or a file-order
  slice each mark the artifact **PROVISIONAL — not publishable**; don't quote a
  provisional number. Provisional and auditable are separate claims — a
  provisional artifact is still fully auditable. **`results/published/` holds
  the five-arm n=100 sitting of 2026-08-16 (Ollama 0.32.13, sha `ae5da2b`
  clean, `provisional: []` on all five — the published number, see its
  README for the full provenance), the predict-only restart-pair evidence
  `mnimi__100q_restart_2026-09-10` (100/100 rows identical to `mnimi__100q`),
  plus the two provisional n=20 `plain-prose-v2` smoke dirs. The 0.32.5 n=100
  set (dirty tree) was never published. Since 2026-09-12 it also holds the
  gpt-4o family's sitting — `<arm>__100q_gpt4o/` for the four arms at
  `dd73736` clean (`no_memory` 5, `oracle` 90, `naive_rag` 80, `mnimi` 75;
  `full_history` cited at 64.0), the two `_json` presentation-pair arms and
  `mnimi__100q_gpt4o_drift_2026-09-12`, `provisional: []` throughout — and
  the Phase 1 variant pairs (`*_r3{,base}`, `*_r45{,base}`,
  `naive_rag__100q_gpt4o_r45`, `*_k10`, `*_k20`): mnimi 79 → 81/82 under
  session scope + the BGE query instruction, naive_rag 82, primary b=2, c=3.
  Phase 2 (2026-09-15, `043c0ea`): `mnimi__100q_gpt4o_{p2base,extract,extract_facts}`
  and `naive_rag__100q_gpt4o_p2` at `82aa8b5` — 80 / 84 / 78 / 79; extraction
  adopted with `round+facts`; primary mnimi 84 vs naive_rag 79, b=7, c=2, p=0.18.
  Phase 5 (2026-09-22, `ab09448`): the five-arm n=500 sitting at `f07c24d` clean —
  `{no_memory,naive_rag,mnimi,oracle}__500q_gpt4o` and `mnimi__500q_gpt4o_decay`,
  6.2 / 74.6 / **84.4** / 91.8 / 78.2, `provisional: []` throughout; the whole benchmark,
  and the n=100 slices nest inside it, so nothing here replicates an n=100 number.
  Phase 4 (2026-09-18, `378a19c`): the three-arm ablation at `25cde7f` —
  `mnimi__100q_gpt4o_p4base` 86 (the v1.10 read path), `…_p4rank` **87** (SPEC's read
  path, adopted), `…_p4decay` 81 (with decay, not adopted); A→B b=2 c=1, B→C b=2 c=8
  (decay X=−6), `provisional: []` throughout. **Phase 6 (2026-09-24, `32d5dcd`): the three
  n=500 arms paired against the published `mnimi__500q_gpt4o` — `mnimi__500q_gpt4o_p6time`
  (L1) 429, `…_p6turns` (L3) 424, `…_p6combo` (L1 + L3, the headline) **426**; b/c 15/8, 22/20,
  18/14; `provisional: []` and `status: complete` throughout, at `f0a7de3` / `3845c4e` /
  `606e110` clean, each promoted by `python -m evals.publish`, each pair under `analyses/`.**
- **`question` + `answer` stay inline in `predictions.jsonl`.** That is the only
  reason Tier 1 exists; removing them to denormalize deletes the audit path.
- **A run that cannot be projected cannot spend.** On the API family the
  harness prices the reader and the judge from a dated table
  (`evals/pricing.py`, `PRICES_AS_OF`), confirms the snapshot is served
  (`models.retrieve`), projects an upper bound from the real request bodies
  before the first call, and refuses when projection plus the ledger's spend
  exceeds the budget (`API_BUDGET_USD = 50`). Actual spend from the API's
  `usage` counts is appended to `.cache/api_ledger.jsonl` (gitignored — a fact
  about this key, not the code) after every run; `python -m evals.pricing`
  prints it. `--api-budget-usd` is the one override and is echoed loudly.
  Money never enters `pins_hash`.
- **The judge is not deterministic on borderline rows.** gpt-4o at temperature 0
  flipped 9 verdicts in 140 gradings (question `eace081b`: 9 yes / 11 no across
  20 cache-bypassed re-grades). Absolute scores carry judge instrument error on
  top of question-sampling error, and no absolute difference smaller than that
  rate is interpretable. Paired comparisons are unaffected.
- **The daemon precondition is part of the run.** Launch the pinned build
  manually — on the development machine `/d/ollama-0.32.13/ollama.exe serve`
  (the release zip, version-named folder, no tray, no updater) with
  `OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0` and stderr redirected to the
  file `OLLAMA_SERVE_LOG` names. The tray app is not installed and must not be:
  its updater moved the build three times (SPEC § daemon precondition).
  Preflight asserts the daemon's *resolved* values and the build, and refuses
  otherwise. Kill `llama-server` children too — they outlive the daemon and hold
  VRAM. Preflight anchors its log read on the runner-start marker, not on a
  byte offset (`d70c191`, 2026-08-16) — the 400 KB tail window that refused
  three verifiably correct arms at n=100 on 2026-07-30 is gone, and the log
  volume of an n=500 sitting no longer matters. `ollama stop` before each arm
  stays as sitting procedure because it makes every arm begin with the pinned
  warm-up load, not as a workaround.

## Reproducibility tiers (what a number is allowed to claim)

- **Tier 1 — auditable.** `python -m evals --stage judge --predictions <file>`
  re-grades committed predictions on any machine: no GPU, no Ollama, no dataset,
  no `--system`. Read-only. Proves scoring, not generation. Say *auditable*,
  never *reproducible* — this is a language rule, not a preference.
- **Tier 2 — reproducible given the pins AND the daemon precondition.**
  `--stage predict` reproduces predictions on comparable hardware. Current
  measured error bar (2026-09-10, Ollama 0.32.13): a fresh mnimi n=100 predict
  replayed **100/100 predictions byte-identical to the published artifact**,
  prompt tokens identical on every row, across a daemon restart, a reinstall of
  the server binary, 25 days, three commits and the schema /5 bump. Predictions
  changed: 0. Score delta: 0. Evidence:
  `results/published/mnimi__100q_restart_2026-09-10/`, re-derived by
  `python -m evals.drift` (v1.6.1; `--verify-drift <reference>` on the
  predict stage writes `drift.json` beside fresh predictions — the N/n
  changed count is the Tier 2 statement, and a pair whose shared pins differ
  is refused). **The gpt-4o family's measured Tier 2 (2026-09-12):** the
  mnimi drift pair, identical `pins_hash`, changed 85/100 predictions at the
  byte level with prompt tokens identical on every row, score 75 → 75, 3 rows
  flipped each way — say "score reproducible within 6/100 flips; text not
  reproducible", never "byte-identical", of this family. Three further
  readings during Phase 1 (61, 65, 61 of 100 changed; scores within 2)
  agree. The 2026-07-30 n=20
  four-arm figure (0.32.5) is consistent but superseded — its build is refused
  now. No bit-identity claim across different GPUs, drivers or CUDA versions;
  a different Ollama build is refused, not tolerated. The
  `run.environment` block is diagnostic only and never enters `pins_hash`;
  `pins_hash` identifies what was *requested*, the run block what was *resolved*.
  **Retired — do not cite:** the "12/20 changed, 5 points" figure (judge-stage
  cache bookkeeping over predictions of unknown provenance) and the
  `plain-prose-v2`-era 0/20 restart table (retired prompt and renderer;
  `evals/stats.py` hard-fails pairings across that boundary).
- **Tier 3** (containerized reference environment) is queued, not forced — see
  `docs/FUTURE.md`.
- **The verdict cache is the drift detector.** An unchanged re-run that misses
  the cache means the reader moved. Check `run.stage` before quoting any
  reproducibility number: only `stage='all'` or a fresh `predict` re-runs the
  reader.

## Kill gates

- **W3 — met 2026-07-30** (gate date was Jul 17, 2026). The artifact is all five
  systems on one stratified slice at n=100, same reader, same prompt, one
  sitting. The v1 criterion was Holm-significant separation from `no_memory`,
  and it cleared on the held-out 80 alone (the n=20 dev slice the 0.95 dedup
  threshold was selected on sits inside the headline and is reported
  separately). `mnimi` vs `naive_rag` — the pre-specified PRIMARY and a
  pre-registered null — came back non-significant, which is the expected outcome
  at v1 and not a negative finding. **Scores, paired statistics and token counts
  live in `runs/<system>__100q/` and the run write-up, never in this file or in
  SPEC**; all five artifacts are provisional (dirty harness tree at run time).
  n=20 remains a smoke test only — categories sit at n=3-4 there, one question
  is 5 points.
- **W6 (Aug 7, 2026):** either a competitive LongMemEval number, or reframe the
  project as a zero-infra usability play (the "one file, no server" pitch) rather
  than a state-of-the-art accuracy play. Superseded by the programme in
  `mnimi docs/PLAN.md` (done = SPEC-complete or Wilson lower bound ≥ 85 on the
  gpt-4o family). Phase 0 closed 2026-09-12: mnimi 75 / naive_rag 80 / oracle
  90 at n=100 on gpt-4o; the primary is a pre-registered null (b=1, c=6).
  Phase 1 closed 2026-09-13 (81/82 vs 82, b=2, c=3); Phase 2 closed
  2026-09-15: extraction adopted, mnimi 84 vs naive_rag 79 (b=7, c=2,
  p=0.18 — ahead, not significant; the verdict is Phase 5's n=500). Phase 3
  closed 2026-09-16 at $0: gates 3-i and 3-ii passed, the screens and supersede
  are the library default; no benchmark number moved and none is claimed (D8).
  Phase 4 closed 2026-09-18 for $2.56: gates 4-i (identity 100/100) and 4-ii (the
  exclusion priced, 0 evidence rounds lost) passed, and the three-arm sitting read
  86 / 87 / 81 — SPEC's read path adopted (b=2, c=1), decay measured at −6 points
  (b=2, c=8) and left off. The adopted arm is 87 beside the published naive_rag 79
  and oracle 90, descriptively; the primary is Phase 5's n=500. **Phase 5 closed
  2026-09-22 for $17.55 (v1.12.0): the five-arm n=500 sitting at `f07c24d` read
  no_memory 6.2 / naive_rag 74.6 / mnimi 84.4 [81.0, 87.3] / mnimi+decay 78.2 /
  oracle 91.8. The 85 criterion is NOT met (441 needed, 422 read). The primary is
  significant for the first time: mnimi vs naive_rag b=75, c=26, p = 1.1 × 10⁻⁶,
  +25 on temporal-reasoning and +12 on multi-session. Decay's ablation on the
  headline run: X = −6.2, p = 2 × 10⁻⁴. Criterion (a) is met — SPEC-complete modulo
  two disclosed render deviations — so the programme's done-condition is satisfied on
  (a) and not on (b). The remaining headroom is 54 questions oracle answers and mnimi
  does not, 27 of them multi-session (DECISIONS "Gate 5-iv read", "Phase 5 closes").
  Phase 6 closed 2026-09-24 for $12.23 (v2.12.0): three levers at the 54, two adopted by the
  pre-registered rule (L1 the time-aware term, L3 the `turns` unit; L2 the reranker failed its
  probe), the headline L1 + L3 reads **426/500 (85.2 %), b=18, c=14 against the published 422**
  — inside the instrument's band, not significant, and **the 85 criterion is still NOT met**
  (lower bound 81.8). 28 of the 54 are wrong in every arm; the extractor is the measured
  bottleneck and its LoRA/prompt trigger is met with the dev-corpus condition (D9); what
  follows is in DECISIONS "Phase 6 closes".**

## Scope rule

- **In scope (library):** write-side policy — extraction, consolidation, conflict
  resolution, decay, salience, retrieval. This is the product.
- **Out of scope:** UI, rendering a chat frontend, a docs site. The library
  produces context strings; what an application does with them is someone else's
  problem. (The one context format the library *does* own is
  `RENDER_TEMPLATE` — cross-arm comparability requires exactly one renderer.)
- Anything deferred lives in `docs/FUTURE.md` with its re-open trigger. Don't
  build it early because it's tempting — hybrid FTS5, cross-encoder rerank,
  time-aware query expansion, JSON context presentation, `pinned` records,
  `access_count` all have triggers.

## Convention

Every change is validated by whether it moves the benchmark number. New code that
does not change a baseline, a system, or the score is suspect.

Phase A = v0.1, Phase B = v0.2, the v1 library build = v0.3.x, Phase C = v0.4.0,
pins schema/2 = v0.5.0; the v1 milestone then reset the counter, and Phase D
closes at v1.3.0. `pyproject.toml` is the version of record and SPEC's §"v1 as
built" header tracks it — bump both in the phase-closing commit. There is no
`CHANGELOG.md`; the commit log and `docs/DECISIONS.md` carry that history.

```bash
python -m evals --system <name> --limit 100                  # predict + judge
python -m evals --system <name> --limit 100 --stage predict  # needs Ollama, no key
python -m evals --system <name> --limit 100 --stage judge    # needs OPENAI_API_KEY
python -m evals --stage judge --predictions <file>           # Tier 1 audit, read-only
python -m evals.stats runs/a__100q runs/b__100q [...]        # paired McNemar + Holm
# gpt-4o family, through the Batch API (half price; resumable — the same
# command in the same --run-dir polls the submitted batch, never resubmits).
# A run is planned as sub-batches under the org's 90k enqueued-token cap
# (`--batch-enqueued-tokens`), submitted one at a time; one arm in flight per key:
python -m evals --system mnimi --limit 100 --stage all --reader-transport openai --batch
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --batch-no-wait
python -m evals.pricing                                     # the API spend ledger vs the cap ($83.85 = $33.85 spent + the $50 top-up of 2026-09-22)
python -m evals.drift runs/a__100q runs/b__100q             # N/n changed between two predict runs of one configuration
# the run-documentation rule (2026-09-22): every run names its purpose; a claim names its committed rule
python -m evals --system mnimi --limit 500 --stage all --reader-transport openai --batch --purpose "..." --claim none --run-dir runs/<run_id>
python -m evals.stats results/published/naive_rag__500q_gpt4o results/published/mnimi__500q_gpt4o   # pairs saved under analyses/
python -m evals.publish runs/<run_id>                       # promote: clean tree + complete manifest, or it refuses
python -m evals.backfill runs/<run_id> --purpose "..."     # a manifest for a run that predates the rule; UNKNOWN where nothing is recorded
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --verify-drift runs/mnimi__100q_gpt4o --run-dir runs/mnimi__100q_gpt4o_drift
# the presentation pair (DECISIONS 2026-09-12): same arm, the other framing, its own run dir
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --render-format json --run-dir runs/mnimi__100q_gpt4o_json
# the extraction era (PHASE2, 2026-09-13). CUDA\v13.2\bin\x64 on PATH; the probe on the
# extraction arm IS the corpus pass (the cache fills as it walks) and gate 4-i's instrument:
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --out runs/probe_mnimi_extract.json
python -m evals.probes.extractor_bench --rounds 50            # s/round + projected corpus hours
python -m evals.probes.extractor_bench --rounds 50 --stability  # 2 fresh processes, byte diff
python -m evals.probes.extractor_bench --dev-set 40           # prompt dev rounds, outside the slice
python -m evals.probes.prefilter_rate --limit 100             # stage-1 drop rate; false drops = 0
# the gate 4-iii sitting: four arms at one clean commit, one batch arm in flight at a time
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor none  --render-unit turns       --run-dir runs/mnimi__100q_gpt4o_p2base --verify-drift results/published/mnimi__100q_gpt4o_k10
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor qwen3 --render-unit round+facts --run-dir runs/mnimi__100q_gpt4o_extract
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor qwen3 --render-unit facts       --run-dir runs/mnimi__100q_gpt4o_extract_facts
# Phase 3 (PHASE3, 2026-09-15/16; $0). The two knobs are pins (schema /9); `--conflict-resolution off`
# is the v1.9 write path. The probe replays the cache (`misses: 0` or stop); identity says whether rows moved:
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --out runs/probe_mnimi_p3s.json
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --conflict-resolution off --dedup-entropy-gate 2.0 --out runs/probe_off.json
python -m evals.probes.identity runs/probe_mnimi_p3.json runs/probe_mnimi_p3s.json   # identical N/100 on ranks, top-50, drops, stored
python -m evals.probes.aggregate runs/probe_mnimi_p3s.json                            # keeps by screen; conflicts and supersessions by rule
# gate 3-ii: the seeded demo set through ScriptedExtractor, mnimi vs the v1.9 path; the exit code IS the gate:
python -m evals.probes.conflict_demo --seed 0 --out runs/conflict_demo.json
python -m evals.probes.conflict_demo --seed 0 --embedder bge --extractor qwen3 --out runs/conflict_demo_qwen3.json  # descriptive, never the gate
# Phase 4 (PHASE4, 2026-09-17/18). Seven read-path flags, pins schema /10; the probes are $0
# and replay the cache (`misses: 0` or stop). gate 4-i (identity under the landing defaults):
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --out runs/probe_mnimi_p4i.json
python -m evals.probes.retrieval --system mnimi --extractor none --limit 100 --ranking score --active-only on --out runs/probe_mnimi_p4ibase.json
# gate 4-ii (the salience-0 exclusion priced) and the two reported probes:
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --active-only on --ranking similarity --out runs/probe_mnimi_p4x.json
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --active-only on --ranking score --out runs/probe_mnimi_p4s.json
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --active-only on --ranking score --consolidate on --out runs/probe_mnimi_p4d.json
# gate 4-iii, the three-arm ablation sitting (one arm in flight at a time, never chained):
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor qwen3 --render-unit round+facts --active-only off --ranking similarity --consolidate off --run-dir runs/mnimi__100q_gpt4o_p4base --verify-drift results/published/mnimi__100q_gpt4o_extract
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor qwen3 --render-unit round+facts --active-only on --ranking score --consolidate off --run-dir runs/mnimi__100q_gpt4o_p4rank
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --extractor qwen3 --render-unit round+facts --active-only on --ranking score --consolidate on --run-dir runs/mnimi__100q_gpt4o_p4decay
# Phase 6 (PHASE6, 2026-09-22/24; probes $0, arms ≈ $4.3 each). Gate 6-i (L1 at the probe) and the two-process rerank stability pair:
PYTHONPATH="src;." python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 500 --time-weight 0.05 --out runs/probe_mnimi_p6t.json
PYTHONPATH="src;." python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 100 --ranking rerank --out runs/probe_mnimi_p6r_stab_a.json   # then _b in a fresh process, then identity
# gate 6-ii (L2 at the probe, FAIL) and 6-iii (render_unit moves nothing the probe measures — the probe has no --render-unit flag):
PYTHONPATH="src;." python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 500 --ranking rerank --out runs/probe_mnimi_p6r.json
PYTHONPATH="src;." python -m evals.probes.identity runs/probe_mnimi_p5c500.json runs/probe_mnimi_p6r.json
# gate 6-iv: three n=500 arms, one in flight at a time, each paired against the PUBLISHED arm (no --verify-drift: the pins differ in resolver_version):
PYTHONPATH="src;." python -m evals --system mnimi --limit 500 --stage predict --reader-transport openai --batch --extractor qwen3 --active-only on --ranking score --consolidate off --render-unit round+facts --time-weight 0.05 --purpose "..." --claim none --rule-commit ecd8178 --run-dir runs/mnimi__500q_gpt4o_p6time
PYTHONPATH="src;." python -m evals --system mnimi --limit 500 --stage predict --reader-transport openai --batch --extractor qwen3 --active-only on --ranking score --consolidate off --render-unit turns                    --purpose "..." --claim none --rule-commit ecd8178 --run-dir runs/mnimi__500q_gpt4o_p6turns
PYTHONPATH="src;." python -m evals --system mnimi --limit 500 --stage predict --reader-transport openai --batch --extractor qwen3 --active-only on --ranking score --consolidate off --render-unit turns --time-weight 0.05 --purpose "..." --claim none --rule-commit ecd8178 --run-dir runs/mnimi__500q_gpt4o_p6combo
PYTHONPATH="src;." python -m evals --stage judge --run-dir runs/mnimi__500q_gpt4o_p6combo --purpose "..." --claim none --rule-commit ecd8178
PYTHONPATH="src;." python -m evals.stats results/published/mnimi__500q_gpt4o results/published/mnimi__500q_gpt4o_p6combo   # the pair of record; schema and commit are REPORTED, not refused (v2.9.0)
PYTHONPATH="src;." python -m evals.publish runs/mnimi__500q_gpt4o_p6combo
```

## Current state vs SPEC (as of v2.12.0, 2026-09-24; library = the extraction era, Phase 3's screens and supersede, Phase 4's decay, ranking and read side, Phase 5's `export()` and concurrency, Phase 6's time-aware term, resolver v2 and the `turns` unit — SPEC-complete modulo three disclosed render deviations; the n=500 headline: 426/500 = 85.2 % on L1 + L3, b=18 c=14 against the published 422, the 85 criterion not met; the primary against naive_rag significant since Phase 5)

SPEC describes the target; much of it is still not built. Don't assume a spec'd
field exists — **read SPEC §"v1 as built" first**, then the code. It is
the shipped-state section; the per-section `**v1 as built:**` notes mark every
place code and target diverge, and `docs/DECISIONS.md` § "v1 build: PLANNED vs
ACTUAL" says why.

**Shipped:**

- `MemoryConfig` — ten fields: `dedup_cosine_threshold = 0.95`,
  `top_k = 10`, `dedup_scope = "session"` (R3, adopted 2026-09-12 — `"store"`
  is the published local-family configuration), `query_instruction =
  BGE_QUERY_INSTRUCTION` (R5, adopted 2026-09-13 — `""` is the local-family
  configuration), `chunk_tokens = 0` / `chunk_overlap = 64` (R4; `k` counts
  rounds, not windows — `Store.search_rounds`; measured worse, left off),
  `render_format = "text"` (v1.6.0), `render_unit = "round+facts"` (PHASE2 D5,
  adopted 2026-09-15: 84 vs `turns` 80, b=7, c=3; `facts` alone 78; `turns`
  is v1's unit and what every other arm renders), `dedup_entropy_gate = 2.0`
  (Phase 3: SPEC's field at SPEC's default, untuned — gates the cosine merge of
  facts after the two screens abstain) and `conflict_resolution = True` (Phase
  3 D11: the one switch for the screens, the gate and supersession; `False` is
  the v1.9 write path). Every
  one is a harness flag and a pin (schema /9); the chunk knobs are also
  `memory_meta` keys. A knob no code reads is not present; the rest land with
  the stage that uses them.
- The real ONNX BGE embedder behind the `[embed]` extra, revision-pinned;
  `HashingEmbedder` (256-dim, numpy-only) stays the default and the CI path.
- The `memory_meta` guard — sixteen rows (the four embedder rows, the R4 chunk
  pair, five extractor rows or `"none"`, `negation_lexicon_hash` live,
  `conflict_rules_hash`, `prefilter_lexicon_hash`, `fact_embed_template_hash`,
  `resolver_version`), written at creation and validated at every open;
  mismatch raises `MemoryMetaError` before a query runs. A DB with a `memories`
  table but no `memory_meta` is refused, not upgraded; so is a v1.8 or a v1.9
  store.
- The embed/render split — `EMBED_TEMPLATE` + `embed_template_hash()` (frozen,
  in `memory_meta`) vs `RENDER_TEMPLATE` + `render_template_hash()`
  (reader-facing, in the harness pins). A test asserts the embed text stayed
  byte-identical across the split.
- `MemoryRecord.turns` — the verbatim `{"role","content"}` turns of the source
  round, persisted as JSON. Rendering only: never embedded, never in the dedup
  key. That is the property that lets the reader format change without moving a
  vector.
- `get_context` — recall → sort oldest-first on `created_at` (id tiebreak) →
  render. One `[Session date: <full ts>]` header per timestamp change, then
  `user:` / `assistant:` labelled lines, through the one shared renderer.
- `store.search` returns `(record, cosine)` with `cos = 1 − d²/2` at exactly one
  site; injected-`ts`-only inserts (`insert` raises `ValueError` on
  `created_at is None`); zero wall-clock calls anywhere in `src/mnimi/`;
  `distance_metric=L2` spelled out in the DDL and asserted by a test.
- v1 dedup — exact-normalize collapse, then one `k=1` cosine probe at
  `dedup_cosine_threshold`, per round, committing as the loop runs (so rounds
  inside one `add()` dedup against each other too).
- `add()` narrowed to `list[dict]` at per-round granularity; `recall` reads
  `config.top_k`; `pinned` dropped from the dataclass and the DDL.
- `evals/systems/` — all five arms: `no_memory`, `full_history`, `oracle`,
  `naive_rag`, `mnimi`.
- **The extraction era (PHASE2, 2026-09-13; `mnimi docs/PHASE2.md`).**
  `src/mnimi/extract/`: `protocol` (the `Extractor` contract and its five pins),
  `schema` (`FACT_SCHEMA`, prose-first field order, strict parser), `prompt`
  (`qwen3-fact-v4`, the literal chat template), `llama` (the pinned
  `QwenLlamaExtractor`, greedy-verify-rescan under the grammar), `fake`
  (`RuleExtractor`, CI's model-free stand-in), `cache` (`CachedExtractor`, one
  SQLite file per extractor configuration), `prefilter` (six rules, no question
  rule, `prefilter_lexicon_hash`), `resolver` (relative dates anchored on `ts`;
  a mention the round does not contain is dropped). `Memory(..., extractor=)`,
  the hybrid store (a round record plus `kind="fact"` records on one
  `round_key`), per-kind dedup screens, `FACT_EMBED_TEMPLATE`, three render
  units, pins schema /8 (`--extractor`, `--render-unit`, `--extractor-cache`),
  the probe walking pieces by kind with the stage's counters,
  `evals/probes/{extractor_bench,prefilter_rate}.py`. Measured: 2.7–3.4 s per
  round on the RTX 1000 Ada (gate 2.1), 50/50 byte-stable (gate 2.2), 0 false
  pre-filter drops (gate 2.3), no-extractor probe identical to Phase 1's
  100/100. Gates read 2026-09-14/15: 4-i ANY@10 93/95 and ALL@10 83/95
  (both predictions held), 4-ii 18/20 = the k10 arm's, 4-iii adopted
  (DECISIONS "Gate 4-i read", "Gate 4-ii read", "Gate 4-iii read").
- **The conflict era (PHASE3, 2026-09-15/16; `mnimi docs/PHASE3.md`, results in
  `mnimi docs/PHASE3-RESULTS.md`).** `src/mnimi/conflict/`: `lexicon` (frozen,
  `negation_lexicon_hash`), `normalize` (`normalize_triple`, `pair_key`,
  `numeric_signature`, `conflict_rules_hash`), `screens` (`screen_pair` in SPEC
  order — negation, value, `token_entropy_bits` gate; shared by the write path
  and the probe through `memory.fact_verdict`), `supersede` (`conflict_between`
  with the three rules, `beats` with the one ordering). `MemoryRecord.pair_key`
  + index; `Store.{active_facts_by_pair,facts_with_pair_key,supersede}`;
  `Memory._resolve_conflicts` after every stored fact in `add()`;
  `consolidate()` live and idempotent; `ScriptedExtractor` (model-free,
  pre-authored facts per round); pins schema /9 (`--conflict-resolution
  {on,off}`, `--dedup-entropy-gate`); `evals/probes/{identity,conflict_demo}.py`.
  Measured, all at $0: gate 3-i PASS (ANY@10 93/95, ALL@10 83/95, 0 evidence
  lost, round drops 16/536 identical, fact/cosine 4,460 → 4,134 = 326 keeps,
  records conserved 83,788, ranks identical 99/100; one pre-registration error
  — fact/exact 392, not 379 — disclosed); the supersede probe identical to
  Task 2's on 100/100 with 46 supersessions on the slice (functional 16,
  numeric 30, negation 0; P1 not held — an extraction limit); gate 3-ii PASS
  (100 seeded pairs: mnimi 80/80 conflicts + 20/20 controls after `add()` and
  after `consolidate()` vs the v1.9 path's 0/80 + 20/20; the real-extractor
  pass, descriptive only: 42/42 of the pairs the model keyed resolved, 57/200
  rounds `[]`). Gates read in DECISIONS "Gate 3-i read", "Supersede lands",
  "Gate 3-ii read", "Phase 3 closes".
- **The read side (PHASE4, 2026-09-17/18; `mnimi docs/PHASE4.md`, results in
  `mnimi docs/PHASE4-RESULTS.md`, report in `mnimi docs/PHASE4-REPORT.md`).**
  `src/mnimi/decay.py` (`now_logical`, `logical_days`, `decayed_salience`, the frozen
  `DECAY_RULES` + `decay_rules_hash`), `src/mnimi/ranking.py` (`recency`,
  `combined_score`, `check_weights`, `score_hit`, `rank_rounds` — the exact top-k),
  `ScoredRecord`, `MemoryRecord.{last_accessed, initial_salience}`, the salience range
  check at insert, `Store.{created_ats, active_records, set_saliences, touch}` and the
  `active_only` keyword through `_knn` / `search` / `search_rounds` / `facts_of`,
  `Memory.{_rank, _now_logical, _decay, _resolve_all_conflicts, decay_stats}`, six new
  `MemoryConfig` fields, `evals/knobs.py` (seven flags, one definition), pins schema
  /10, `MnimiSystem(consolidate=)` with `consolidate_if_wired()`, and the probe's
  carriers / stale-fact / decay counters plus `identity.evidence_moves`. Measured, all
  on the slice at $0 except the sitting: gate 4-i PASS (100/100 identical to Phase 3's
  probes, 46 supersessions, `misses: 0`), gate 4-ii PASS (ANY@10 93/95, ALL@10 83/95,
  **0** evidence rounds out of the top-10; the 46 supersessions classified by hand —
  28–43 of them false positives of the exact-match rules), the score probe unchanged at
  93/83 and the decay probe down at 89/75; gate 4-iii (the sitting, $2.56) 86 / 87 / 81
  with decay at −6 points. Tests 348 → 392.

- **Close-out (PHASE5 Tasks 9–10, 2026-09-18; `mnimi docs/PHASE5.md`).** `export()` — the
  fifth and last locked public method: `src/mnimi/export.py` + `Store.all_records`, read-only
  and deterministic, rounds through the one shared renderer (a test compares the bytes), a
  superseded fact shown and **marked** rather than dropped, no clock, and unreachable from
  `evals/systems/` (an import-graph test), which is why it cannot move a number. Concurrency:
  one connection with WAL, `synchronous=NORMAL`, `BUSY_TIMEOUT_SECONDS = 30.0` and
  `check_same_thread=False` (the store and the extraction cache alike), plus one
  `threading.RLock` on `Memory` through a `_serialized` decorator over `add` / `consolidate` /
  `recall`. The connection was thread-HOSTILE before this, not merely unserialized — a second
  thread raised `ProgrammingError`, so the flag and the lock are one change. L17's third log
  format went from `log.debug("kept: ...")` to `routed to conflict: {reason} on {pair}` at
  INFO. Tests 392 → 407.
- **Phase 6 (2026-09-22/24; `mnimi docs/PHASE6.md`, results in `PHASE6-RESULTS.md`, report in
  `PHASE6-REPORT.md`, the analysis in `PHASE6-ANALYSIS.md`).** `src/mnimi/temporal.py`
  (`parse_window`, `split_query`, `dated_query`, the frozen `TEMPORAL_RULES` +
  `temporal_rules_hash`), `MemoryConfig.time_weight` (default **0.05** since v2.12.0) and the
  term in `ranking.combined_score` / `rank_rounds`; `Memory._query_embedding` strips the prefix,
  `_window` parses it, `_rank_query` is the one entry for `recall` and the probe;
  `mnimi/extract/resolver.py` v2 (`RESOLVER_VERSION = "v2"`, `FUTURE_MARKERS` / `PAST_MARKERS`,
  `last weekend`); `src/mnimi/rerank.py` (`CrossEncoderReranker` behind `[embed]`,
  `OverlapReranker` for CI, `ranking="rerank"`, `rerank_pool`); `MemoryConfig.render_unit`
  default **`"turns"`** since v2.12.0; `MemorySystem.set_question_date` / `MnimiSystem.query_for`;
  pins schema /11 (`time_weight`, `temporal_rules_hash`, `reranker_model`, `reranker_revision`,
  `rerank_pool`); `evals.knobs` `--time-weight`, `--rerank-pool`, `--ranking rerank`; the probe's
  `parsed_window` / `time_matches_top10`; `evals.stats` `REPORTED_NOT_REFUSED` / `parity_notes` /
  `harness_notes`. `MemoryConfig` is eighteen fields. Tests 409 → 485. Gates: 6-i PASS, 6-ii
  FAIL, 6-iii PASS (by the allowed shortcut — the probe has no `--render-unit` flag), 6-iv three
  arms 429 / 424 / 426 against 422, all adopted by the rule, the criterion not met.

**Still absent:**

- `context_token_budget` and `raw` in the rendered block — **decided, not pending** (PHASE5
  D3/D4, 2026-09-18): SPEC's budget of 2000 would cut ~60 % of the measured context (the
  headline arm feeds 4,946 reader prompt tokens) and `raw` duplicates a span the rendered
  turns already carry verbatim. Both moved to `docs/FUTURE.md` with triggers, so criterion (a)
  is "SPEC-complete **modulo three disclosed render deviations**" since Phase 6, never bare.
- Decay in the *harness*: `consolidate()` is live and wired behind
  `--consolidate`, but `MNIMI_DEFAULT_CONSOLIDATE = False` — gate 4-iii read decay
  at −6 points, so no benchmark arm calls it unless asked. The library pass itself is
  built and tested.
- A recency weight > 0: **superseded** (Phase 6 D3) — it is decay's mechanism at read time
  (−6.2 at n=500) and the time-aware term took the slot; it stays at 0.0 with no trigger left.
  `recall_min_relevance` > 0 ships at SPEC's default (0.0) and has never been run — a FUTURE.md
  item with a trigger.
- The third render deviation (Phase 6 D11, since `render_unit="turns"` is the default): the
  rendered block carries the round's verbatim turns and its date and **no facts** — facts are
  retrieval units, not reader-facing lines. With `context_token_budget` and `raw`, "SPEC-complete
  modulo **three** disclosed render deviations". D6's L3′ (`round+dated-facts`) is the one
  follow-up, filed with its trigger fired (temporal-reasoning −5 under `turns`).
- `source` holds the round's roles (`"user+assistant"`), not the spec'd
  `conversation_id/turn_id/role` provenance pointer (deferred to Phase E).

## Do not

- Do not import eval deps from `src/mnimi/`. `llama_cpp` and `huggingface_hub`
  are imported only inside `mnimi.extract.llama`; a test asserts
  `import mnimi.extract` never loads them.
- Do not add public methods to `Memory`.
- Do not read wall-clock time anywhere in scoring, decay, or ordering.
- Do not put an LLM in the merge/conflict/decay loop, or on the read path.
- Do not edit a pinned prompt, lexicon, or `EMBED_TEMPLATE` in place under a run.
  `mnimi/conflict/lexicon.py` and `normalize.py` are frozen (hashes
  `330604b5772e…` / `7d19c48828c8…`, pinned by tests): an edit is a version
  bump, a new hash, a re-ingest and a dated DECISIONS entry — and never to make
  a gate pass.
- Do not edit `mnimi.decay.DECAY_RULES` in place (hash `d4a0bcf07330…`, pinned by a
  test): an edit is a version bump, a new hash, a re-ingest and a dated DECISIONS
  entry, never in place and never to make a gate pass.
- Do not select a dedup threshold against LongMemEval again — further threshold
  selection on this benchmark is prohibited, and the 0.95 selection evidence
  cannot be regenerated.
- Do not tune the reader prompt per question type, for any system.
- Do not commit downloaded eval data or `*.db` files (see `.gitignore`).
