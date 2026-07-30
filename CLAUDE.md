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
- **Extractor:** Qwen3-1.7B-Instruct GGUF via llama-cpp-python, grammar-constrained,
  `temperature=0`, `top_k=1`, fixed seed/threads/batch. Local, pinned, never API.
- **Embedder:** `BAAI/bge-small-en-v1.5` @ 384-dim, HF revision pinned. Vectors
  unit-normalized inside `embeddings.py`; vec0 uses `distance_metric=L2`.
  Embedded string, scoped by era: extraction-era = `f"{raw}\n{content}"`; v1
  (pre-extraction — no `raw` exists yet) = bare `content`, role as metadata
  only, never in the vector (a role prefix drags all similarities together and
  degrades the cosine dedup threshold). Reopens when extraction lands — SPEC
  CHANGELOG #15.
- **Ranking:** `score = (w_sim·relevance + w_rec·recency) · salience`, defaults
  `{similarity: 1.0, recency: 0.0}`. Salience is a multiplier, not a term. No
  `access_count` — inert under the eval protocol.
- **Salience 0 means superseded, and only that.** Decay clamps at `decay_floor`
  (0.15): it down-ranks, never excludes.
- **Benchmark = LongMemEval** (`longmemeval_s`, ~500 questions). The harness in
  `evals/` is the source of truth.
- **Baselines = no-memory (floor), full-history (truncated-context baseline —
  NOT a ceiling: at 32K it sees ~23% of each history), oracle (THE ceiling —
  annotated evidence sessions only, the paper's own choice), naive round-RAG
  (the bar).** mnimi must beat naive-RAG and approach oracle at a fraction of
  full-history's tokens. Competitor runs (OMEGA first) go through the same
  harness.
- **Harness pins:** reader = local Ollama `qwen2.5:1.5b-instruct-q4_0`
  (`num_gpu=99`, `num_batch=512`, `num_thread=8`, `top_k=1`, `seed=0`, temp 0,
  `num_ctx=32768`) **plus two daemon-level pins that cannot be sent per request**
  (`OLLAMA_FLASH_ATTENTION=1`, `LLAMA_ARG_CACHE_RAM=0`); judge =
  `gpt-4o-2024-08-06` with the paper's five per-type prompts. Reader is pinned in
  `evals/`, never in `memory_meta` — it is a harness property, not a store one.
- **One reader prompt for every question type**, for every system under test.
  Per-category answer-prompt tuning is prohibited; it is a comparability leak.

## Invariants that silently invalidate a number

Break one of these and the benchmark still runs — it just stops meaning anything.

- **`memory_meta` guard.** Embedder name/revision/dim, embed-template hash,
  extractor model/quant/runtime/decode-hash/prompt-hash, negation-lexicon hash,
  prefilter-lexicon hash. Written once at DB creation, all checked on load,
  mismatch raises. Editing a lexicon or prompt is a versioned migration plus
  re-ingest — never an in-place edit under a run.
- **Normalize at the boundary.** Any insert path that bypasses `embeddings.py`
  breaks ranking correctness silently.
- **Config, not constants.** Every threshold reads from `MemoryConfig`. No
  hardcoded threshold, no hardcoded `k=5` in write- or read-path logic.
- **Ingestion granularity is per-round**, and `naive_rag` must match mnimi's
  exactly. `k` is identical across all systems.
- **Sampling is stratified.** The dataset is category-clustered, so a file-order
  `--limit` slice is single-category and not comparable across systems.
- **Run artifacts.** Every run writes `pins.json` / `predictions.jsonl` /
  `results.json` under `runs/<system>__<Nq>/`. `runs/` is gitignored scratch; a
  number that gets quoted has its three files copied to
  `results/published/<system>__<Nq>/` and committed. Deviating from a decode pin
  marks the artifact **PROVISIONAL — not publishable**; don't quote a provisional
  number. Provisional and auditable are separate claims — a provisional artifact
  is still fully auditable.
- **`question` + `answer` stay inline in `predictions.jsonl`.** That is the only
  reason Tier 1 exists; removing them to denormalize deletes the audit path.
- **The daemon precondition is part of the run.** Launch Ollama manually with
  `OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0`, with the tray app not
  serving. Preflight asserts the daemon's *resolved* values and refuses
  otherwise. Kill `llama-server` children too — they outlive the daemon and hold
  VRAM.

## Reproducibility tiers (what a number is allowed to claim)

- **Tier 1 — auditable.** `python -m evals --stage judge --predictions <file>`
  re-grades committed predictions on any machine: no GPU, no Ollama, no dataset,
  no `--system`. Read-only. Proves scoring, not generation. Say *auditable*,
  never *reproducible* — this is a language rule, not a preference.
- **Tier 2 — reproducible given the pins AND the daemon precondition.**
  `--stage predict` reproduces predictions on comparable hardware. Measured error
  bar across a daemon restart: **0/20 predictions changed** on both published
  systems. No bit-identity claim across different GPUs, drivers, CUDA versions or
  Ollama builds. The `run.environment` block is diagnostic only and never enters
  `pins_hash`; `pins_hash` identifies what was *requested*, the run block what
  was *resolved*.
- **Tier 3** (containerized reference environment) is queued, not forced — see
  `docs/FUTURE.md`.
- **The verdict cache is the drift detector.** An unchanged re-run that misses
  the cache means the reader moved. Check `run.stage` before quoting any
  reproducibility number: only `stage='all'` or a fresh `predict` re-runs the
  reader.

## Kill gates

- **W3 (Jul 17, 2026):** the read/write loop must beat the naive baselines by a
  repeatable margin. If it does not, stop and rethink the approach. The W3
  artifact is all five systems (no_memory, full_history, oracle, naive_rag,
  mnimi) on one stratified slice at n≥100, same reader, same prompt, one
  sitting. n=20 is a smoke test only — it proves the loop closes; smoke numbers
  never enter a published table (categories sit at n=3-4, one question is 5
  points). Cost check: full_history ≈ 3.2 min/20q on GPU, so n=100 ≈ 16 min.
- **W6 (Aug 7, 2026):** either a competitive LongMemEval number, or reframe the
  project as a zero-infra usability play (the "one file, no server" pitch) rather
  than a state-of-the-art accuracy play.

## Scope rule

- **In scope (library):** write-side policy — extraction, consolidation, conflict
  resolution, decay, salience, retrieval. This is the product.
- **Out of scope:** UI, rendering, chat frontends, a docs site. The library
  produces context strings; what renders them is someone else's problem.
- Anything deferred lives in `docs/FUTURE.md` with its re-open trigger. Don't
  build it early because it's tempting — hybrid FTS5, cross-encoder rerank,
  time-aware query expansion, `pinned` records, `access_count` all have triggers.

## Convention

Every change is validated by whether it moves the benchmark number. New code that
does not change a baseline, a system, or the score is suspect. One minor version
per harness phase (Phase A = v0.1, B = v0.2, …).

```bash
python -m evals --system <name> --limit 10                  # predict + judge
python -m evals --system <name> --limit 10 --stage predict  # needs Ollama, no key
python -m evals --system <name> --limit 10 --stage judge    # needs OPENAI_API_KEY
```

## Current state vs SPEC (as of v0.3.2)

SPEC describes the target; much of it is still not built. Don't assume a spec'd
field exists — **read SPEC §"v1 as built" first**, then the code. It is the
shipped-state section and Phase C wires against it; the per-section
`**v1 as built:**` notes mark every place code and target diverge, and
`docs/DECISIONS.md` § "v1 build: PLANNED vs ACTUAL" says why.

Shipped in v1 (v0.3.0–v0.3.2): `MemoryConfig` (two fields only —
`dedup_cosine_threshold`, `top_k`), the real ONNX BGE embedder behind the
`[embed]` extra (revision-pinned; `HashingEmbedder` stays the default and the
CI path), the `memory_meta` guard (3 keys, raises at open), cosine surfaced
from `store.search`, `pinned` dropped, injected-`ts`-only inserts, v1 dedup
(exact-normalize + one cosine probe), `recall` reading `config.top_k`, and
`add()` narrowed to `list[dict]` at per-round granularity.

Still absent: `export()` (the public surface is 4 of the 5 locked methods),
`ScoredRecord`, `raw` + triple fields, `valid_time` (`created_at` carries `ts`
and plays the `system_time` role), extraction, conflict, decay, ranking, the
`get_context` block format, and `embed_template_hash` in the guard —
that last one is a documented invariant with no enforcement, so treat it as a
live hole, not a gap. `salience`/`supersedes` exist and are read by nothing.
`consolidate()` is a no-op stub. `systems/` has only `no_memory` and
`full_history` — `naive_rag`, `oracle`, and `mnimi` ship in one batch.

## Do not

- Do not import eval deps from `src/mnimi/`.
- Do not add public methods to `Memory`.
- Do not read wall-clock time anywhere in scoring, decay, or ordering.
- Do not put an LLM in the merge/conflict/decay loop, or on the read path.
- Do not edit a pinned prompt, lexicon, or embed template in place under a run.
- Do not tune the reader prompt per question type, for any system.
- Do not commit downloaded eval data or `*.db` files (see `.gitignore`).
