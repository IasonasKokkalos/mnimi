# CLAUDE.md — mnimi

Project-level instructions for working in this repo. Read before changing code.
`docs/SPEC.md` is the contract; this file is the short version plus the rules
that are easy to break silently. On any conflict, SPEC wins — and fix this file.
The history behind every rule here is `docs/DECISIONS.md`, cited by entry title.

## What this is

An embeddable, local-first agent-memory library. Thin public API, deep write-side
policy. Zero infra: one SQLite file via `sqlite-vec`, no server, no external
services. The library must import and run with only its two core deps.

## Stack

- **Language:** Python (>=3.11), src-layout (`src/mnimi/`; the export CLI is the sibling
  package `src/mnimi_cli/`, the MCP server `integrations/mcp/`, both in scope for release work).
- **Storage:** SQLite + `sqlite-vec` for vector search. One file on disk.
- **Core deps:** `sqlite-vec`, `numpy`. Nothing else. Adding a third is a
  decision, not a convenience — justify it in `docs/DECISIONS.md`.
- **Embedder extra** (`[embed]`): `onnxruntime`, `tokenizers`, `huggingface-hub` — the real
  BGE embedder. Optional so the core stays two deps; `HashingEmbedder` (numpy-only) is the
  default import path and CI. **Extractor extra** (`[extract]`): `llama-cpp-python==0.3.35`.
- **Eval-only deps** (`[eval]` extra): `huggingface-hub`, `ollama`, `openai`, `python-dotenv`.
  Never imported by the library.

## API contract (locked — keep these signatures)

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder,
                 config: MemoryConfig = MemoryConfig(), *, extractor=None): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # -> list[ScoredRecord]
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
    def export(self, user_id: str) -> str                   # human-readable dump
```

Five methods. Do not widen the surface. Depth goes behind `add` and
`consolidate`, not into new public methods. The contract also lives in the **message shape**:
each message dict is `{"role": "user"|"assistant", "content": str, "ts": ISO}`. `ts` is the
session timestamp, becomes the record's `system_time`, and is the anchor for relative-date
resolution. Both roles are in extraction scope.

## Locked decisions (the rule; the evidence is the DECISIONS entry named)

- **Language = Python. Storage = sqlite-vec** (local-first, zero-infra), never a hosted vector DB.
- **All time is logical.** `now_logical` = max `system_time` in the user's store. No library code
  path reads wall-clock for scoring, decay, recency, or ordering.
- **Exactly one LLM in the library, at extraction.** Merge / conflict / decay / both dedup
  screens stay deterministic. No LLM on the read path.
- **Extractor:** `Qwen/Qwen3-1.7B-GGUF` Q8_0 @ `90862c4b…`, llama-cpp-python 0.3.35 (CUDA on the
  development machine; the CPU profile differs in two decode keys and has its own hash),
  grammar-constrained greedy-verify-rescan, `temperature=0`, `seed=0`, prompt `qwen3-fact-v4`.
  Local, pinned, never API; injected as `Memory(..., extractor=)`; the harness replays an on-disk
  cache (DECISIONS "Extractor runtime", "Gate 4-iii read"). Every round keeps its record; facts
  are extra records on the round. The pre-escape, the sub-round split and client facts
  (`mnimi.extract.wrappers`) are on the MCP/chat path only, never the harness's.
- **Embedder:** `BAAI/bge-small-en-v1.5` @ 384-dim, revision `5c38ec7c…`; vectors unit-normalized
  inside `embeddings.py`; vec0 `distance_metric=L2`.
- **Embed/render split.** `EMBED_TEMPLATE` (frozen, hashed into `memory_meta`) builds the embedded
  string and the dedup key; `RENDER_TEMPLATE` (hashed into the harness pins) builds what a reader
  sees. Never collapse the two. Fact records embed `FACT_EMBED_TEMPLATE = "${raw}\n${fact}"`.
- **Ranking:** `score = (w_sim·relevance + w_rec·recency) · salience`, the k rounds *selected* by
  score over an exact candidate pool (`ranking="score"`, DECISIONS "Gate 4-iii read"); plus the
  time-aware term `time_weight · time_match` (0.05, `mnimi.temporal`, DECISIONS "Phase 6 closes").
  Salience is a multiplier. **Salience 0 means superseded, and only that**; decay clamps at
  `decay_floor` (0.15) and down-ranks, never excludes; `active_only=True` filters salience-0.
- **Conflict:** only fact records conflict; a round is evidence and is never superseded. Two facts
  on one `pair_key` conflict only under a rule — `negation`, `functional`, `numeric` — otherwise
  they are two facts; the lexicon and the normalization tables are frozen and hashed; one ordering
  (effective time, session date, raw `ts`, user over assistant, id — greater wins); the loser gets
  `salience = 0`. `conflict_resolution=False` is the pre-conflict write path byte for byte
  (DECISIONS "Gate 3-i read", "Gate 3-ii read").
- **Decay** lives in `consolidate()` after the conflict pass, frozen rules hashed
  (`decay_rules_hash`), idempotent through `initial_salience`; **built, measured at −6 points and
  off** in every benchmark arm (`--consolidate` only; DECISIONS "Gate 4-iii read", "Phase 5 closes").
- **Benchmark = LongMemEval** (`longmemeval_s`, 500 questions); the harness in `evals/` is the
  source of truth. **Five arms, roles marked:** `no_memory` (the floor), `full_history` (a
  truncated-context baseline, *cited* at 64.0 on the gpt-4o family, never run there), `oracle`
  (the evidence-availability bound — at or above it is not prima facie a bug), `naive_rag` (the
  strong K=V baseline, ingestion granularity identical to mnimi's), `mnimi`. Third-party systems
  go through `evals/systems/competitor.py` under the same pins, "through this harness", never a rank.
- **Two reader families, never paired against each other** (pins schema): the local family
  (Ollama `qwen2.5:1.5b-instruct-q4_0`, server build `0.32.13` pinned and checked — any other
  build is refused; the daemon precondition in SPEC) and the gpt-4o family (`gpt-4o-2024-08-06`
  reader and judge, Batch API, projected and budgeted before the first call: `API_BUDGET_USD`,
  `evals/pricing.py`, the ledger `.cache/api_ledger.jsonl`). Reader prompt `mnimi-con-v1` for
  both; judge prompt `longmemeval-paper-v3`. **One reader prompt for every question type, for
  every system** — per-category tuning is a comparability leak.
- **The judge is not deterministic on borderline rows** (6/100 flips measured): no absolute
  difference smaller than that is interpretable; paired comparisons are unaffected.

## Invariants that silently invalidate a number

Break one of these and the benchmark still runs — it just stops meaning anything.

- **`memory_meta` guard.** Embedder name/revision/dim, embed-template hashes, the chunk pair,
  extractor model/quant/runtime/decode-hash/prompt-hash (or `"none"`), negation-lexicon hash,
  conflict-rules hash, prefilter-lexicon hash, `resolver_version`, `decay_rules_hash` —
  seventeen rows written once at DB creation, all checked on load, mismatch raises. A store with
  a `memories` table but no `memory_meta`, or from an earlier era, is refused, not upgraded.
  Editing a lexicon or prompt is a versioned migration plus re-ingest — never an in-place edit
  under a run.
- **Normalize at the boundary.** Any insert path that bypasses `embeddings.py` breaks ranking
  correctness silently.
- **The screens only ever route a pair away from a merge.** They never drop a record the
  pre-conflict path kept, and rounds go through its screens byte for byte (a test compares
  stored contents).
- **Config, not constants.** Every threshold reads from `MemoryConfig`. No hardcoded threshold,
  no hardcoded `k=5` in write- or read-path logic.
- **Ingestion granularity is per-round**, and `naive_rag` must match mnimi's exactly. `k` is
  identical across all systems.
- **One renderer.** Every context-bearing arm renders through `mnimi.memory.render_turns` /
  `render_records`. Per-arm item granularity differs; the format does not. The renderer has two
  formats (`--render-format text|json`); a text run and a JSON run never share a pins hash and
  never pair. Do not flip the default on taste.
- **Sampling is stratified.** The dataset is category-clustered, so a file-order `--limit` slice
  is single-category and not comparable across systems.
- **Run documentation (DECISIONS "The run-documentation rule lands").** Every run writes
  `runs/<run_id>/manifest.json` and a row in the append-only `runs/INDEX.md` (both tracked:
  commit after every run or `freeze.check` refuses the next); `predictions.jsonl` rows carry
  `retrieved_ids` and `verdicts` (never overwritten — a re-grade is `judge_replay_N.json`);
  `--purpose` always, `--claim` needs `--rule-commit`; never overwrite a run directory. Paired
  comparisons come from rows (`python -m evals.stats`, saved under `analyses/`), never from
  summary scores; a quoted number cites its run_id; cited numbers live in their own table.
  Promotion is `python -m evals.publish <run_dir>` — clean tree and a complete manifest, or it
  refuses.
- **Run artifacts.** `runs/` is gitignored scratch; a number that gets quoted is promoted to
  `results/published/<run_id>/` and committed. A deviated pin, a dirty tree, another reader
  prompt or a file-order slice marks the artifact **PROVISIONAL — not publishable**; never quote
  a provisional number. `results/published/README.md` is the provenance record of every row.
- **`question` + `answer` stay inline in `predictions.jsonl`.** That is the only reason Tier 1
  exists; removing them deletes the audit path.
- **A run that cannot be projected cannot spend.** The API family prices the reader and the
  judge from `evals/pricing.py`, confirms the snapshot is served, projects an upper bound from
  the real request bodies, and refuses above the remaining budget. Money never enters `pins_hash`.
- **The daemon precondition is part of the run** (local family): the pinned build launched
  manually with `OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0`, preflight asserts the resolved
  values and the build, `ollama stop` before each arm; kill `llama-server` children too.
- **The freeze digest.** `evals/freeze.py` pins a digest of `src/mnimi/`; a run of record
  (`--limit ≥ 100` predict) refuses an unfrozen library, a dirty tree or an uncommitted
  `--config` unless `--allow-unfrozen` (provisional). A library edit recomputes the digest in the
  same commit with a DECISIONS entry — never an edit that makes `tests/test_paper_freeze.py` pass.

## Reproducibility tiers (what a number is allowed to claim)

- **Tier 1 — auditable.** `python -m evals --stage judge --predictions <file>` re-grades committed
  predictions anywhere. Proves scoring, not generation. Say *auditable*, never *reproducible* —
  a language rule. Measured: 17 of 4,000 arm-rows changed verdict on cold audits.
- **Tier 2 — reproducible given the pins and the daemon precondition.** Local family: 100/100
  predictions byte-identical across a restart (`evals.drift`). gpt-4o family: "score reproducible
  within 12/500 flips; text not reproducible" — never "byte-identical". A different Ollama build
  is refused, not tolerated. `run.environment` is diagnostic and never enters `pins_hash`.
- **The verdict cache is the drift detector.** Check `run.stage` before quoting a reproducibility
  number: only `stage='all'` or a fresh `predict` re-runs the reader.
- Retired figures (do not cite): the "12/20 changed, 5 points" figure and the `plain-prose-v2`-era
  restart table (`evals/stats.py` hard-fails pairings across that boundary).

## Kill gates and the programme (closed; the record is DECISIONS)

W3 met 2026-07-30; W6 superseded by `mnimi docs/PLAN.md` (done = SPEC-complete or Wilson lower
bound ≥ 85 on the gpt-4o family). Phases 0–6 closed 2026-09-12 → 09-25 (entries "Phase N closes"):
criterion (a) met modulo two disclosed render deviations, (b) **not met** (the shipped L1 reads
429/500 = 85.8 %, lower bound 82.5); the primary against `naive_rag` significant since Phase 5
(b=75, c=26, p = 1.1 × 10⁻⁶; the paper's pairing b=77, c=21). Phases 7–8 (the paper's harness,
no library change) closed 2026-10-06 at tag `paper-v1` = `21239e2`. n=20 is a smoke test only.

## Scope rule

- **In scope (library):** write-side policy — extraction, consolidation, conflict
  resolution, decay, salience, retrieval. This is the product.
- **Out of scope:** UI, rendering a chat frontend, a docs site. The library
  produces context strings; what an application does with them is someone else's
  problem. (The one context format the library *does* own is
  `RENDER_TEMPLATE` — cross-arm comparability requires exactly one renderer.)
- Anything deferred lives in `docs/FUTURE.md` with its re-open trigger. Don't
  build it early because it's tempting — hybrid FTS5, `access_count`, a recency
  weight, `context_token_budget`, `raw` in the rendered block all have triggers.

## Convention

Every change is validated by whether it moves the benchmark number. New code that
does not change a baseline, a system, or the score is suspect. `pyproject.toml` is the
version of record and SPEC's §"v1 as built" header tracks it; `uv lock` after every bump (CI's
locked job fails otherwise). There is no `CHANGELOG.md`; the commit log and DECISIONS carry history.

```bash
PYTHONPATH="src;." python -m pytest -q; python -m ruff check .   # before every push
python -m evals --config configs/<run_id>.json                 # a run of record: a committed argument list
python -m evals --system <name> --limit 500 --stage all --reader-transport openai --batch --purpose "..." --claim none --run-dir runs/<run_id>
python -m evals --stage judge --predictions <file> [--audit-out <file>]   # Tier 1 audit, read-only
python -m evals --stage judge --run-dir runs/<id> --judge-model gpt-4.1-2025-04-14 --purpose "..." --claim none --rule-commit <sha>   # a replay, never merged
python -m evals.stats results/published/a results/published/b [--judge ... --family F1 --by-category]   # pairs → analyses/
python -m evals.publish runs/<run_id>   [--replay runs/<id> judge_replay_N.json]
python -m evals.drift runs/a runs/b; python -m evals.pricing; python -m evals.paper_tables
python -m evals.probes.retrieval --system mnimi --extractor qwen3 --limit 500 --out runs/probe.json   # $0; misses: 0 or stop
python -c "from evals import freeze; print(freeze.check() or 'frozen and clean')"
```

## Current state (v3.1.1, 2026-10-15)

**Shipped.** The library of the paper is tag `paper-v1` (v2.14.0's `src/mnimi/`); the five
methods; `MemoryConfig` (eighteen fields, every one a harness flag and a pin); the seventeen-row
guard; the extraction era (`mnimi.extract`: protocol, schema, prompt, llama, cache, prefilter,
resolver **v3**, `verify`, `wrappers`); the conflict era (`mnimi.conflict`); decay, ranking and the
read side (`mnimi.decay`, `mnimi.ranking`, `ScoredRecord`); `export()` and concurrency (WAL, one
connection, one `RLock`); the time-aware term (`mnimi.temporal`); the cross-encoder reranker
(built, off). The developer release 3.1.x departs from `paper-v1` in extraction only (DECISIONS
"The developer release lifts the paper freeze", "3.1.1: the pre-escape puts the backslashes
back"); its n=500 reading is 429/500, b=9 c=9 against the paper system's 429 — a developer-release
reading, never the paper's number. Around the library: `src/mnimi_cli/` (`mnimi export`),
`examples/chat.py` (the CPU recipe), `integrations/mcp/` (`mnimi-mcp` 0.2.0), `evals/` (the
paper's harness: freeze, manifests, configs, replays, agreement, accounting, the third-party
contract, `paper_tables`), `results/published/` (every quoted row; its README is the record).

**Absent, by decision** (each with its trigger in `docs/FUTURE.md`): `context_token_budget` and
`raw` in the rendered block (the two disclosed render deviations, PHASE5 D3/D4); decay in any
benchmark arm; a recency weight > 0 (superseded by the time-aware term); `recall_min_relevance`
> 0 (never run); the spec'd `conversation_id/turn_id/role` provenance pointer (`source` holds the
roles); cross-key supersession (the MCP test week's issue 7); a `memory_meta` row for the verify
rules (`VERIFY_VERSION`, declared, not yet guarded).

## Do not

- Do not import eval deps from `src/mnimi/`. `llama_cpp` and `huggingface_hub`
  are imported only inside `mnimi.extract.llama`; a test asserts
  `import mnimi.extract` never loads them.
- Do not add public methods to `Memory`.
- Do not read wall-clock time anywhere in scoring, decay, or ordering.
- Do not put an LLM in the merge/conflict/decay loop, or on the read path.
- Do not edit a pinned prompt, lexicon, `EMBED_TEMPLATE`, `mnimi.decay.DECAY_RULES` or
  `mnimi.temporal.TEMPORAL_RULES` in place under a run. The lexicon and the normalization
  tables are frozen (hashes pinned by tests): an edit is a version bump, a new hash, a
  re-ingest and a dated DECISIONS entry — and never to make a gate pass.
- Do not select a dedup threshold against LongMemEval again — further threshold
  selection on this benchmark is prohibited, and the 0.95 selection evidence
  cannot be regenerated.
- Do not tune the reader prompt per question type, for any system.
- Do not commit downloaded eval data or `*.db` files (see `.gitignore`).
- Do not touch `src/mnimi/` from release work without a DECISIONS entry and a new freeze digest
  in the same commit. The paper's library is tag `paper-v1` and its arms are never rerun from a
  release branch; a library fix the paper needs is `paper-v1.1` with every paper arm rerun.
  Lifting or moving the freeze is never an edit that makes `tests/test_paper_freeze.py` pass.
