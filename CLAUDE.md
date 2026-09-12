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
- **Extractor:** Qwen3-1.7B-Instruct GGUF via llama-cpp-python, grammar-constrained,
  `temperature=0`, `top_k=1`, fixed seed/threads/batch. Local, pinned, never API.
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
  extraction lands — SPEC CHANGELOG #15.
- **Ranking:** `score = (w_sim·relevance + w_rec·recency) · salience`, defaults
  `{similarity: 1.0, recency: 0.0}`. Salience is a multiplier, not a term. No
  `access_count` — inert under the eval protocol.
- **Salience 0 means superseded, and only that.** Decay clamps at `decay_floor`
  (0.15): it down-ranks, never excludes.
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
  API budget is $50 for the whole programme, `mnimi docs/PLAN.md`;
  `evals/pricing.py`). Both families: reader prompt =
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
  prefilter-lexicon hash. Written once at DB creation, all checked on load,
  mismatch raises. (At v1 the four writable keys are all written and all
  validated; the rest are extraction-era and unwritable because nothing exists
  to hash.) Editing a lexicon or prompt is a versioned migration plus re-ingest
  — never an in-place edit under a run.
- **Normalize at the boundary.** Any insert path that bypasses `embeddings.py`
  breaks ranking correctness silently.
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
  set (dirty tree) was never published.**
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
  `results/published/mnimi__100q_restart_2026-09-10/`. The 2026-07-30 n=20
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
  than a state-of-the-art accuracy play.

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
# command in the same --run-dir polls the submitted batch, never resubmits):
python -m evals --system mnimi --limit 100 --stage all --reader-transport openai --batch
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --batch-no-wait
python -m evals.pricing                                     # the API spend ledger vs the $50 cap
# the presentation pair (DECISIONS 2026-09-12): same arm, the other framing, its own run dir
python -m evals --system mnimi --limit 100 --stage predict --reader-transport openai --batch --render-format json --run-dir runs/mnimi__100q_gpt4o_json
```

## Current state vs SPEC (as of v1.6.0; library = v1.3.0 @ 658f516 + the JSON render format)

SPEC describes the target; much of it is still not built. Don't assume a spec'd
field exists — **read SPEC §"v1 as built" first**, then the code. It is
the shipped-state section; the per-section `**v1 as built:**` notes mark every
place code and target diverge, and `docs/DECISIONS.md` § "v1 build: PLANNED vs
ACTUAL" says why.

**Shipped:**

- `MemoryConfig` — three fields: `dedup_cosine_threshold = 0.95`,
  `top_k = 10`, `render_format = "text"` (v1.6.0; `"json"` is the other
  value, read by `get_context`). A knob no code reads is not present; the
  rest land with the stage that uses them.
- The real ONNX BGE embedder behind the `[embed]` extra, revision-pinned;
  `HashingEmbedder` (256-dim, numpy-only) stays the default and the CI path.
- The `memory_meta` guard — 4 keys (`embedder_name`, `embedder_revision`,
  `embedder_dim`, `embed_template_hash`), written at creation and validated at
  every open; mismatch raises `MemoryMetaError` before a query runs. A DB with a
  `memories` table but no `memory_meta` is refused, not upgraded. **The
  `embed_template_hash` hole is closed** (v1.3.0) — there are no guard holes
  left in v1.
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

**Still absent:**

- `export()` — the public surface is 4 of the 5 locked methods.
- `ScoredRecord` — `recall()` returns `list[MemoryRecord]` and drops the cosine
  at the facade; the score is available one layer down.
- `raw`, the `(subject, predicate, object)` triple, `valid_time`,
  `last_accessed`. `created_at` carries `ts` and plays the `system_time` role.
- Extraction — no LLM anywhere in the library.
- Conflict / supersede / decay. `salience` and `supersedes` are written, stored
  and returned, and **read by nothing** — placed, not live.
- The ranking layer. Order is raw vec0 L2 ascending, which for unit vectors is
  exactly cosine-descending — so v1 *collapses to* the spec'd default
  (`similarity 1.0`, `recency 0.0`, salience uniformly 1.0) by construction
  rather than by computing it. Weights and the salience multiplier land with
  decay.
- `context_token_budget`, `raw` in the rendered block, the salience-0 exclusion,
  the active-record filter, `recall_min_relevance`.
- `consolidate()` is a no-op stub — and `systems/mnimi.py` deliberately does not
  call it, so the day it grows behaviour the benchmark cannot change without an
  explicit edit in `evals/`.
- `source` holds the round's roles (`"user+assistant"`), not the spec'd
  `conversation_id/turn_id/role` provenance pointer (deferred to Phase E).

## Do not

- Do not import eval deps from `src/mnimi/`.
- Do not add public methods to `Memory`.
- Do not read wall-clock time anywhere in scoring, decay, or ordering.
- Do not put an LLM in the merge/conflict/decay loop, or on the read path.
- Do not edit a pinned prompt, lexicon, or `EMBED_TEMPLATE` in place under a run.
- Do not select a dedup threshold against LongMemEval again — further threshold
  selection on this benchmark is prohibited, and the 0.95 selection evidence
  cannot be regenerated.
- Do not tune the reader prompt per question type, for any system.
- Do not commit downloaded eval data or `*.db` files (see `.gitignore`).
