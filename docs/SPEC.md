# SPEC

The contract. Signatures here are locked; changing them is a breaking change.
Storage backend: SQLite + sqlite-vec for v1.

## Public API — `agentmem.Memory`

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # raw retrieval -> list[ScoredRecord]
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
    def export(self, user_id: str) -> str                   # human-readable dump
```

- `add` — ingest messages for a user. Today: naive store of message text. Later: extraction, valid_time resolution, salience, dedup.
- `recall` — raw vector retrieval. Returns records with per-hit scores attached (relevance / recency / salience exposed, not hidden). No assembly.
- `get_context` — rank + assemble retrieved records into a single context string for a reader, within a token budget.
- `consolidate` — merge duplicates, resolve conflicts, decay stale memories. Stub today.
- `export` — human-readable text/markdown dump of the store. No UI.

## `MemoryConfig`

Tunable parameters, passed at construction (`Memory(db_path, embedder, config=MemoryConfig())`).
Never hardcoded in write-path logic — every threshold below must read from here.

`dedup_cosine_threshold` (float), `dedup_entropy_gate` (float),
`decay_half_life_days` (float), `salience_weights` (dict) — exact fields finalized in W2/W4 as merge/decay land.

{ dedup_cosine_threshold: float in [0,1] — min cosine similarity for two records
 to be treated as merge candidates during dedup (stage 2, after exact-normalize).

 dedup_entropy_gate: float — min Shannon-entropy score a record's content must
 have before the cosine threshold is trusted; below this, skip auto-merge
 (content too short/low-information for a reliable similarity call).

 decay_half_life_days: float — for standing facts (valid_time = None), number
 of days since last_accessed after which salience halves; drives recency decay.

 salience_weights: dict — relative weighting of signal components (e.g. semantic
 similarity, recency, access frequency) combined into a record's retrieval rank. }

## `MemoryRecord` (storage shape — canonical)

Single source of truth for the record schema. Constrains the whole write path.

| field | type | purpose |
|---|---|---|
| `id` | str | primary key |
| `user_id` | str | owner |
| `content` | str | human-readable fact text — never embedding-only |
| `embedding` | vec0 virtual table | vector for retrieval |
| `system_time` | datetime | when extracted/written (was `created_at`) |
| `valid_time` | datetime \| None | when the fact is true in the real world; null = standing fact |
| `last_accessed` | datetime | drives recency decay for standing facts |
| `salience` | float | retrievability weight; decay lowers it, 0 = inactive |
| `source` | str | provenance |
| `supersedes` | id \| None | points to the record this one replaced |
| `pinned` | bool | decay-exempt + conflict-winning (see below) |

Superseded records are marked inactive (salience floored), not deleted — history is kept and stays exportable.

## Write path (`add` → `consolidate`)

Order of operations on the write side:

1. **Extract** — pull salient facts from messages. Produces `content` + `salience` + `valid_time`.
2. **Resolve valid_time** — relative dates ("next Tuesday", "in 3 weeks", "on the 20th") resolved to absolute datetime using the *conversation's* timestamp as anchor. Most facts resolve to `valid_time = null` (standing) — treat null as the default case, not the exception.
3. **Dedup** — see strategy below.
4. **Conflict / supersede** — a new fact contradicting an existing one (same subject, new value) sets `new.supersedes = old.id`, marks old inactive, logs the decision. For dated facts, `valid_time` ordering decides which is current. **Pinned records win conflicts** and are never auto-superseded.
5. **Decay** — branches on fact type:
   - `valid_time` set (dated/episodic) → hard step-function: salience → 0 once `now > valid_time`. Not a curve.
   - `valid_time` null (standing/semantic) → recency/access half-life decay on `now - last_accessed`.
   - `pinned` → skipped entirely.

## Dedup strategy (v1 — reconsider in later weeks)

Recommended stack, cheap and deterministic, no MinHash/LSH at this scale:

1. **Exact-normalize pre-filter** — lowercase, strip whitespace/punctuation → collapse literal duplicates. Free, zero false positives.
2. **Cosine threshold** — embedding similarity via sqlite-vec brute-force KNN against existing records; above a *tunable* threshold → merge candidate.
3. **Entropy/confidence gate** — short or low-information content (e.g. bare names) is unreliable under cosine; below an entropy threshold, don't auto-merge — keep separate (or defer to LLM if one's in the loop). Prevents wrong merges on ambiguous inputs.

The threshold being deterministic + tunable + inspectable *is* the differentiator vs competitors' LLM-prompt merges — keep it that way.

## Dedup: negation screen (locked)

**Problem:** cosine measures topical/contextual similarity, not truth value.
Contradictory facts about the same entity ("likes pizza" / "dislikes pizza")
routinely score high cosine — same topic, same entities, opposite polarity.
Undetected, this merges facts that should conflict-resolve instead, and
destroys the superseded fact before W4 conflict logic ever runs.

**Pipeline (updated):**
exact-normalize → cosine gate → negation screen → entropy gate → merge
│
└─ negation mismatch → route to
conflict resolution (W4), not merge

**Negation screen:** deterministic, no LLM in the bookkeeping loop (per
existing constraint).
- Fixed lexicon of polarity/negation markers: `not`, `no longer`, `stopped`,
  `doesn't`, `never`, `dislikes` vs `likes`, etc. — extend as false
  negatives surface, don't over-engineer up front.
- Applied only to cosine-gate-pass pairs (candidates), not the full corpus —
  cheap, O(candidates) not O(N).
- Mismatch on one side only (one record has a negation cue, the other
  doesn't, same entity/topic) → flag, route to conflict path.
- No cue on either side → proceed to entropy gate as before.

**Scope boundary:** this is a screen, not a classifier. It doesn't need to
catch every negation — it needs to stop the *obvious* cases from silently
destroying data via merge. False negatives here are a W4 conflict-resolution
problem (facts that should've been flagged but weren't will still surface
as unresolved contradictions eventually). False positives just mean
occasional unnecessary conflict-path routing — cheap, not correctness-risking.

**Explicitly not doing:** semantic/entailment-based contradiction detection,
LLM-judged polarity, NLI models. Lexicon-based screen only, until the
LongMemEval number shows this is the bottleneck.

## Read path (`recall` → `get_context`)

- Ranking signal = semantic similarity + recency + salience (not cosine alone).
- `recall` returns each hit with its component scores attached — retrieval relevance is inspectable as data, not a black box.(this should have a threshold < salience check so noin relevant info doesnt get extracted)
- `get_context` assembles the top-ranked records into a tight context string within a token budget. Inactive (salience 0) records excluded.

## Embedding normalization (locked)

**Decision:** normalize embeddings to unit length at write time, before insert.
Store the normalized vector, not the raw model output.

**Distance metric:** vec0 table uses `distance_metric=L2`, not `cosine`.
For unit vectors, L2² = 2 − 2·cos_sim → identical ranking to cosine, cheaper op
(no norm/sqrt/division at query time, matters under brute-force scan × N candidates).

**Where it lives:**
- `embeddings.py` — normalize immediately after the embed() call, before returning
- `store.py` — vec0 schema: `embedding float[D] distance_metric=L2`
- `recall()` — no change; ranking is equivalent, just cheaper

**Constraint this creates:** any code path that inserts a vector without going
through `embeddings.py`'s normalize step silently breaks ranking correctness.
Normalize once at the boundary, never assume callers did it.

## Embedding config (locked for LongMemEval baseline)

**Embedder:** single hardcoded local model for the eval run. No user-facing
config field. Pluggable choice deferred to FUTURE.md.

- Model: `<fill in>`
- Dim: `<fill in>`
- Normalization: `embed()` returns unit-normalized vectors. Normalization
  happens once, inside `embeddings.py`. No call site outside it may skip this.

**vec0 schema:**
```sql
CREATE VIRTUAL TABLE memories USING vec0(
  embedding float[D] distance_metric=L2
);
```
L2 on unit vectors ranks identically to cosine (`L2² = 2 − 2·cos_sim`) —
cheaper per-comparison, matters under brute-force scan × N candidates.

**Embedder metadata guard:**
```sql
CREATE TABLE memory_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- rows: ('embedder_name', '<model>'), ('embedder_dim', '<D>')
```
- Written once, at DB creation.
- Checked on every load: configured embedder vs. stored metadata.
- Mismatch → raise, do not proceed.

**Rationale:** reproducibility requires the embedder to be fixed for the life
of a benchmark run. Without this check, a config change or a forgotten model
choice silently invalidates the LongMemEval number — retrieval still runs,
scores still come out, comparing incompatible vector spaces. This check is
the only thing between "the number is real" and "the number is quietly wrong."

**Out of scope → FUTURE.md:** `Embedder` protocol / pluggable embedder
choice, API-backed embedders, Matryoshka dim truncation, hot-swap /
re-embed workflow.

## Concurrency

SQLite serializes writers even in WAL mode. All writes go through a single connection/queue; WAL enabled for concurrent reads. Background `consolidate()` must not race a live `add()` — same queue.

## Benchmark contract — `evals.base.MemorySystem`

Separate from `Memory`. Any system under test implements `reset()`, `add(messages)`, `get_context(query) -> str`. The harness drives it identically for every system so results are comparable.

## Human-readable memory (locked, not future scope)

- Content stored as readable text + structured metadata (schema above). Original fact always recoverable without reverse lookup.
- `export()` dumps the store as readable text/markdown. No UI.
- Every merge / decay / supersede decision logged with a reason:
  - dated: `"expired: valid_time {date} < now"`
  - standing: `"decayed: {days} since last access, salience {old} → {new}"`
  - conflict: `"superseded {old_id}: {reason}"`

**Out of scope:** dashboard, card UI, visual memory browser — integrating developer's job.

**Why:** matches the gap across all five competitors (CompetitorTeardowns.md) — none score 4+ on both write-path and inspect/reproduce. Letta's ADE/SQL-queryable blocks are the bar to match, without the runtime/server overhead.

## Open questions (undecided — leaning noted)

- **Tiered model (core/recall/archival, Letta-style)?** — Lean *no* for v1. Your differentiator is the write-path policy, not tier structure; tiering adds surface area without moving the benchmark number. Reconsider only if context-assembly proves it needs it.
- **Self-editing memory (agent edits its own store via tools, Letta-style)?** — Lean *no*. Your library manages memory *for* the agent (Mem0-style). Self-editing is a different paradigm/product. Out of scope unless a real reason surfaces.