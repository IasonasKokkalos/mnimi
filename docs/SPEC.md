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

## `MemoryRecord` (storage shape — canonical)

Single source of truth for the record schema. Constrains the whole write path.

| field | type | purpose |
|---|---|---|
| `id` | str | primary key |
| `user_id` | str | owner |
| `content` | str | human-readable fact text — never embedding-only |
| `embedding` | list[float] | vector for retrieval |
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

## Read path (`recall` → `get_context`)

- Ranking signal = semantic similarity + recency + salience (not cosine alone).
- `recall` returns each hit with its component scores attached — retrieval relevance is inspectable as data, not a black box.
- `get_context` assembles the top-ranked records into a tight context string within a token budget. Inactive (salience 0) records excluded.

## `Embedder` protocol

`embed(texts: list[str]) -> list[list[float]]` and a `dim` property.

> ⚠ FLAG: "default impl, no dependency beyond numpy" — a numpy-only embedder can't produce semantic embeddings (no model). Either ship a small model (adds a dep) or make the default an explicit test-only baseline (hash/bag-of-words) and require users to pass a real embedder for actual use. Decide before W1 storage work.

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