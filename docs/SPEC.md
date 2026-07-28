# SPEC

The contract. Signatures here are locked; changing them is a breaking change.
Storage backend: SQLite + sqlite-vec for v1.

---

## CHANGELOG (locked decisions changed, with evidence — newest first)

11. **Decay floor added.** Salience decays toward `decay_floor` (default 0.15),
    never to zero; salience 0 is reserved exclusively for superseded records.
    Decay can now only down-rank old evidence, never exclude it — closing the
    recall-gating risk (retrieval correctness gates ~90% of correct answers,
    LongMemEval Fig 14). Prior art: OMEGA v1.5.5 ships decay floors 0.15/0.35
    (`omega/sqlite_store/_base.py:226-227`, source-verified 2026-07-20).
    Benchmark effect unmeasured — covered by the existing decay-on/off
    ablation.
12. **Extraction schema field order locked: free prose before constrained
    slots.** Format constraints measurably degrade capacity-limited models,
    and the degradation mechanism is premature serialization — performance
    recovers when unconstrained generation precedes structured fields
    (arXiv 2408.02442, measured; arXiv 2606.09410, measured). Qwen3-1.7B is
    exactly the capacity class at maximum risk. `content` and `raw` generate
    before `subject/predicate/object` and times in the pinned grammar.
13. **Harness: one fixed reader prompt across all question types.**
    Per-category answer-prompt tuning is prohibited for every system under
    test. Category-tuned prompts are a disclosed practice in at least one
    competitor headline number (OMEGA 95.4%, vendor-stated) and a documented
    comparability leak (CompetitorTeardowns Appendix B).

1. **Embedding input = `raw` + `content`, not `content` alone.** Fact-only keys
   underperform raw+fact by +9.4% recall@k / +5.4% QA accuracy on average
   (LongMemEval Table 3, §5.3, arXiv 2410.10813). New `raw` field added to the
   schema; extraction output contract extended.
2. **`get_context` emits `raw` + `content` per record, time-ordered, with
   visible timestamps.** Fact-only *values* lose information at the reading
   stage except on multi-session questions (Fig 5, §5.2); the paper's own
   pipeline sorts retrieved items by timestamp for the reader (§5.1) and
   structured formatting is worth up to 10 points even at oracle retrieval
   (Fig 6, §5.5).
3. **valid_time step-function decay removed.** "salience → 0 once
   now > valid_time" zeroes every past event immediately, making
   temporal-reasoning questions (27% of the benchmark, §3.2/Fig 9a)
   structurally unanswerable — they ask about past dated events. valid_time is
   temporal metadata, not an expiry. Prospective-expiry policy → FUTURE.md.
4. **Extraction scope includes assistant turns.** single-session-assistant is
   11% of the benchmark; commercial systems score ~0 on it precisely because
   they store nothing assistant-side (LongMemEval §3.2, Appendix B).
5. **Extraction output contract extended: `raw` + nullable
   `(subject, predicate, object)` triple + `ts` passthrough.** The triple makes
   the value-substitution screen deterministic (see Dedup). This is an
   extraction-layer change, inside the one permitted LLM; the bookkeeping loop
   stays LLM-free.
6. **`pinned` removed from the schema → FUTURE.md.** No LongMemEval question
   type depends on decay-immunity of identity facts within the eval horizon;
   the field had consumers but no producer (dead behavior). Removal over
   invention.
7. **Ranking third term dropped; no `access_count`.** The eval protocol is
   reset-per-instance + one query per question (§3.1), so an access-frequency
   term is a constant zero at query time — it cannot move the number. Ranking
   = weighted similarity + recency (recency default 0.0 — the paper's winning
   config ranks by dense similarity alone, §5.1), stored salience as
   multiplier.
8. **All time is logical.** Decay and recency read `now_logical` = the latest
   session timestamp observed in the store, never wall-clock. Wall-clock in
   decay makes the same DB produce different numbers on different days —
   reproducibility failure. (Both closest prior-art systems use wall-clock
   decay — OMEGA `_base.py:209`, Hypabase `memory/strength.py:26` — which is
   the reproducibility exhibit, not a pattern to copy.) `system_time` = the
   session timestamp carried by the ingested messages (the benchmark supplies
   (t_i, S_i), §3.1), not ingest time.
9. **Reader pinned — at the harness layer, not memory_meta.** The reader is a
   benchmark-contract property, not a store property (mnimi ships without
   one). Reader model + reader prompt are hash-pinned in `evals/` config.
   Reader swings on identical retrieval are large: ~15 pts across readers in
   Table 3; 84.23% → 94.87% from a reader swap alone in Mastra OM
   (mastra.ai/research/observational-memory, vendor-reported).
10. **Reproducibility guards expanded** to cover embedder revision, extractor
    quantization + runtime version + sampler config, embed-template hash, both
    lexicon hashes. llama.cpp documents non-bit-identical logits across batch
    sizes and builds (llama.cpp server README; issues #1340, #7052); an HF
    model name without a revision is mutable.

---

## Public API — `mnimi.Memory`

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder, config: MemoryConfig = MemoryConfig()): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # raw retrieval -> list[ScoredRecord]
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
    def export(self, user_id: str)  -> str                   # human-readable dump
```

- `add` — ingest messages for a user. **Each message dict carries
  `role` (`"user"` | `"assistant"`), `content`, and `ts` (session timestamp,
  ISO).** Both roles are in extraction scope (CHANGELOG #4). `ts` becomes the
  record's `system_time` and the anchor for relative-date resolution.
  Signature unchanged — the contract lives in the message shape.
- `recall` — raw vector retrieval via the Retriever (below). Returns records
  with per-hit component scores attached. No assembly.
- `get_context` — rank + assemble retrieved records into a single context
  string within a token budget. See Read path for the locked format.
- `consolidate` — merge duplicates, resolve conflicts, decay stale memories.
- `export` — human-readable text/markdown dump of the store. No UI.

## `MemoryConfig`

Tunable parameters, passed at construction. Never hardcoded in write-path
logic — every threshold below must read from here.

Starting values — **v1 defaults, all unmeasured guesses to be tuned on W2/W4
eval evidence, not sacred:**

```python
dedup_cosine_threshold = 0.85   # merge-candidate floor. BGE model card: absolute
                                # similarity values unreliable; pick per data
                                # distribution, e.g. 0.8/0.85/0.9. 0.85 also the
                                # empirically chosen near-duplicate threshold in
                                # NovAScore (arXiv 2409.09249). Tune range 0.80-0.92.
dedup_entropy_gate     = 2.0    # bits; below this skip auto-merge. Convergent
                                # prior art: Graphiti gates fuzzy dedup on
                                # Shannon entropy (dedup_helpers.py:52-80,
                                # source-verified).
decay_half_life_days   = 30.0   # standing-fact half-life on logical time
decay_floor            = 0.15   # salience never decays below this (CHANGELOG
                                # #11). 0 is reserved for superseded records.
                                # Decay down-ranks, never excludes.
top_k                  = 10     # Table 3: Top-10 >= Top-5 for capable readers;
                                # if the pinned reader is 8B-class, revisit
                                # (weak readers degrade past ~3k retrieved
                                # tokens, LongMemEval §5.2)
context_token_budget   = 2000
recall_min_relevance   = 0.0    # relevance (query-similarity) floor. DEFAULT OFF:
                                # BGE absolute scores are unreliable and recall
                                # gates ~90% of correct answers (Fig 14). Activate
                                # only if W3 error analysis shows distractor-driven
                                # reading failures. Prior art: OMEGA ships this ON
                                # at 0.60 vec similarity (_base.py:230) — noted,
                                # not copied; their floor is untested on
                                # LongMemEval-style recall. This is a RELEVANCE
                                # floor, not a salience floor — inactive
                                # (salience 0) records are excluded separately.
salience_weights = {"similarity": 1.0, "recency": 0.0}
                                # paper-validated baseline ranks by similarity
                                # alone (§5.1). recency is a tunable knob, not a
                                # validated signal. Stored salience is a
                                # MULTIPLIER on the combined score, not a term.
```

## `MemoryRecord` (storage shape — canonical)

Single source of truth for the record schema. Constrains the whole write path.
**Every field lists its producer (P) and consumer (C).** A field with no
producer or no consumer is spec drift.

| field | type | purpose | P / C |
|---|---|---|---|
| `id` | str | primary key | P: store · C: everything |
| `user_id` | str | owner | P: `add` · C: retriever scoping |
| `content` | str | human-readable fact text — never embedding-only | P: extraction · C: get_context, export |
| `raw` | str | verbatim source span the fact was extracted from, role-prefixed (`"user: …"` / `"assistant: …"`); immutable; on merge, the surviving record's `raw` is kept — never concatenate raw spans (unbounded growth) | P: extraction · C: embed input, get_context |
| `subject` | str \| None | normalized subject of the fact triple | P: extraction · C: value-substitution screen, conflict |
| `predicate` | str \| None | normalized predicate | P: extraction · C: value-substitution screen, conflict |
| `object` | str \| None | value slot of the triple | P: extraction · C: value-substitution screen, conflict |
| `embedding` | vec0 virtual table | vector for retrieval | P: embeddings.py · C: retriever |
| `system_time` | datetime | the **session timestamp** of the source message (`ts`), not ingest wall-clock | P: `add` · C: conflict tiebreak (later wins when valid_time null/equal), get_context ordering, export |
| `valid_time` | datetime \| None | when the fact is true in the real world; null = standing fact. **Temporal metadata only — never an expiry** (CHANGELOG #3) | P: extraction + resolver · C: conflict ordering (dated facts), get_context timestamp display |
| `last_accessed` | datetime | drives recency decay | P: **`recall()` sets it to `now_logical` on every returned hit**; initialized to `system_time` at write · C: decay |
| `salience` | float | retrievability weight; decay lowers it toward `decay_floor`; 0 = superseded/inactive only | P: extraction (initial), decay (updates), conflict (floors superseded to 0) · C: ranking multiplier, active-record filter |
| `source` | str | provenance pointer: `conversation_id/turn_id/role` | P: `add` · C: export provenance line; conflict tiebreak when valid_time and system_time both tie |
| `supersedes` | id \| None | points to the record this one replaced | P: conflict resolution · C: export, audit |

Superseded records are marked inactive (salience set to 0), not deleted —
history is kept and stays exportable. Decay never produces 0 (floor,
CHANGELOG #11); only supersession does. `pinned` removed (CHANGELOG #6).

### `ScoredRecord` (read-side)

A `MemoryRecord` plus attached retrieval scores. Return type of `recall`.

| field | type | purpose |
|---|---|---|
| `record` | MemoryRecord | the retrieved record |
| `relevance` | float | query–record similarity (post-normalize) |
| `recency` | float | recency component (from `now_logical − system_time`) |
| `salience` | float | current stored salience |
| `score` | float | combined rank (see Ranking) |

## Logical time (locked)

All decay and recency computations use `now_logical` = the maximum
`system_time` present in the user's store. No code path in the library reads
wall-clock time for scoring or decay. Rationale: wall-clock `now` makes the
same DB score differently depending on the day the eval runs — a silent
reproducibility failure the guards can't catch. Both closest prior-art
systems (OMEGA, Hypabase) decay on wall-clock days-since-access — their
numbers are not replayable; mnimi's are. (CHANGELOG #8.)

## Extraction (locked)

Write-path step 1. Turns a raw message stream into fact candidates that enter
the deterministic pipeline. **This is the only place an LLM is permitted in
mnimi.**

**This stage is the primary prior-art delta.** The closest in-scope
competitor (OMEGA v1.5.5, source-verified) has no raw-conversation ingestion:
its write path assumes agent-curated typed records, and its "fact extraction"
is regex over already-distilled content (`bridge.py:652`, CamelCase /
backticks / decision-verbs — coding-shaped). Turning raw chat into memory is
mnimi's product, not a feature.

**Scope: both user and assistant turns** (CHANGELOG #4). Assistant-side facts
(recommendations, commitments, answers given) are first-class records with
`role` recorded in `raw` and `source`. single-session-assistant is 11% of
LongMemEval and is unanswerable from a user-only store (Appendix B).

**Two stages: deterministic pre-filter → LLM extraction on survivors.**

```
messages (role, content, ts)
│
├─ stage 1: speech-act pre-filter (deterministic, no LLM)
│     drop → logged, never reaches the model
│
└─ stage 2: LLM extraction (schema-constrained, pinned, temp 0)
      returns [] → nothing stored
      returns facts → each becomes a MemoryRecord candidate → dedup
```

### Stage 1 — speech-act pre-filter (deterministic)

Rejects turns that cannot contain a stored fact, before any LLM call.

Drop rules (a turn is dropped if it matches, no fact-shaped content survives):
- imperative directed at the assistant ("save that", "remember this", "do X")
- self-referential to the memory system ("that statement you made", "your last answer")
- question (trailing `?`, wh-word lead, aux-inversion) — user turns only;
  assistant turns are not questions to the store
- no candidate slot at all (no entity, no preference/event/relation cue)
- assistant-turn boilerplate (greetings, offers of help, closing pleasantries)

- Deterministic lexicon + shallow pattern rules. No LLM. Extend as false
  drops surface — **but every lexicon edit changes the corpus and therefore
  the number: the lexicon is hashed into the guard (below), and an edit is a
  versioned migration, not a hot fix.**
- Every drop logged with the rule that fired: `"filtered: {rule}"`.
- Scope: a screen, not a classifier. False negatives (junk that slips
  through) are caught by stage 2 returning `[]`. False positives (a real fact
  dropped) are the only correctness risk — keep rules conservative.

### Stage 2 — LLM extraction (schema-constrained)

Only turns surviving stage 1 reach the model. The model returns atomic facts
or an empty list; it never decides merge/conflict/decay — those stay
deterministic.

**Output contract** — each extracted fact maps to write-path step 1 fields.
**Field order in the grammar is locked: free-prose fields first, constrained
slots last** (CHANGELOG #12):
```json
[
  { "content":   "<human-readable fact text>",
    "raw":       "<verbatim source span this fact was extracted from>",
    "subject":   "<normalized subject | null>",
    "predicate": "<normalized predicate | null>",
    "object":    "<value slot | null>",
    "salience":  <float>,
    "valid_time": "<ISO datetime | null>" }
]
```
Rationale for the order: format constraints measurably degrade
reasoning-shaped generation, and the mechanism is premature serialization —
recovery comes from letting unconstrained generation precede structured
fields (arXiv 2408.02442; arXiv 2606.09410). Capacity-limited models take
the largest penalty, and Qwen3-1.7B is exactly that class. Generating
`content`/`raw` prose first lets the model commit its reading of the turn
before filling the exact triple slots. The order is part of the pinned
grammar and therefore part of `extractor_prompt_hash`'s covered surface.

Empty list (`[]`) when the turn carries no fact. The triple is nullable as a
unit: if the model cannot produce a clean (subject, predicate, object), all
three are null and the downstream value-substitution screen abstains
(conservative degradation, see Dedup).

**Decoding is grammar-constrained.** The output schema is enforced via
llama.cpp GBNF / llama-cpp-python's JSON-schema-to-grammar converter, so
structurally invalid output is impossible at the sampler level (llama.cpp
grammars/README). The grammar constrains but does not inform — the schema
must also be described in the pinned prompt, since llama.cpp does not inject
it. Known caveat: grammars cannot prevent truncation at the token limit —
set `max_tokens` with headroom and treat a truncated parse as `[]` for that
turn, logged. Constrained decoding on extraction-shaped (classification-like)
tasks is the measured-safe case (arXiv 2408.02442: structured formats help
classification); the reasoning-shaped risk is handled by field order above.

**Model:** single hardcoded **local** model, deterministic decode, pinned.
No API-backed extractor for the benchmark — an API model can change under the
run and silently invalidate the number.

- Extractor model: Qwen3 1.7B-Instruct GGUF via llama-cpp-python
- Decode config (pinned as part of the guard): `temperature=0`, **`top_k=1`**
  (greedy is not guaranteed by temp=0 alone), fixed `seed`, fixed
  `n_threads`, fixed `n_batch`, CPU-only. llama.cpp logits are not
  bit-identical across batch sizes/builds (server README; issues #1340,
  #7052) — the config is part of the reproducibility surface.
- Prompt: pinned, versioned by hash (see guard below)
- **Known load-bearing risk, stated:** zero-shot triple extraction quality at
  1.7B is unmeasured, and measured results show a clear size-performance
  tradeoff at 1–3B with off-the-shelf triple extractors performing poorly
  untuned (clinical distillation study, PMC12065832; REBEL fine-tune,
  arXiv 2507.13827). The measured lever if W3 shows extraction is the
  bottleneck: LoRA fine-tune on teacher-generated data (see Deferred).

**The "no LLM in the bookkeeping loop" constraint is unchanged.** It scopes
merge / conflict / decay (dedup pipeline, both screens, decay). Extraction is
a separate, upstream loop. Emitting a structured triple here does not move
any decision into the LLM — the record it emits is what enters the
deterministic pipeline, and everything downstream stays exactly as spec'd.
External corroboration for keeping the write-loop LLM-free: Mem0's LLM-routed
ADD/UPDATE/DELETE was measured emitting malformed routing output ~25% of the
time in an independent study (arXiv 2606.15903), and Mem0's own README now
declares a retreat to single-pass ADD-only accumulation while the router
code remains in-tree (README:57 vs prompts.py:176-444, commit 726bcc8,
source-verified 2026-07-20).

### Reproducibility guard (consolidated)

A pinned artifact whose identity is persisted and validated, so a silent
change fails loudly instead of quietly invalidating a benchmark run.

```sql
CREATE TABLE memory_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- rows, written once at DB creation, ALL checked on every load,
-- any mismatch → raise, do not proceed:
--   ('embedder_name',          '<model>')
--   ('embedder_revision',      '<HF commit sha>')
--   ('embedder_dim',           '<D>')
--   ('embed_template_hash',    '<sha256 of the raw+content template>')
--   ('extractor_model',        '<model>')
--   ('extractor_quant',        '<gguf quantization tag, e.g. Q8_0>')
--   ('extractor_runtime',      '<llama-cpp-python version>')
--   ('extractor_decode_hash',  '<sha256 of the pinned decode config>')
--   ('extractor_prompt_hash',  '<sha256 of the pinned prompt + grammar>')
--   ('negation_lexicon_hash',  '<sha256 of the sorted lexicon>')
--   ('prefilter_lexicon_hash', '<sha256 of the sorted stage-1 lexicon>')
```

- Written once, at DB creation. Checked on every load. Mismatch → raise.
- Consequence, stated plainly: **editing a lexicon or prompt invalidates
  existing DBs for comparability.** That is the point. A lexicon edit is a
  versioned release + re-ingest, never an in-place change under a run.
- The reader is deliberately NOT here — it is pinned in the benchmark
  contract (below), because it is a harness property, not a store property.

**Rationale:** reproducibility requires every artifact that determines the
corpus or the vectors to be fixed for the life of a benchmark run. Quant tag
and runtime version are included because Q4 vs Q8 changes logits and temp-0
decode is not bit-identical across llama.cpp builds. The HF revision is
included because a model name without a commit sha is mutable.

### Out of scope → FUTURE.md

Pluggable / API-backed extractors, per-record-type extraction schemas,
multi-turn coreference windows beyond the single conversation, deterministic
(spaCy/rule) extraction fallback, prospective-fact expiry policy, `pinned`
records. Locked choice for v1: deterministic pre-filter + pinned local LLM.
Re-litigate only if the LongMemEval number shows extraction is the
bottleneck.

## Write path (`add` → `consolidate`)

Order of operations on the write side:

1. **Extract** — pull salient facts from messages (both roles). Produces
   `content` + `raw` + triple + `salience` + `valid_time`.
2. **Resolve valid_time** — relative dates ("next Tuesday", "in 3 weeks")
   resolved to absolute datetime using the message's `ts` as anchor. Most
   facts resolve to `valid_time = null` (standing) — treat null as the
   default case, not the exception.
3. **Dedup** — see strategy below.
4. **Conflict / supersede** — a new fact contradicting an existing one sets
   `new.supersedes = old.id`, marks old inactive (salience 0), logs the
   decision. Ordering: dated facts — `valid_time` decides which is current;
   standing facts — later `system_time` (session timestamp) wins; full tie —
   prefer higher-trust `source`. This ordering is what makes
   knowledge-update questions answerable: the judge credits the updated
   value (LongMemEval Fig 10), and the current record is the only active
   one.
5. **Decay** — uniform: active records lose salience with half-life
   `decay_half_life_days` on `now_logical − last_accessed`, **clamped at
   `decay_floor`** (CHANGELOG #11). No step-function, no special-casing of
   dated facts (CHANGELOG #3) — past events are memories, not expirations;
   zeroing them forfeits temporal-reasoning questions (27% of LongMemEval).
   Decay never produces 0; only supersession does. Superseded records are
   already inactive.

## Dedup strategy (v1 — reconsider in later weeks)

Cheap and deterministic, no MinHash/LSH at this scale:

```
exact-normalize → cosine gate → negation screen → value-substitution screen
   → entropy gate → merge
                        │
   negation mismatch ───┤
   value mismatch ──────┴─→ conflict resolution, not merge
```

1. **Exact-normalize pre-filter** — lowercase, strip whitespace/punctuation →
   collapse literal duplicates. Free, zero false positives.
2. **Cosine threshold** — embedding similarity via sqlite-vec brute-force KNN
   against existing records; above `dedup_cosine_threshold` → merge
   candidate.
3. **Negation screen (locked)** — deterministic, fixed hashed lexicon of
   polarity/negation markers (`not`, `no longer`, `stopped`, `doesn't`,
   `never`, `dislikes` vs `likes`, …). Applied only to cosine-gate-pass
   pairs. Mismatch on one side only → route to conflict path. Lexicon edits
   are versioned migrations (see guard). No LLM.
4. **Value-substitution screen (locked, deterministic, no LLM)** — applied to
   cosine-gate-pass pairs that carry triples. If normalized `subject` and
   `predicate` match but `object` differs → route to conflict, not merge.
   ("lives in Boston" vs "lives in Seattle" has high cosine, no negation cue
   — without this screen it merges destructively, which is the documented
   ChatGPT overwrite failure, LongMemEval Appendix B, and destroys the
   knowledge-update category, 16% of the benchmark.) If either record's
   triple is null, the screen abstains and the pair proceeds to the entropy
   gate — conservative: false abstentions are a conflict-resolution problem
   later, not silent data loss now. The comparison is exact match on
   normalized strings — no NLI, no LLM, no fuzzy matching in v1.
5. **Entropy/confidence gate** — short or low-information content (e.g. bare
   names) is unreliable under cosine; below `dedup_entropy_gate`, don't
   auto-merge — keep separate. Prevents wrong merges on ambiguous inputs.
   Convergent prior art: Graphiti gates fuzzy dedup on Shannon entropy and
   defers low-entropy names (dedup_helpers.py:52-80, source-verified).

The pipeline being deterministic + tunable + inspectable *is* the
differentiator vs competitors' LLM-prompt merges — keep it that way.

**Prior-art delta, source-verified (2026-07-20):** the closest in-scope
system (OMEGA v1.5.5) deduplicates by exact `content_hash` at write time
plus an *offline* `compact` action merging near-duplicates at **lexical
Jaccard 0.6** (`schema.py:309`; `server/tool_schemas.py:243`). Write-time
*embedding-space* merge remains unoccupied, and lexical near-dup fails on
paraphrase by construction. OMEGA's conflict engine is a deterministic
4-signal heuristic (negation asymmetry, antonyms, preference value changes,
temporal markers — `contradictions.py`); mnimi's triple-exact comparison is
the sharper, auditable variant of the same idea. Positioning claims must
say this precisely — "they're SHA256-exact" and "they're an MCP server, not
a library" are both stale as of OMEGA v1.5.5 (direct Python API in
`__init__.py`).

**Honest scope note:** LongMemEval's history is constructed to be
non-conflicting outside the planted evidence (Appendix A.2), so dedup itself
is not directly rewarded by the benchmark. Its benchmark role is
(a) protective — the screens prevent KU point loss — and (b) indirect —
more distinct facts per token budget. Dedup's standalone value is a product
claim, validated by the non-regression ablation in the benchmark contract,
not by a score increase.

## Read path (`recall` → `get_context`)

### Retriever (read-side, named component — swappable)

```
Inputs:  (query: str, user_id: str, k: int = MemoryConfig.top_k)
Steps:   embed query via the SAME pinned embedder → unit-normalize → vec0 L2
         KNN over the user's ACTIVE records (salience > 0) → take top-k →
         attach component scores → drop hits below recall_min_relevance
         (default off) → update last_accessed = now_logical on returned hits
Output:  list[ScoredRecord], length ≤ k.
```

- Query embedding goes through `embeddings.py` (same normalize boundary as
  writes); a query embedded any other way breaks ranking parity.
- `k` is held identical across mnimi and every baseline. Retrieval
  correctness gates ~90% of correct answers (LongMemEval Fig 14) — this seam
  is where W3 lives or dies.
- The retriever is a seam: the naive-RAG baseline uses the same retriever
  over raw rounds; only the indexed value differs.

### Ranking (canonical)

```
score = (w_sim · relevance + w_rec · recency) · salience
```
Weights from `MemoryConfig.salience_weights`; defaults `{similarity: 1.0,
recency: 0.0}` — the paper's winning configuration ranks by dense similarity
alone (§5.1); recency weighting is unmeasured and available as a knob. The
stored `salience` field is a decay-managed multiplier, not an additive term;
with the decay floor it ranges [decay_floor, 1] for active records, so decay
re-orders but never hides. No access-frequency term and no `access_count`
field: under the eval protocol (reset per instance, one query per question,
§3.1) it is a constant and cannot move the number. (Both OMEGA and Hypabase
boost by access count — `_base.py` access-reduces-decay;
`memory/strength.py:26` `log(1+access_count)` — rejected here on the same
protocol grounds plus the production feedback-loop risk.)

### `get_context` (locked format)

Assembles top-ranked records into a context string within
`context_token_budget`:

- Records are **ordered by `system_time` ascending** (the paper's pipeline
  sorts retrieved items by timestamp for the reader, §5.1).
- Each record renders as a structured block containing its **timestamp,
  `raw`, and `content`** — the raw span is included because fact-only values
  lose information at the reading stage (Fig 5, §5.2), and the timestamp is
  included because temporal questions are unanswerable without visible dates
  (corroborated externally: Mastra's measured formatting fixes —
  timestamps + question date in prompt — moved their RAG score from ~66 to
  80, mastra.ai/blog/use-rag-for-agent-memory).
- Inactive (salience 0) records excluded.
- The block template is fixed and hashed with the embed template family —
  a silent format change is a silent number change.

## Embedding normalization (locked)

**Decision:** normalize embeddings to unit length at write time, before
insert. Store the normalized vector, not the raw model output.

**Distance metric:** vec0 table uses `distance_metric=L2`, not `cosine`.
For unit vectors, L2² = 2 − 2·cos_sim → identical ranking to cosine, cheaper
op (no norm/sqrt/division at query time, matters under brute-force scan × N
candidates).

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

- Model: `BAAI/bge-small-en-v1.5`, **revision pinned to a specific HF commit
  sha** (a bare model name is mutable under the run)
- Dim: `384`
- Normalization: `embed()` returns unit-normalized vectors, once, inside
  `embeddings.py`. No call site outside it may skip this.
- Prior-art note: OMEGA uses the same embedder (bge-small-en-v1.5 via ONNX
  CPU, `cli.py:54-57`) — the embedder is table stakes, not differentiation.

**Embedding input composition (locked):** the embedded string is `raw` +
`content` concatenated (K=V+fact), NOT content alone. Fact-only keys
underperform raw+fact on both retrieval and downstream QA across every reader
tested (LongMemEval Table 3, §5.3). Template, fixed at write time:

```
embed_input = f"{raw}\n{content}"
```

- `embeddings.py` embeds `embed_input`; `content` alone is never embedded.
- `raw` is stored and immutable. On merge, keep the surviving record's `raw`;
  do not concatenate raw spans (unbounded growth).
- The template string's hash is pinned in the guard so a silent template
  change fails loudly.

**vec0 schema:**
```sql
CREATE VIRTUAL TABLE memories USING vec0(
  embedding float[384] distance_metric=L2
);
```

**Known trade, stated:** BGE-small (33M params) trails the paper's Stella V5
1.5B; retriever choice moves Recall@5 by up to ~9 points and Contriever beats
Stella on several cells (Table 9, Appendix E.2). mnimi's absolute recall will
sit below the paper's numbers; if W3 misses the margin, the embedder is the
first swap (BGE-base 768-dim, same family, bump `embedder_dim`, re-embed).

**Out of scope → FUTURE.md:** `Embedder` protocol / pluggable embedder
choice, API-backed embedders, Matryoshka dim truncation, hot-swap /
re-embed workflow.

## Concurrency

SQLite serializes writers even in WAL mode. All writes go through a single
connection/queue; WAL enabled for concurrent reads. Background
`consolidate()` must not race a live `add()` — same queue.

## Benchmark contract — `evals.base.MemorySystem`

Separate from `Memory`. Any system under test implements `reset()`,
`add(messages)`, `get_context(query) -> str`. The harness drives it
identically for every system so results are comparable.

**Harness pins (checked at harness start, mismatch → refuse to run):**
- `reader_model` — `<model id + revision/hash>`, temperature 0. The reader is
  the QA model that consumes `get_context()` output; it is a SEPARATE pinned
  artifact from the extractor — different identity, different job, never the
  same config field. Held constant across mnimi and all baselines so the only
  variable is the context each system assembles. Evidence this is
  load-bearing: ~15-pt spread across readers on identical retrieval
  (LongMemEval Table 3); a 10.6-pt swing from a reader swap alone in Mastra
  OM (84.23 → 94.87, mastra.ai/research/observational-memory).
- `reader_prompt_hash` — the reader prompt uses Chain-of-Note + JSON-formatted
  records: worth up to 10 points even at oracle retrieval (Fig 6, §5.5). The
  prompt is part of the number; it is hashed like everything else.
- **One reader prompt for all question types (CHANGELOG #13).** Per-category
  answer-prompt tuning is prohibited for every system under test, including
  mnimi. Category-tuned prompts are a disclosed practice behind at least one
  competitor headline (OMEGA 95.4%, vendor-stated) and a documented
  comparability leak (score swings of 5–15 points on prompt/model config,
  CompetitorTeardowns Appendix B). One prompt, one hash, all 500 questions.
- Question date is passed to the reader ("Today's date is {question_date}") —
  temporal questions are unanswerable without it.

**Locked baselines:**
- No-memory (question only) — floor.
- Naive round-RAG: rounds stored verbatim, K=V, same retriever, same k, same
  reader, same prompt. This is the paper's strong baseline (Table 3, K=V
  rows) and the W3 kill-gate bar: mnimi's write-side policy must beat it by a
  repeatable margin or the thesis fails.
- Competitor runs (OMEGA first — the only other in-scope system) go through
  this same harness, same reader, same prompt. Note in methodology: OMEGA has
  no raw-chat ingestion path, so the comparison requires an ingestion
  adapter feeding its store; the adapter is disclosed as part of the setup.

**Required ablation:** every headline run reports mnimi decay-on vs decay-off
at default config. LongMemEval cannot reward decay (it never tests
forgetting); it can only measure decay's cost. With the decay floor
(CHANGELOG #11) the expected cost bound tightens: decay re-ranks but no
longer excludes. The publishable claim is "principled decay at ≤X-point
cost," and X is measured, not asserted.

## Human-readable memory (locked, not future scope)

- Content stored as readable text + structured metadata (schema above).
  Original fact always recoverable without reverse lookup — `raw` now makes
  the source span itself recoverable, not just the distilled fact.
- `export()` dumps the store as readable text/markdown. No UI.
- Every merge / decay / supersede decision logged with a reason:
  - standing: `"decayed: {days} since last access, salience {old} → {new}"`
  - conflict: `"superseded {old_id}: {reason}"`
  - screens: `"routed to conflict: {negation|value-substitution} on {pair}"`

**Out of scope:** dashboard, card UI, visual memory browser — integrating
developer's job.

## Open questions (undecided — leaning noted)

- **Tiered model (core/recall/archival, Letta-style)?** — Lean *no* for v1.
  Differentiator is the write-path policy, not tier structure.
- **Self-editing memory (agent edits its own store via tools)?** — Lean *no*.
  Library manages memory *for* the agent. Out of scope unless a real reason
  surfaces.

## Deferred / Unvalidated (tempting, not benchmark-proven for mnimi — do not build in v1)

- **Hybrid FTS5 + reciprocal-rank fusion** — FTS5 lives inside SQLite (zero
  new dependencies) and OMEGA ships vector→FTS5→RRF in-process
  (source-verified). But LongMemEval Table 9 measured BM25-*alone* well
  below dense retrieval, and the hybrid's effect on this benchmark is
  unmeasured. Cheapest-upside W3+ experiment: exact-term queries (IDs,
  names, numbers) are where dense misses. Deterministic, in-scope, one flag.
- **In-process cross-encoder rerank** — feasibility now source-proven:
  OMEGA ships cross-encoder/ms-marco-MiniLM-L-6-v2 via ONNX, int8 = 571 MB
  disk / ~650 MB RSS, lazy-loaded (`reranker.py`). Effect on LongMemEval
  unmeasured; adds a second model download to the install story. Revisit
  only if W3 shows retrieval-ordering (not recall) failures.
- **Extractor LoRA fine-tune** — the measured lever for small-model
  extraction quality: fine-tuned small models match much larger teachers on
  extraction tasks and off-the-shelf triple extractors perform poorly
  untuned (PMC12065832; arXiv 2507.13827). Trigger: W3 error analysis shows
  extraction (not retrieval or reading) is the bottleneck. Effort cost, not
  infra cost.
- **Time-aware query expansion** — +6.8–11.3% TR recall in the paper
  (Table 4) but only with a strong LLM extracting time ranges; weak models
  hallucinate ranges and false positives actively hurt by pruning the search
  space (Table 11, Appendix E.4). Would also put an LLM on the read path.
  Revisit only if W3 error analysis shows TR retrieval (not reading) is the
  bottleneck, and then evaluate a deterministic date-range parser first.
- **Recency weight > 0 in ranking** — unmeasured; risks burying old evidence
  (IE questions). Tune on W3 evidence only.
- **`recall_min_relevance` > 0** — see config note; activate on measured
  distractor failures only. OMEGA ships 0.60-on; untested on this benchmark.
- **Access-frequency ranking / `access_count`** — provably inert under the
  eval protocol; feedback-loop risk in production; shipped by both OMEGA and
  Hypabase, rejected here on protocol grounds. FUTURE.md if ever.
- **Per-type decay rates / permanent types** — OMEGA's `_DECAY_LAMBDAS`
  (λ=0 for constraints/preferences) is `pinned`-by-type; reintroduces an
  unmeasured type judgment delegated to a 1.7B model. Stays with `pinned`
  in FUTURE.md.
- **`pinned` records** — no benchmark question type requires them;
  extraction-inferred pinning is an unmeasured judgment. FUTURE.md.
- **Prospective-fact expiry** (the legitimate residue of the old
  step-function) — FUTURE.md with the constraint that past events never
  expire.
- **spaCy extraction fallback, pluggable embedders, Matryoshka, API
  embedders** — already in FUTURE.md, unchanged.

## Reproducibility Tiers

The harness makes two claims at two different strengths, and a third is queued
in FUTURE.md. They are published separately because conflating them overstates
what the artifacts prove. The vocabulary is fixed: Tier 1 artifacts are
**auditable**, Tier 2 runs are **reproducible given the pins and the daemon
precondition**. Tier 1 is never described as reproducible, in this document or
anywhere else.

### Tier 1 — Auditable (hardware-independent, anyone, anywhere)

**Claim:** given the published predictions, the judge produces the published
score.

Every run emits three artifacts. A run whose number is quoted anywhere has its
three copied to `results/published/<system>__<N>q/` and committed — `runs/` is
gitignored scratch, `results/published/` is the record.

| artifact | contents |
|---|---|
| `pins.json` | `pins` + `pins_hash`: dataset sha256, reader model + digest, every decode pin, both prompt hashes |
| `predictions.jsonl` | one row per question: `question_id`, `category`, `is_abstention`, `question`, `answer` (gold), `predicted`, `reader_prompt_tokens`, `truncated`, `tokens_dropped` |
| `results.json` | header + `provisional` + run stats + graded rows |

**The self-sufficiency contract.** `question` and `answer` travel inline on
every prediction row. This is not incidental convenience — it is the property
that makes Tier 1 exist, and it is load-bearing:

```bash
python -m evals --stage judge \
    --predictions results/published/no_memory__20q/predictions.jsonl
```

That command needs **no dataset download, no `--system`, no memory store, no
reader, no Ollama, and no GPU**. It runs in seconds, costs cents in judge API
calls, and costs nothing where the verdict cache hits. It is read-only: auditing
a published artifact cannot modify it. It recomputes the score, compares against
the `results.json` beside the predictions, and prints `MATCHES` or `DIVERGES`.

Any change that removes `question` or `answer` from the row schema deletes Tier
1. Treat those two fields as part of the benchmark contract, not as convenient
denormalization.

**What this proves:** the scoring step. It removes the judge transport, the five
per-question-type prompt templates, the abstention dispatch, and the aggregation
logic from the trust surface. A disagreement here is a bug in the harness, and
it is locatable. `judge_prompt_hash()` is in the header, so an edit to any
template cannot ship silently and invalidate a prior verdict set.

**What this does NOT prove:** that the predictions were generated by the
pipeline the pins header claims. Tier 1 verifies scoring, not generation. It is
an audit artifact, not a reproduction — **auditable**, never *reproducible*.
Regenerating predictions is Tier 2 and needs the reader, the pins, and the
daemon precondition.

**Auditable and provisional are orthogonal.** `provisional` (a non-empty list in
`results.json`) answers *may I quote this number?*. Tier 1 answers *does the
judge reproduce it from these predictions?*. A provisional artifact is fully
auditable; it is published for the second question and fails the first. The two
currently published runs are provisional on both `reader_prompt_version`
(`plain-prose-v2`, pending the Phase D `json-con-v1` prompt) and a `-dirty`
`harness_git_sha`.

### Tier 2 — Reproducible given the pins AND the daemon precondition

**Claim:** re-running `--stage predict` under the published pins, on a daemon
launched with the required environment, reproduces the published predictions on
comparable hardware.

#### The pins, and which layer they live at

Request-level, sent on every call and recorded in `pins_hash`:
`temperature=0`, `top_k=1` (temperature 0 alone is not greedy), `seed=0`,
`num_gpu=99`, `num_batch=512`, `num_thread=8`, `num_ctx=32768`.

Daemon-level, resolved when the daemon starts and impossible to set per request:
`OLLAMA_FLASH_ATTENTION=1`, `LLAMA_ARG_CACHE_RAM=0`. These are also in
`pins_hash`, which means a `pins_hash` can describe a configuration the serving
daemon does not implement — hence the preflight below.

Harness-level: the per-question cache-bust prefix (`plain-prose-v2`) and
`_force_model_load`, which are pins in everything but name.

#### The bisection, in full, including two wrong conclusions

The self-corrections are the evidence. A version of this section that stated
only the final answer would be shorter and would teach the next person nothing,
including the next person to run this harness.

**Stage 1 — "GPU inference is non-deterministic; the reader must be CPU-only."
WRONG.** The evidence was real: at `temperature=0`, three identical GPU calls
returned two distinct answers, diverging mid-sentence after an identical prefix
— the signature of logits shifting enough to flip greedy decoding at a near-tie.
It was attributed to CUDA atomics reordering float reductions: an unfixable
hardware property.

It was wrong because **that test never pinned `num_batch`**. Ollama picked a
batch size per load, and llama.cpp logits are not bit-identical across batch
sizes. The drift came from an unpinned harness setting, not from the GPU. An
unpinned batch size mimics the atomics signature exactly, which is what made the
misattribution easy — and is why the wrong conclusion is recorded rather than
quietly overwritten.

**Stage 2 — `num_batch=512` pinned; GPU declared reproducible. INSUFFICIENT.**
Re-tested byte-for-byte: identical output across three calls on one loaded
instance, a cold model reload, a full machine reboot, and 91% GPU utilisation
under contention. All 12 generations hashed to one value. GPU is also ~23x
faster on a 32k prefill (10.4s vs 237.2s), so full offload became the pinned
default.

The protocol was still wrong: **every step replayed the same single question**,
which held cache state constant by accident. In a real run each question has a
distinct context but shares the system-prompt prefix, so question *i*'s prefill
boundary depends on question *i−1*. The transferable lesson: *a protocol that
replays one input cannot detect an order-dependent variable.* Vary the sequence,
not just the repetition count.

**Stage 3 — real runs drifted anyway; the prompt cache reintroduced the same
mechanism through a different door.** llama.cpp keeps a content-addressed prompt
cache. A cold prefill computes the final logits inside a 512-token batch; a
cache hit computes them in a batch of **one** (`n_past = 27622`, *"need to
evaluate at least 1 token"*). Different batch size at the logits position,
different reduction order, different argmax — precisely what `num_batch` was
pinned to prevent. Both states are individually deterministic and both survive a
daemon restart, so this presents as a **bistable output, not as noise**.
`prompt_eval_count` reports the full prompt length on a cache hit, which is why
the existing instrumentation never saw it. `cache_prompt: false` in the request
options is **ignored** by this Ollama build; only `--cache-ram 0`, reached via
the `LLAMA_ARG_CACHE_RAM` passthrough, actually disables it.

Three further variables were isolated and closed in the same pass:

| variable | effect | fix |
|---|---|---|
| Slot prefix reuse | the 64-token shared system prompt is reused, so question *i*'s prefill boundary depends on question *i−1* | per-question cache-bust prefix (`plain-prose-v2`); prefill goes to the full token count every time |
| Flash attention | changes attention tiling and therefore reduction order: with cache state held constant, FA=0 vs FA=1 changed 2/2 probe predictions (byte 0 of a 1,924-char answer; byte 255 of a 383-char one). Unset, the daemon resolves `flash_attn = auto`, and what `auto` picks is a property of the host GPU | pinned to `1`, asserted at preflight |
| CUDA graph warmup | the first inference after a model load runs against a cold graph cache (555 graphs reused vs 1,608 once warm) and answers differently | `_force_model_load` before question 1 |

`_force_model_load` deserves its name in the spec because it looks like a
preflight helper and is not. Determinism does **not** come from saturating the
graph cache — a synthetic warmup reaching 1,033 of 1,608 did not reproduce the
steady state. It comes from an **identical request sequence**: both runs of a
restart pair begin with the same fixed 30-token generate, so graph reuse tracks
identically (1, 95, 205, 458 in both). Remove the call and question 1 drifts
again.

#### Measured error bar

Same pins, same `pins_hash`, daemon killed and relaunched between runs, on a
clean GPU:

| system | predictions changed | score |
|---|---|---|
| `no_memory` | **0/20** | 10.0% → 10.0% |
| `full_history` | **0/20** | 20.0% → 20.0% |

Zero. The honest error bar across a daemon restart is 0/20 predictions and 0
points. **Retired: the earlier "12/20 changed, 5 points" figure must not be
cited.** It was a cache-state artifact, and it was weaker evidence than it
looked: both artifacts it came from record `stage='judge'` — judge-stage replays
over a stored `predictions.jsonl` — so `full_history`'s
`judge_cache_hits=8 / misses=12` *was* the "12/20 changed" number. That is
verdict-cache bookkeeping over predictions of unknown provenance, not a
controlled comparison. **Before quoting any reproducibility number, check
`run.stage`:** only `stage='all'` or a fresh `predict` re-runs the reader, and
only those can measure reader drift.

#### The daemon precondition, and why it is part of the claim

The daemon must be launched manually with both variables set, and **the tray app
must not be serving**: it starts a daemon on 11434 with neither set, silently
yielding `flash_attn = auto` plus a live 8 GiB prompt cache — a different
configuration wearing this configuration's `pins_hash`.

```
OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0 OLLAMA_SERVE_LOG=<path> ollama serve
```

`preflight_reader_env` reads the daemon's **resolved** values out of its load log
and refuses to run on a mismatch — requested-vs-resolved, the discipline that
caught flash attention. It demands *positive proof* that the prompt cache is
disabled: the daemon announces the two states in different sentences and emits
no `cache state` line at all when the cache is off, so keying off the limit alone
treated absent evidence as a pass. A preflight that passes on no evidence is
worse than no preflight.

Killing the daemon is not enough — `llama-server.exe` child runners outlive
`Stop-Process -Name ollama` and keep holding VRAM. Six accumulated unnoticed
during this investigation, holding 5.3 GiB on a 6 GiB card; a 27k prefill that
takes 7.3s on a clean card took 50-59s under that contention. Contention did not
change output — controlled comparisons under it stayed byte-identical — but it
makes every timing number meaningless.

#### `pins_hash` identifies the request; the `run` block records the resolution

`pins_hash` is the hash of the configuration the harness **asked for**. The
`run.environment` block records what the daemon **actually resolved**:
`flash_attention_reported`, `prompt_cache_reported`, `ollama_version`,
`gpu_model`, `driver_version`, `cuda_version`, `offloaded_layers`,
`kv_cache_type`, `model_blob_path`, `runner_cmd`, client and daemon `OLLAMA_*`
environments, and an explicit `ollama_env_mismatch`, with the verbatim load block
saved beside the run as `model_load.log`.

This block is **diagnostic only — it never enters `pins_hash`**, so it cannot
alter the identity of a run. Its purpose is to make a divergence report locatable
rather than mysterious: the next drift should fall out of a diff between two run
blocks.

#### Tolerance, stated honestly

These pins fix reduction order on a given machine. They do not guarantee
bit-identity across different GPUs, drivers, CUDA versions, or Ollama builds —
each can change kernel selection and therefore the order of float reductions,
which flips argmax at near-ties. Divergence across dissimilar hardware is
expected and is not a harness defect.

The `run` block does not currently record CPU model or SIMD dispatch path. That
is a real gap only on the `--num-gpu 0` path, where llama.cpp's AVX2 / AVX-512 /
NEON dispatch decides reduction order; at the pinned full offload the GPU-side
fields above are the identifying ones.

**Stated as:** "reproducible given these pins and the daemon precondition, on
comparable hardware." Never as universal bit-identity. Measured cross-hardware
divergence, once available, is published as an observed delta (N/20 divergent
predictions, score delta X%) rather than asserted as negligible.

`--num-gpu` opts into exploratory runs off the pin; any artifact produced that
way self-marks provisional, as does any deviation from `num_batch` or
`cache_ram`.

#### The verdict cache is the drift detector

The judge verdict cache keys on `(question_id, sha256(predicted))`. It exists to
save money, but its real value is diagnostic: **an unchanged re-run that misses
the cache means the reader moved.** Every finding in this section was caught that
way. A re-run under identical pins should report all hits and zero misses; a run
of misses on unchanged inputs is a drift alarm, and it fires before any score is
compared.

**SEE TIER 3 IN FUTURE.md** — a containerized reference environment, queued
rather than forced.

### Applies to competitors

Competitor runs (OMEGA first) emit the same three artifacts through the same
judge, so their numbers are auditable on identical terms. Where a competitor's
internals are non-deterministic (e.g. wall-clock decay), that is disclosed on
that system's row rather than silently absorbed.
