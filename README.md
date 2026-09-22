# MNIMI

Embeddable, local-first agent memory. One SQLite file, zero infra.

The number that matters is the benchmark number. Everything in this repo is
judged by whether it moves the table below.

## LongMemEval (longmemeval_s, ~500 questions)

Reader model, reader prompt, retrieval `k`, and ingestion granularity are held
identical across every row — the only variable is the context each system
assembles. `no_memory` is the floor (answer with no history); `full_history` is
a truncated-context baseline (stuff as much history as fits the pinned 32K
window — it truncates, so it measures what naive context-stuffing buys, not
what is achievable); `oracle` is the evidence-availability bound (the reader is
handed exactly the evidence sessions); `naive_rag` (rounds stored verbatim) is
the bar mnimi has to clear while using a fraction of the tokens. That is the
entire bet.

### The published number — five arms, n=100, one sitting

| System         | Score         | 95% CI (Wilson) | mean fed tokens | truncated |
| -------------- | ------------: | --------------: | --------------: | --------: |
| no_memory      |  4/100 (4%)   |     [1.6, 9.8]  |             166 |     0/100 |
| full_history   | 17/100 (17%)  |    [10.9, 25.5] |          27,464 |   100/100 |
| oracle         | 50/100 (50%)  |    [40.4, 59.6] |           5,153 |     0/100 |
| naive_rag      | 41/100 (41%)  |    [31.9, 50.8] |           4,710 |     0/100 |
| **mnimi**      | 35/100 (35%)  |    [26.4, 44.7] |           4,708 |     0/100 |

**Provenance:** sitting of 2026-08-16 · reader transport **Ollama 0.32.13** ·
harness commit `ae5da2b` (clean tree) · reader `qwen2.5:1.5b-instruct-q4_0`
(digest `635e70c8…`), prompt `mnimi-con-v1` · judge `gpt-4o-2024-08-06` ·
stratified 100-question slice, seed 0 · NVIDIA RTX 1000 Ada Laptop GPU, driver
595.95, CUDA 13.2 · `provisional: []` on all five. The artifacts, the full
provenance table, the paired McNemar tests and every caveat live in
[`results/published/`](results/published/); the numbers above are copied from
there, never typed in.

Read it with the caveats attached:

- **mnimi is significantly above the floor** (vs `no_memory`: b=33, c=2,
  Holm-corrected p≈0) and **below the evidence-availability bound**
  (vs `oracle`: b=3, c=18, p_holm=0.003).
- **mnimi vs naive_rag — the pre-specified primary — went against mnimi**
  (b=0, c=6, p=0.031): all six discordant questions are ones the v1 dedup screen
  changed the retrieved set on. Six is the smallest discordance at which p<0.05
  exists, and the pre-registration was written for an earlier reader build, so
  this is a directionally uniform signal to investigate, not a confirmatory
  result. v1's write side is a dedup screen and nothing else; the bet attaches
  to the extraction era.
- `full_history` truncates every question to the most recent ~27k tokens; it is
  a truncation policy, not full history.
- Absolute scores carry judge instrument error (gpt-4o at temperature 0 flipped
  9 of 140 re-gradings on one borderline row) on top of sampling error. No
  per-category cell is quoted: at n=100 they hold 16–17 questions each.
- **The reader build is part of the number.** The same pins on Ollama 0.32.5
  scored 5 / 12 / 43 / 41 / 43 with byte-identical prompts — every point of
  movement was the reader binary. The build is now a harness pin (below).

### The gpt-4o-era number — four arms + the cited `full_history` row, n=100, one sitting

The same harness, the same 100 questions, the paper's reader and judge
snapshot (`gpt-4o-2024-08-06`, 128K window, temperature 0, `seed=0`). Decided
2026-09-11; pre-registered before the first row; run 2026-09-12.

| System         | Score         | 95% CI (Wilson) | mean fed tokens | truncated |
| -------------- | ------------: | --------------: | --------------: | --------: |
| no_memory      |  5/100 (5%)   |     [2.2, 11.2] |             135 |     0/100 |
| full_history   | *cited* 64.0% (paper Fig. 3b, GPT-4o + Chain-of-Note, ~500 q) | — | — | — |
| oracle         | 90/100 (90%)  |    [82.6, 94.5] |           4,992 |     0/100 |
| naive_rag      | 80/100 (80%)  |    [71.1, 86.7] |           4,538 |     0/100 |
| **mnimi**      | 75/100 (75%)  |    [65.7, 82.5] |           4,537 |     0/100 |

**Provenance:** sitting of 2026-09-12 · reader transport **openai**, snapshot
`gpt-4o-2024-08-06` served through the Batch API in 7–8 sub-batches per arm ·
harness commit `dd73736` (clean tree, v1.6.2) · prompt `mnimi-con-v1`, text
renderer (chosen by the pre-registered presentation pair: JSON scored lower on
both oracle, 88, and mnimi, 71) · judge `gpt-4o-2024-08-06` · stratified
100-question slice, seed 0, the same ids as the local sitting ·
`provisional: []` on every directory · $5.04 of API spend for the whole
sitting. Artifacts, the full provenance table, the paired tests and the pair
and drift evidence live in [`results/published/`](results/published/).

Read it with the caveats attached:

- **mnimi vs naive_rag — the pre-registered primary — is a null, as
  pre-registered** (b=1, c=6, p=0.125), and it leans the same way as the local
  family (there 0 / 6): the v1 dedup screen costs rows it never earns back.
  That is the input to the next phase's R3, not a capability claim.
- **mnimi is significantly above the floor** (vs `no_memory`: b=71, c=1) and
  **significantly below the evidence-availability bound** (vs `oracle`: b=2,
  c=17, p_holm=0.0007). Oracle at 90 sits inside the paper's 92.4.
- **This family is reproducible within measured drift, not byte-identical.**
  A second mnimi predict run an hour later, same pins: 85/100 answers rewritten
  at the byte level, prompt tokens identical, score 75 → 75, 3 rows flipped
  each way. The statement is "score within 6/100 flips"; the text is not
  reproducible. (`python -m evals.drift` on the two published directories.)
- **`full_history` is cited, not run**: at 128K it costs ~$15 per n=100 sitting
  for a row the paper reports on this exact reader, so the harness refuses the
  arm on this transport and the table carries the paper's number, marked.
- Absolute scores carry judge instrument error on top of sampling error; no
  per-category cell is quoted.

**Phase 1 (2026-09-12/13, same family, same 100 questions).** Each retrieval
knob was probed for recall first (`evals/probes/`, no API), then measured as
a pre-registered pair of mnimi arms at one commit:

| knob | baseline | variant | b / c | verdict |
| --- | ---: | ---: | :---: | --- |
| dedup scope, store → session (R3) | 77 | 79 | 5 / 3 | adopted: 4 dropped evidence rounds recovered |
| BGE query instruction (R5) | 79 | 81 | 5 / 3 | adopted; naive_rag under it: 82 |
| chunk long rounds (R4) | — | — | — | rejected at the probe (ANY@10 91 → 89) |
| `top_k` 10 → 20 (R6) | 82 | 81 | 6 / 7 | k=10 stays; twice the tokens for nothing |

Under the adopted configuration **mnimi reads 81 and 82 in two sittings and
naive_rag 82; the primary is b=2, c=3** — the v1 write side no longer costs
anything measurable against verbatim storage, which is all it can claim
before extraction. Of the 18 remaining misses, 14 are reading misses with
the evidence inside the top-10 (oracle also fails 6 of them); that is where
Phase 2 starts. Nothing here is significant at n=100 and nothing is claimed
to be. Artifacts, provenance and drift readings in
[`results/published/`](results/published/); the rulings in `docs/DECISIONS.md`.

**Phase 2 — extraction (2026-09-13 → 15, same family, same 100 questions).**
A local, pinned extractor (Qwen3-1.7B Q8_0 through llama-cpp-python on the
GPU, grammar-constrained, byte-stable) turns each round into fact records
stored beside the round; the reader sees a retrieved round under a `facts:`
header. Pre-registered gates, in order: the corpus pass's retrieval probe
(ANY@10 93/95 vs 91, ALL@10 83/95 vs 77, no API), the n=20 dev prefix
(18/20 = the k10 arm's), then one four-arm sitting at one clean commit:

| arm | score | b / c vs baseline | mean fed tokens | verdict |
| --- | ---: | :---: | ---: | --- |
| baseline (no extractor, `turns`) | 80 | — | 4,515 | drift vs k10: 97/100 texts changed, 0/100 prompt tokens, 82 → 80 |
| **extraction, `round+facts`** | **84** | 7 / 3 | 5,298 | **adopted** (library and harness default since v1.9.0) |
| extraction, `facts` only | 78 | 2 / 8 (vs `round+facts`) | 1,701 | rejected: drops the turns single-session rows need |
| naive_rag | 79 | 7 / 2 (primary, vs `round+facts`) | 4,534 | mnimi ahead for the first time; p=0.18, not significant |

The pass took 27 h once (cached for every later run); Phase 2 spent $2.98.
Nothing here is significant at n=100 and nothing is claimed to be; the
extraction era's primary is settled at n=500 (Phase 5). Artifacts and the
full provenance in [`results/published/`](results/published/); the rulings
in `docs/DECISIONS.md` ("Gate 4-i read", "Gate 4-ii read", "Gate 4-iii
read").

**Phase 3 — conflict and the deterministic screens (2026-09-15 → 16, no API, $0).**
The write side gained SPEC's dedup steps 3–5 and write-path step 4 with no
LLM: a frozen, hashed negation lexicon, a value-substitution screen over
normalized triples, a token-entropy gate, and supersession — a fact that
conflicts with an earlier one under one of three rules (negation, a
functional predicate such as residence or employer, a changed number on the
same residue) leaves exactly one active fact, the loser at salience 0 with
the winner pointing at it. LongMemEval's history is built non-conflicting,
so the slice can only guard, not falsify: the retrieval probe read ANY@10
93/95 and ALL@10 83/95 unchanged, 0 evidence rounds lost, 326 fact pairs
kept apart that the old screen had merged, evidence ranks identical on
99/100 questions, and 46 facts superseded across the 100 stores. The
falsification is a seeded, CI-run demo set of 100 authored pairs
(`evals/probes/conflict_demo.py`, `python -m evals.probes.conflict_demo
--seed 0`; the exit code is the gate):

| family | pairs | mnimi (screens + supersede) | exact-dedup (the v1.9 write path) | pre-registered |
| --- | ---: | ---: | ---: | --- |
| value change (functional 15, count 15) | 30 | **30** | 0 | ≥ 27 |
| negation (marker 15, antonym 15) | 30 | **30** | 0 | ≥ 27 |
| dated update (in order 10, reversed 10) | 20 | **20** | 0 | ≥ 18 |
| controls: paraphrase 10 + unrelated 10 (must not conflict) | 20 | **20** | 20 | 20 |

Identical after `add()` and after `consolidate()`; identical under the real
BGE embedder. The same rounds through the real 1.7B extractor instead of the
authored facts (descriptive, never the gate): 42 of 80 pairs came out keyed
on the authored pair and all 42 resolved correctly, 0 controls were touched,
and the rest are rounds the model returned `[]` on. **No benchmark score
moved and none is claimed**: the read path does not read `salience` until
Phase 4, so a superseded fact still renders. Rulings in `docs/DECISIONS.md`
("Phase 3 pre-registration" through "Phase 3 closes").

**Phase 4 — decay, ranking and the read side (2026-09-17 → 18, gpt-4o family, same 100 questions).**
The read path gained SPEC's ranking — `score = (w_sim·relevance + w_rec·recency)·salience` over an
exact top-k of rounds — the salience-0 exclusion (a superseded fact neither ranks nor renders),
`recall_min_relevance`, the `last_accessed` write-back, and decay with its half-life and floor
inside `consolidate()`. Two $0 probe gates ran first: identity under the landing defaults
(100/100 rows unchanged) and the exclusion priced on the slice (ANY@10 93/95, ALL@10 83/95, 0
evidence rounds out of the top-10). Then one three-arm sitting at one clean commit:

| arm | score | b / c vs the previous arm | mean fed tokens | verdict |
| --- | ---: | :---: | ---: | --- |
| A the v1.10 read path | 86 | — | 5,301 | drift vs the Phase 2 arm: 91/100 texts changed, 9/100 prompt tokens, 84 → 86 |
| **B SPEC's read path** (`active_only`, `ranking=score`) | **87** | 2 / 1 | 5,302 | **adopted** (library defaults since v1.11.0) |
| C B plus decay (`consolidate on`) | 81 | 2 / 8 | 5,307 | **not adopted**: decay costs 6 points (X = −6), outside the family's drift band |

SPEC's decay-on/off ablation is the B → C row, and it is a negative result reported as one: decay
multiplies down exactly the old evidence rounds LongMemEval asks about (the retrieval probe read
ANY@10 93 → 89 and ALL@10 83 → 75 before the arms ran, and four of the eight rows the reader lost
are rounds whose evidence left the top-10 there). Decay ships built, tested and off — one flag
away — and the phase spent $2.56. Nothing here is significant at n=100 and nothing is claimed to
be; the primary against `naive_rag` is settled at n=500 (Phase 5). Rulings in `docs/DECISIONS.md`
("Phase 4 pre-registration" through "Phase 4 closes").

**Phase 5 — the verdict on the full benchmark (2026-09-21 → 22, gpt-4o family, all 500 questions).**
The five arms at one clean commit (`f07c24d`), one day, $17.55, `provisional: []` on all five.
The n=100 slices nest inside this one, so these are the first full-benchmark numbers, not
replications of earlier ones.

| arm | score | Wilson 95 % | role |
| --- | ---: | :---: | --- |
| `no_memory` | 6.2 % | [4.4, 8.7] | the floor |
| `full_history` | *64.0 %* | — | cited from the paper (Fig 3b); refused by the harness |
| `naive_rag` | 74.6 % | [70.6, 78.2] | the strong K=V baseline, identical ingestion |
| **`mnimi`** | **84.4 %** | **[81.0, 87.3]** | the adopted configuration (v1.11 read path, extraction on, decay off) |
| `mnimi --consolidate on` | 78.2 % | [74.4, 81.6] | SPEC's decay-on/off ablation: X = −6.2, p = 2 × 10⁻⁴ |
| `oracle` | 91.8 % | [89.1, 93.9] | the evidence-availability bound |

**mnimi beats the strong K=V baseline by 9.8 points, significantly** — paired exact McNemar
b=75, c=26, **p = 1.1 × 10⁻⁶** — for the first time in the programme, and where memory has
to be more than retrieval: temporal-reasoning +25, multi-session +12; the single-session
cells are saturated for both. The programme's pre-registered 85 criterion (a Wilson lower
bound ≥ 85.0, i.e. 441/500) is **not met**: 422 read, lower bound 81.0. Of mnimi's 78 misses,
24 are questions oracle also fails and 54 are questions oracle answers — 27 of those
multi-session, where the retriever shows the reader part of a multi-hop question's evidence
(ALL@10 77.7 %). Rulings in `docs/DECISIONS.md` ("Phase 5 pre-registration" through "Phase 5
closes"); every number is auditable with `python -m evals --stage judge --predictions <file>`
over `results/published/*__500q_gpt4o*`.

Two n=20 smoke artifacts from 2026-07-28 (`no_memory__20q`,
`full_history__20q`, reader prompt `plain-prose-v2`, dirty tree) remain in
`results/published/` because a published artifact is immutable. They are marked
provisional by the harness and are quoted nowhere.

## Running the harness

The harness is the source of truth for every claim in this repo. Two reader
families: the local one runs the reader through Ollama and only the judge
touches an API; the gpt-4o one reads and judges over the OpenAI API.

```bash
pip install -e ".[eval]"
ollama pull qwen2.5:1.5b-instruct-q4_0     # the pinned reader
echo 'OPENAI_API_KEY=sk-...' >> .env       # judge only (gpt-4o-2024-08-06)

# Three daemon-level pins, none of which can be sent per request:
#   - the Ollama build itself must be 0.32.13 — preflight reads GET /api/version
#     and refuses any other build BEFORE loading a model (a build change moved
#     20/20 predictions on identical prompts). Run the release zip from a
#     version-named folder, not the desktop installer: the desktop app's updater
#     replaces the build on its own schedule and cannot be switched off;
#   - OLLAMA_FLASH_ATTENTION=1 and LLAMA_ARG_CACHE_RAM=0, resolved at daemon
#     start; nothing else may be serving on 11434.
# The daemon logs to stderr; preflight reads the file OLLAMA_SERVE_LOG names.
# A run under the wrong daemon is a different configuration wearing this one's
# pins_hash, so preflight refuses to run.
export OLLAMA_SERVE_LOG="$PWD/serve.log"
OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0 /path/to/ollama-0.32.13/ollama.exe serve >> "$OLLAMA_SERVE_LOG" 2>&1 &

python -m evals --system mnimi --limit 100          # predict + judge, local family
python -m evals --system no_memory --limit 500      # the full set

# The gpt-4o family: the same harness with an API reader, through the Batch
# API at half price, as a sequence of sub-batches under the organization's
# enqueued-token cap (90k at this key's tier; a whole n=100 arm would fail
# validation). Resumable — re-running the same command in the same
# --run-dir polls the submitted batch instead of paying again. Every run
# projects its cost before the first call and refuses above the budget;
# `python -m evals.pricing` shows the spend ledger.
python -m evals --system mnimi --limit 100 --reader-transport openai --batch
```

`--limit` defaults to 10 (a cheap smoke run) and selects questions
**stratified** across categories, because the dataset is clustered by category
and a file-order slice comes back single-category — not comparable across
systems. Pass a limit at or above the dataset size to run everything. n=20 is a
smoke test only: categories sit at 3–4 questions and one question is 5 points.

The two stages split, so neither half needs the other's dependency:

```bash
python -m evals --system no_memory --limit 100 --stage predict   # Ollama, no API key
python -m evals --system no_memory --limit 100 --stage judge     # API key, no GPU
python -m evals.stats runs/mnimi__100q runs/naive_rag__100q      # paired McNemar + Holm
python -m evals.drift runs/mnimi__100q runs/mnimi__100q_again    # N/n changed, one configuration
```

Each run writes `pins.json`, `predictions.jsonl`, and `results.json` under
`runs/<system>__<N>q/`. The pins header records the dataset sha256, reader model
digest, the Ollama build, every decode pin, and both prompt hashes, so a run is
self-identifying; `results.json` additionally records what the daemon actually
resolved. `runs/` is gitignored scratch; a run whose number gets quoted is
copied to [`results/published/`](results/published/) and committed.

## Reproducibility

Two claims at two strengths, plus one queued. They are published separately
because conflating them overstates what the artifacts prove.

### Tier 1 — auditable. Verify any published number yourself:

```bash
python -m evals --stage judge \
    --predictions results/published/mnimi__100q/predictions.jsonl
```

Seconds to run. **No GPU, no Ollama, no model weights, no dataset download, and
no `--system`** — every prediction row carries its question and gold answer
inline, so the judge needs nothing else. Costs cents in API calls, and nothing
where the verdict cache hits. Read-only: it cannot modify what it audits. It
recomputes the score, compares it to the committed `results.json`, and prints
`MATCHES` or `DIVERGES`.

This proves the **scoring** step — judge transport, the five per-type prompt
templates, abstention dispatch, aggregation. It does **not** prove the
predictions came from the pipeline the pins claim. Tier 1 is *auditable*, never
*reproducible*.

### Tier 2 — reproducible given the pins and the daemon precondition

`--stage predict` regenerates the predictions given the request pins
(`num_gpu=99`, `num_batch=512`, `num_thread=8`, `top_k=1`, `seed=0`,
temperature 0, `num_ctx=32768`) **and** the daemon-level pins — the Ollama build
(`0.32.13`), `OLLAMA_FLASH_ATTENTION=1`, `LLAMA_ARG_CACHE_RAM=0` — which are
resolved at daemon start and asserted at preflight before any model load.

Measured error bar on the published build (2026-09-10, Ollama 0.32.13): a
fresh `--stage predict` of the mnimi arm at n=100 replayed **100/100
predictions byte-identical** to the published artifact, prompt tokens identical
on every row, across a daemon restart, a reinstall of the server binary from
the release zip, 25 days, three harness commits and the pins schema bump. The
evidence is committed beside the published set as
`results/published/mnimi__100q_restart_2026-09-10/`, and anyone can diff the
two `predictions.jsonl` files — or run `python -m evals.drift` on the two
directories, which prints the same 0/100 (a test does exactly that).

Across builds the reader binary is the whole error bar: the 0.32.5 → 0.32.13
change, with pins, prompts and fed-token counts byte-identical on every row,
changed **20/20 predictions** on every arm and moved the headline by up to 8
points. That is why the build is a pin and any other build is refused. Earlier
restart figures were measured under a retired prompt or a retired build and
are not cited. The bisection that found the request-level and daemon-level
pins, including the two wrong conclusions along the way, is in
[docs/SPEC.md](docs/SPEC.md) and [docs/DECISIONS.md](docs/DECISIONS.md),
because the self-corrections are the evidence.

Bit-identity across different GPUs, drivers or CUDA versions is **not**
claimed: each can change kernel selection and therefore float reduction order.
Deviating from a request pin self-marks the artifact provisional; deviating
from a daemon pin, the build included, is refused outright.

### Tier 3 — containerized reference environment

Queued, not forced: the variables that would have made it mandatory (flash
attention, the prompt cache, the server build) proved controllable in-process
or checkable at preflight. Trigger conditions in [docs/FUTURE.md](docs/FUTURE.md).

Competitor runs go through the same harness, the same reader, and the same judge,
so their numbers are auditable on identical terms.

## API

```python
from mnimi import Memory

mem = Memory(db_path="agent.db", embedder=embedder)   # optional: config=MemoryConfig(...)
mem.add(messages, user_id="u1")               # write path
mem.recall(query, user_id="u1")               # raw retrieval -> list[ScoredRecord]
mem.get_context(query, user_id="u1")          # assembled context -> str
mem.consolidate(user_id="u1")                 # merge / resolve conflicts / decay
mem.export(user_id="u1")                      # human-readable dump -> str
```

Each message is `{"role": "user"|"assistant", "content": str, "ts": "<ISO>"}`.
`ts` is the session timestamp, and it is the only clock: decay, recency, and
ordering all read logical time derived from the `ts` values the store has seen,
never wall-clock. Same inputs, same numbers, on any day.

Five methods. The surface stays thin on purpose; the depth lives behind `add`
and `consolidate`. Core install pulls in only `sqlite-vec` and `numpy` — no
server, no external services, one file on disk.

## Status

Pre-alpha, v1.12.0. The block above is the locked contract; what ships today
is narrower. Built: per-round ingestion; the one LLM the library will ever
call — a local, pinned extractor (`Memory(..., extractor=...)`, the `[extract]`
extra) whose facts are stored beside each round; dedup as exact match, one
cosine probe (`dedup_cosine_threshold=0.95`) and, for facts, the negation,
value-substitution and entropy screens; conflict resolution and supersession
over a frozen, hashed lexicon; **the read side — SPEC's ranking
(`(w_sim·relevance + w_rec·recency)·salience` over an exact top-k of rounds),
`recall()` returning `ScoredRecord`, the salience-0 exclusion, the
`last_accessed` write-back — and decay with its half-life and floor inside
`consolidate()`**; the one shared context renderer (two framings, three units,
one pinned hash each); the `memory_meta` guard (seventeen rows, decay's rules
among them) that refuses a store built under different pins; the real
`BAAI/bge-small-en-v1.5` embedder behind the `[embed]` extra (the default
import path is a numpy hashing placeholder); and the full five-arm eval
harness. Decay is built, measured and **off by default**: the pre-registered
ablation read it at −6 points on n=100, so the harness calls `consolidate()`
only behind `--consolidate`. Not built: `export()`, the context token budget,
`raw` in the rendered block. `docs/SPEC.md` § "v1 as built" is the exact list;
the rest of SPEC describes the target.

See [docs/SPEC.md](docs/SPEC.md) for the contract,
[docs/DECISIONS.md](docs/DECISIONS.md) for locked decisions, and
[docs/FUTURE.md](docs/FUTURE.md) for what is deliberately deferred.

## License

[Apache-2.0](LICENSE).
