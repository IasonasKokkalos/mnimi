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

Pre-alpha, v1.8.0. The block above is the locked contract; what ships today is
narrower. Built: per-round ingestion with an exact-match plus cosine dedup
screen (`dedup_cosine_threshold=0.95`), top-k retrieval over `sqlite-vec`, the
one shared context renderer (two framings, text and JSON, one pinned hash per
framing), the `memory_meta` guard that refuses a store built
by a different embedder or embed template, the real `BAAI/bge-small-en-v1.5`
embedder behind the `[embed]` extra (the default import path is a numpy hashing
placeholder), and the full five-arm eval harness. Not built: extraction (the
one LLM the library will ever call), conflict resolution, decay and the salience
multiplier — `consolidate()` is a no-op — and `export()`; `recall()` currently
returns plain records without the score. `docs/SPEC.md` § "v1 as built" is the
exact list; the rest of SPEC describes the target.

See [docs/SPEC.md](docs/SPEC.md) for the contract,
[docs/DECISIONS.md](docs/DECISIONS.md) for locked decisions, and
[docs/FUTURE.md](docs/FUTURE.md) for what is deliberately deferred.

## License

[Apache-2.0](LICENSE).
