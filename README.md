# MNIMI

Embeddable, local-first agent memory. One SQLite file, zero infra.

The number that matters is the benchmark number. Everything in this repo is
judged by whether it moves the table below.

## LongMemEval (longmemeval_s, ~500 questions)

Reader model, reader prompt, retrieval `k`, and ingestion granularity are held
identical across every row — the only variable is the context each system
assembles. `no_memory` is the floor (answer with no history); `full_history` is
the ceiling (stuff the entire conversation history into context); `naive_rag`
(rounds stored verbatim) is the bar mnimi has to clear while using a fraction of
the tokens. That is the entire bet.

| System         | Overall | single-session-user | single-session-assistant | single-session-preference | temporal-reasoning | knowledge-update | multi-session |
| -------------- | ------: | ------------------: | -----------------------: | ------------------------: | -----------------: | ---------------: | ------------: |
| no_memory      |     TBD |                 TBD |                      TBD |                       TBD |                TBD |              TBD |           TBD |
| full_history   |     TBD |                 TBD |                      TBD |                       TBD |                TBD |              TBD |           TBD |
| naive_rag      |       — |                   — |                        — |                         — |                  — |                — |             — |
| **mnimi**      |       — |                   — |                        — |                         — |                  — |                — |             — |

`—` = not built yet. `naive_rag` and `mnimi` ship in the same batch, because the
two numbers are only interpretable side by side.

### Provisional smoke run — not a published number

A 20-question stratified slice, reader `qwen2.5:1.5b-instruct-q4_0`, judge
`gpt-4o-2024-08-06`. Artifacts committed under
[`results/published/`](results/published/):

| System       | Overall | ssu | ssa | ssp |  tr |  ku |  ms |
| ------------ | ------: | --: | --: | --: | --: | --: | --: |
| no_memory    |  2/20   | 1/4 | 1/4 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_history |  4/20   | 1/4 | 0/4 | 0/3 | 1/3 | 2/3 | 0/3 |

The harness marks both **PROVISIONAL — not publishable**, for two reasons: the
reader prompt is `plain-prose-v2` and the pinned configuration calls for JSON +
Chain-of-Note (`json-con-v1`, Phase D), and the harness tree was dirty at run
time. A 20-question slice also carries a sampling error far larger than any
effect worth reporting. Quote these numbers nowhere.

`full_history` truncated all 20 questions — mean 27,210 prompt tokens fed, ~1.8M
dropped (~77% of history) — so that row is "most recent ~27k tokens", not a
ceiling.

Provisional and auditable are separate claims: these artifacts fail the first and
pass the second. Anyone can recompute their score in seconds (below).

## Running the harness

The harness is the source of truth for every claim in this repo. The reader runs
locally through Ollama; only the judge touches an API.

```bash
pip install -e ".[eval]"
ollama pull qwen2.5:1.5b-instruct-q4_0     # the pinned reader
echo 'OPENAI_API_KEY=sk-...' >> .env       # judge only (gpt-4o-2024-08-06)

# The daemon MUST be launched with these set, and the tray app must not be
# serving on 11434. Both are resolved at daemon start and cannot be sent
# per-request; a run under the wrong daemon is a different configuration
# wearing this one's pins_hash, so preflight refuses to run.
OLLAMA_FLASH_ATTENTION=1 LLAMA_ARG_CACHE_RAM=0 OLLAMA_SERVE_LOG=serve.log ollama serve

python -m evals --system full_history --limit 20    # predict + judge
python -m evals --system no_memory --limit 500      # the full set
```

`--limit` defaults to 10 (a cheap smoke run) and selects questions
**stratified** across categories, because the dataset is clustered by category
and a file-order slice comes back single-category — not comparable across
systems. Pass a limit at or above the dataset size to run everything.

The two stages split, so neither half needs the other's dependency:

```bash
python -m evals --system no_memory --limit 20 --stage predict   # Ollama, no API key
python -m evals --system no_memory --limit 20 --stage judge     # API key, no GPU
```

Each run writes `pins.json`, `predictions.jsonl`, and `results.json` under
`runs/<system>__<N>q/`. The pins header records the dataset sha256, reader model
digest, every decode pin, and both prompt hashes, so a run is self-identifying.
`runs/` is gitignored scratch; a run whose number gets quoted is copied to
[`results/published/`](results/published/) and committed.

## Reproducibility

Two claims at two strengths, plus one queued. They are published separately
because conflating them overstates what the artifacts prove.

### Tier 1 — auditable. Verify any published number yourself:

```bash
python -m evals --stage judge \
    --predictions results/published/no_memory__20q/predictions.jsonl
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
temperature 0, `num_ctx=32768`) **and** the daemon-level pins
(`OLLAMA_FLASH_ATTENTION=1`, `LLAMA_ARG_CACHE_RAM=0`), which are resolved at
daemon start and asserted at preflight.

Measured error bar, same `pins_hash` across a daemon restart on a clean GPU:
**0/20 predictions changed** on both published systems (10.0% → 10.0%,
20.0% → 20.0%). Getting there took three attempts and two wrong conclusions —
the full bisection, including the retired ones, is in
[docs/SPEC.md](docs/SPEC.md) and [docs/DECISIONS.md](docs/DECISIONS.md), because
the self-corrections are the evidence.

Bit-identity across different GPUs, drivers, CUDA versions or Ollama builds is
**not** claimed: each can change kernel selection and therefore float reduction
order. Deviating from a pin self-marks the artifact provisional.

### Tier 3 — containerized reference environment

Queued, not forced: both variables that would have made it mandatory (flash
attention, the prompt cache) proved controllable in-process. Trigger conditions
in [docs/FUTURE.md](docs/FUTURE.md).

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

Pre-alpha, and the library is the part that isn't built. The eval harness runs
today with two baselines (`no_memory`, `full_history`); the write-side policy
that `Memory` promises — extraction, dedup, conflict resolution, decay — is
specified but not implemented, and the default embedder is a numpy hashing
placeholder. `docs/SPEC.md` describes the target, not the current code.

See [docs/SPEC.md](docs/SPEC.md) for the contract,
[docs/DECISIONS.md](docs/DECISIONS.md) for locked decisions, and
[docs/FUTURE.md](docs/FUTURE.md) for what is deliberately deferred.

## License

[Apache-2.0](LICENSE).
