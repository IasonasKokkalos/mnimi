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
`gpt-4o-2024-08-06`:

| System       | Overall | ssu | ssa | ssp |  tr |  ku |  ms |
| ------------ | ------: | --: | --: | --: | --: | --: | --: |
| no_memory    |  1/20   | 1/4 | 0/4 | 0/3 | 0/3 | 0/3 | 0/3 |
| full_history |  4/20   | 1/4 | 1/4 | 0/3 | 0/3 | 2/3 | 0/3 |

The harness marks both runs **PROVISIONAL — not publishable**: the reader prompt
is still `plain-prose-v1`, and the JSON + Chain-of-Note prompt that the pinned
configuration calls for has not landed. A 20-question slice also carries a
sampling error far larger than any effect worth reporting. Quote these numbers
nowhere.

## Running the harness

The harness is the source of truth for every claim in this repo. The reader runs
locally through Ollama; only the judge touches an API.

```bash
pip install -e ".[eval]"

ollama pull qwen2.5:1.5b-instruct-q4_0     # the pinned reader
ollama serve

echo 'OPENAI_API_KEY=sk-...' >> .env       # judge only (gpt-4o-2024-08-06)

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
digest, decode pins, and both prompt hashes, so a run is self-identifying.

## Reproducibility

Three claims at three strengths, published separately because conflating them
overstates what the artifacts prove.

- **Tier 1 — auditable.** Given a committed `predictions.jsonl`, anyone can
  re-derive the published score on any machine: no GPU, no Ollama, no dataset
  download, seconds to run.

  ```bash
  python -m evals --system <name> --limit <N> --stage judge \
      --run-dir runs/<system>__<N>q --judge-model gpt-4o-2024-08-06
  ```

  This proves the *scoring*, not the generation. It is an audit, not a
  reproduction.
- **Tier 2 — reproducible on comparable hardware.** `--stage predict` regenerates
  the predictions given the pins (`num_gpu=99`, `num_batch=512`, `top_k=1`,
  `seed=0`, temperature 0), within a stated tolerance: llama.cpp dispatches to
  different SIMD paths across CPUs and reduces floats in different orders, so
  divergence across dissimilar hardware is expected, not a defect. Deviating from
  a decode pin self-marks the artifact as non-reproducible.
- **Tier 3 — pinned reference image.** Deferred, with trigger conditions in
  [docs/FUTURE.md](docs/FUTURE.md).

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
[docs/DECISIONS.md](docs/DECISIONS.md) for locked decisions,
[docs/FUTURE.md](docs/FUTURE.md) for what is deliberately deferred, and
[CHANGELOG.md](CHANGELOG.md).

## License

[Apache-2.0](LICENSE).
