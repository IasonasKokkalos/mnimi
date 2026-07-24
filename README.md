# MNIMI

Embeddable, local-first agent memory. One SQLite file, zero infra.

The number that matters is the benchmark number. Everything in this repo is
judged by whether it moves the table below.

## LongMemEval (longmemeval_s, ~500 questions)

Reader model held fixed across rows. `no_memory` is the floor (answer with no
history); `full_history` is the ceiling (stuff the entire conversation history
into context). mnimi has to beat the naive baselines while using a fraction
of the tokens — that is the entire bet.

| System         | Overall | single-session-user | single-session-assistant | single-session-preference | temporal-reasoning | knowledge-update | multi-session |
| -------------- | ------: | ------------------: | -----------------------: | ------------------------: | -----------------: | ---------------: | ------------: |
| no_memory      |     TBD |                 TBD |                      TBD |                       TBD |                TBD |              TBD |           TBD |
| full_history   |     TBD |                 TBD |                      TBD |                       TBD |                TBD |              TBD |           TBD |
| naive_rag      |       — |                   — |                        — |                         — |                  — |                — |             — |
| **mnimi**      |       — |                   — |                        — |                         — |                  — |                — |             — |

Numbers are produced by the eval harness, which is the source of truth:

```bash
pip install -e ".[eval]"
export ANTHROPIC_API_KEY=...           # reader + judge run on the Anthropic API
python -m evals --system full_history --limit 10   # smoke run
python -m evals --system no_memory                 # full ~500-question run
```

## API

```python
from mnimi import Memory

mem = Memory(db_path="agent.db", embedder=embedder)
mem.add(messages, user_id="u1")              # write path
mem.recall(query, user_id="u1")              # raw retrieval -> list[MemoryRecord]
mem.get_context(query, user_id="u1")         # assembled context -> str
mem.consolidate(user_id="u1")                # merge / resolve conflicts / decay
```

Four methods. The surface stays thin on purpose; the depth lives behind `add`
and `consolidate`. Core install pulls in only `sqlite-vec` and `numpy` — no
server, no external services, one file on disk.

## Status

Pre-alpha. The library API is stubbed; the eval harness runs today. See
[docs/SPEC.md](docs/SPEC.md) for the contract, [docs/DECISIONS.md](docs/DECISIONS.md)
for locked decisions, and [CHANGELOG.md](CHANGELOG.md).

## License

[Apache-2.0](LICENSE).
