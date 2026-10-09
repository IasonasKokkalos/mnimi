# examples/

## `chat.py`: a chat loop with memory

The smallest real use of the library: each turn fetches context with `get_context`, puts it
in the system prompt, gets a reply from a chat backend, and stores both turns with `add()`.
One timestamp per launch, computed here (a day is a session, the benchmark's shape); the
library never reads a clock.

```
python examples/chat.py --embedder hashing --extractor none              # nothing to install beyond mnimi + a backend
python examples/chat.py --backend ollama --embedder bge --extractor cpu  # the real embedder and the CPU extractor
```

| flag | values | default |
| --- | --- | --- |
| `--db` | path of the SQLite store | `./agent.db` |
| `--user` | the `user_id` the records are stored under | `me` |
| `--backend` | `ollama` (model `qwen2.5:7b-instruct`, env `OLLAMA_HOST`) or `openai` (model `gpt-4o-mini`, env `OPENAI_API_KEY`); `--model` overrides | `ollama` |
| `--embedder` | `bge` (`BAAI/bge-small-en-v1.5`, the `[embed]` extra) or `hashing` (numpy only, for smoke tests) | `bge` |
| `--extractor` | `cpu` (the recipe, `decode={**DECODE, "n_gpu_layers": 0}`), `gpu` (the pinned extractor), `none` (rounds only, no fact records) | `none` |

Commands inside the loop: `/recall <q>` (each hit's score, salience, kind and first line),
`/facts` (the fact records stored from the last round), `/export` (writes `memory.md` beside
the DB through `mnimi export --format md`), `/consolidate` (the conflict and decay passes, with
their INFO lines), `/quit`. The `mnimi.memory` logger prints `superseded …`, `routed to
conflict …` and `decayed …` as they happen.

### What to install

These are the example's dependencies, not the library's (the core stays `sqlite-vec` + `numpy`).

| backend / option | install |
| --- | --- |
| Ollama backend | `pip install "mnimi[embed,extract]" ollama`, plus a running Ollama with the model pulled (`ollama pull qwen2.5:7b-instruct`) |
| OpenAI backend | `pip install "mnimi[embed,extract]" openai` and `OPENAI_API_KEY` in the environment |
| `--embedder hashing --extractor none` | `pip install mnimi` and one backend client |
| `--extractor cpu` | the `[extract]` extra with a CPU build of `llama-cpp-python==0.3.35`; the compiler-free install line and the measured seconds per round land here with MERGED-PLAN T5 |
| `--extractor gpu` | the `[extract]` extra built with CUDA (docs/DECISIONS.md "Extractor runtime") |

The extractor's cache (`extract-cache-<cpu|gpu>.sqlite`) is written beside the DB. A store
remembers its embedder and extractor pins in `memory_meta`: reopening it under different pins is
refused, which is the guard working (test 5 below).

### The week's six tests (LAUNCH §6.1)

Run over a week on one DB; write the date and what you saw in the last two columns. The readings
go to `mnimi docs/LAUNCH.md` §6.1.

| # | test | expected | date | reading |
| --- | --- | --- | --- | --- |
| 1 | three facts on day 1; on day 3 ask for one indirectly | recall + reading: the right round in the context, the right answer | | |
| 2 | contradict one of them on day 4 | a `superseded …` INFO line; `/export` shows the old fact struck through | | |
| 3 | a relative date ("dentist next Thursday"); later ask "when is my dentist" | `valid_time` resolved from the session `ts` (needs `--extractor cpu` or `gpu`: `valid_time` lives on fact records); the time-aware term ranks the round | | |
| 4 | an aside buried in a long assistant reply; ask about it later | the known miss (NEXT-STEPS §3 option 1): expect it missed, and say so in the launch post | | |
| 5 | restart on the same DB; then launch with the other `--embedder` | the guard accepts the first and refuses the second (`MemoryMetaError`) | | |
| 6 | `/consolidate` twice in a row | the second pass changes nothing (idempotent; no new `decayed …` lines) | | |
