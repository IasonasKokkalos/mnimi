# examples/

## `chat.py`: a chat loop with memory

The smallest real use of the library: each turn fetches context with `get_context`, puts it
in the system prompt, gets a reply from a chat backend, and stores both turns with `add()`.
One timestamp per launch, computed here (a day is a session, the benchmark's shape); the
library never reads a clock.

```
python examples/chat.py                                                  # the defaults: Ollama, BGE, the CPU extractor
python examples/chat.py --embedder hashing --extractor none              # nothing to install beyond mnimi + a backend
```

| flag | values | default |
| --- | --- | --- |
| `--db` | path of the SQLite store | `./agent.db` |
| `--user` | the `user_id` the records are stored under | `me` |
| `--backend` | `ollama` (model `qwen2.5:7b-instruct`, env `OLLAMA_HOST`) or `openai` (model `gpt-4o-mini`, env `OPENAI_API_KEY`); `--model` overrides | `ollama` |
| `--embedder` | `bge` (`BAAI/bge-small-en-v1.5`, the `[embed]` extra) or `hashing` (numpy only, for smoke tests) | `bge` |
| `--extractor` | `cpu` (the recipe, `decode={**DECODE, "n_gpu_layers": 0}`, measured below), `gpu` (the pinned extractor, needs the CUDA build), `none` (rounds only, no fact records) | `cpu` |

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
| `--extractor cpu` | the compiler-free line below: the project's own CPU wheel of `llama-cpp-python==0.3.35` (PyPI carries only the sdist, which needs a C++ compiler) |
| `--extractor gpu` | the `[extract]` extra built with CUDA (docs/DECISIONS.md "Extractor runtime") |

```
pip install "llama-cpp-python==0.3.35" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install "mnimi[embed,extract]"
```

The first line installs `llama_cpp_python-0.3.35-py3-none-win_amd64.whl` (the index also carries
manylinux x86_64 / aarch64 and macOS arm64 wheels of 0.3.35); the second then finds the pin
satisfied and installs the rest. Order matters: the other way round, pip builds the sdist.

### The CPU extractor, measured (2026-10-10)

`--extractor cpu` passes `decode={**DECODE, "n_gpu_layers": 0}` to the public
`QwenLlamaExtractor(decode=)` keyword and nothing else: `flash_attn=True` holds on the CPU
build (`llama_context: flash_attn = enabled` in the load log), so one key is the whole
difference. The model (`Qwen3-1.7B-Q8_0.gguf`, 1.8 GB, downloaded by revision on first use) runs
in-process under the same grammar and the same prompt hash as the GPU pin; the store it builds
carries its own `extractor_decode_hash`, so the `memory_meta` guard refuses it under the GPU
pins and vice versa (`tests/test_cpu_profile.py`, no model).

Measured on the 40 dev rounds of `evals.probes.extractor_bench --dev-set 40` (rounds outside the
benchmark slice; mean 1,583 prompt tokens, 237 completion tokens), on an AMD Ryzen 7 PRO 8845HS
(8 cores, `n_threads=8`), the CPU wheel, Python 3.12, no GPU:

| profile | seconds / round mean | p50 | max | min | tok/s overall | truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu` (this recipe) | 21.0 | 19.6 | 46.7 | 7.6 | 87 | 0 / 40 |
| `gpu` (the paper's pin, RTX 1000 Ada, DECISIONS "Extractor runtime") | 2.7–3.4 | | | | | |

The first 20 of those rounds in a second fresh process: **20/20 raw outputs byte-identical**
(mean 22.7 s / round there; a descriptive reading, never a stability claim). Hashes: `DECODE` → `ba1a81ab…b8241`, the CPU dict →
`40552b1e…6d65`. One disclosure: the `extractor_runtime` row is a module constant naming the
CUDA build, so a CPU store carries that string too and is told apart by its decode hash alone.

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
