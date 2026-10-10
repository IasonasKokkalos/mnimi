# mnimi

[![PyPI](https://img.shields.io/pypi/v/mnimi.svg)](https://pypi.org/project/mnimi/)
[![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)](pyproject.toml)
[![CI](https://github.com/IasonasKokkalos/mnimi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/IasonasKokkalos/mnimi/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) — the preprint lands 2026-11-06 -->

mnimi is an embeddable, local-first memory layer for LLM agents: one SQLite file (`sqlite-vec`),
two dependencies, no server. Raw chat turns go in; a small pinned local model extracts facts at
write time; deterministic screens handle duplicates, contradictions and supersession; nothing on
the read path calls a model. All time is logical (the `ts` you hand each message), so the same
store gives the same answers on any day. Every memory is readable as Markdown. On LongMemEval-S
(500 questions, gpt-4o as reader) it reads 85.8 %, paired and audited: every published row
re-grades with one command.

## The number

The gpt-4o family on all 500 questions of LongMemEval-S, one reader prompt for every question
type, retrieval `k=10` and per-round ingestion identical across the rows that retrieve:

| System | Correct / 500 | Wilson 95 % | mean fed tokens | role | run_id |
| --- | ---: | :---: | ---: | --- | --- |
| no_memory | 31 (6.2 %) | [4.4, 8.7] | 133 | the floor: the question alone | `no_memory__500q_gpt4o` |
| full_history | *cited: 64.0 %* | — | — | *cited (LongMemEval Fig. 3b, GPT-4o + Chain-of-Note), never run here* | — |
| naive_rag | 373 (74.6 %) | [70.6, 78.2] | 4,685 | rounds stored verbatim, cosine top-10: the bar | `naive_rag__500q_gpt4o` |
| **mnimi** | **429 (85.8 %)** | **[82.5, 88.6]** | 5,498 | the shipped configuration | `mnimi__500q_gpt4o_p6time` |
| oracle | 459 (91.8 %) | [89.1, 93.9] | 5,771 | the evidence-availability bound: the reader is handed the evidence sessions | `oracle__500q_gpt4o` |

**Provenance.** Reader and judge `gpt-4o-2024-08-06` over the OpenAI API, temperature 0; reader
prompt `mnimi-con-v1`, judge prompt `longmemeval-paper-v3`; harness commits `f07c24d` (no_memory,
naive_rag, oracle) and `f0a7de3` (mnimi), clean trees, `provisional: []` on all four. Counts and
intervals are copied from [`results/paper/tables.md`](results/paper/tables.md) (T1, T5, T8), which
`python -m evals.paper_tables` exports from the committed predictions; mean fed tokens is the mean
`reader_prompt_tokens` over each arm's `predictions.jsonl`. mnimi against naive_rag on the same
500 questions: b=77, c=21, exact McNemar p = 1.1 × 10⁻⁸ (T3). Paired, never ranked.

Two caveats. The reader and the judge are one model snapshot (LongMemEval's own pairing); a second
judge re-graded every arm and every pair of record keeps its sign. "Auditable" is not
"reproducible": every row re-grades from its committed predictions (§ Verify the number), while a
fresh run of the same configuration reproduces the score within 12/500 flips and not the text.
The full record of every run, the third-party rows and their disclosures, is
[`results/published/README.md`](results/published/README.md).

## The problem

- An agent forgets everything between sessions, or stuffs its window with history until the
  evidence it needs is truncated away.
- Routing every write through an LLM is slow, costs money per turn, and leaves no auditable
  trail of why a memory was kept, merged or dropped.
- Hosted memory is someone else's database: your users' conversations leave the machine.
- You cannot read what your agent remembers, so you cannot check it, correct it or delete it.

## Quick install

```
pip install mnimi                                   # the library: rounds stored verbatim, no model
```

```
pip install "llama-cpp-python==0.3.35" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install "mnimi[embed,extract]"                  # the full write path on a CPU, no compiler
```

```
git clone https://github.com/IasonasKokkalos/mnimi && cd mnimi && pip install -e ".[embed,extract,dev]"
```

| extra | what it adds | download |
| --- | --- | --- |
| *(none)* | `Memory` over `sqlite-vec` + `numpy`; `HashingEmbedder` (numpy-only placeholder, for tests) | — |
| `embed` | `BgeSmallEmbedder`: `BAAI/bge-small-en-v1.5` through ONNX Runtime, revision-pinned | ≈ 130 MB model, 13.8 MB onnxruntime wheel |
| `extract` | `QwenLlamaExtractor`: `Qwen/Qwen3-1.7B-GGUF` Q8_0 through llama-cpp-python 0.3.35, revision-pinned | 1.8 GB model, 7.1 MB CPU wheel |

`[extract]` pins `llama-cpp-python==0.3.35`, which PyPI ships only as a source distribution; the
project's own index carries prebuilt CPU wheels (Windows x64, manylinux x86_64 / aarch64, macOS
arm64), so install that first and nothing compiles. A CUDA build of the same version is the GPU
path (`docs/DECISIONS.md` "Extractor runtime"). Both are the same pinned model under the same
grammar and prompt; the CPU profile differs in one decode key and is measured at 21.0 s per round
on an 8-core laptop CPU against 2.7–3.4 s on the development GPU
([`examples/README.md`](examples/README.md)).

## 60-second quickstart

A message is `{"role": "user" | "assistant", "content": str, "ts": "<ISO timestamp>"}`. `ts` is
the session's timestamp and the only clock the library knows: dedup scope, conflict ordering,
decay and the time-aware term all run on the `ts` values the store has seen, never on the wall
clock, so a store built today and replayed next year scores identically. A message without `ts`
is refused.

```python
from mnimi import Memory
from mnimi.embeddings import BgeSmallEmbedder
from mnimi.extract.cache import CachedExtractor
from mnimi.extract.llama import DECODE, QwenLlamaExtractor

extractor = CachedExtractor(QwenLlamaExtractor(decode={**DECODE, "n_gpu_layers": 0}), "extract-cache.sqlite")
mem = Memory("agent.db", BgeSmallEmbedder(), extractor=extractor)
mem.add([{"role": "user", "content": "I moved to Boston last month. My dentist is Dr. Lee.", "ts": "2026-10-01T00:00:00"},
         {"role": "assistant", "content": "Noted: you live in Boston and Dr. Lee is your dentist.", "ts": "2026-10-01T00:00:00"}], "u1")
mem.add([{"role": "user", "content": "I live in Cambridge now.", "ts": "2026-10-08T00:00:00"},
         {"role": "assistant", "content": "Updated: you live in Cambridge.", "ts": "2026-10-08T00:00:00"}], "u1")
print(mem.get_context("where do I live?", "u1"))
print(mem.export("u1"))
```

`decode={**DECODE, "n_gpu_layers": 0}` is the CPU profile; drop the keyword on a CUDA build. The
first run downloads both models by pinned revision and takes about a minute on a laptop CPU; the
cache file beside the DB makes every later `add()` of the same round free. Run as pasted, it
prints:

```
[Session date: 2026-10-01T00:00:00]
facts:
- The user's dentist is Dr. Lee.
user: I moved to Boston last month. My dentist is Dr. Lee.
assistant: Noted: you live in Boston and Dr. Lee is your dentist.
[Session date: 2026-10-08T00:00:00]
facts:
- The user lives in Cambridge now.
user: I live in Cambridge now.
assistant: Updated: you live in Cambridge.
# memory export
user_id: u1
now_logical: 2026-10-08T00:00:00
records: 5 (rounds 2, facts 3)

[Session date: 2026-10-01T00:00:00]
user: I moved to Boston last month. My dentist is Dr. Lee.
assistant: Noted: you live in Boston and Dr. Lee is your dentist.
  fact: The user moved to Boston last month.  salience 0.0000  (initial 1.0000)  superseded  valid_time 2026-09  pair user|lives in
  fact: The user's dentist is Dr. Lee.  salience 1.0000  pair user|dentist

[Session date: 2026-10-08T00:00:00]
user: I live in Cambridge now.
assistant: Updated: you live in Cambridge.
  fact: The user lives in Cambridge now.  salience 1.0000  pair user|lives in  supersedes 2
```

The first fact of the first session is gone from the context: `I live in Cambridge now.` superseded
`moved to Boston` on the key `user|lives in` (a functional predicate, two values, the later session
wins), and a superseded fact neither ranks nor renders. The export below still shows it, marked.

## What it does

The write path, `add(messages, user_id)`, per round of one user and one assistant turn:

1. **Store the round** verbatim as one record (its embedded text is frozen; its turns are kept for
   rendering), after an exact-match and one cosine dedup probe against the session.
2. **Extract** facts with the pinned local model, under a grammar, through an on-disk cache; six
   deterministic pre-filter rules skip rounds with nothing to extract. Relative dates in a fact
   ("next Thursday") resolve against the message's `ts`, never by the model.
3. **Screen** each fact against the facts already stored: an exact collapse, a frozen negation
   lexicon, a value-substitution check over normalized triples, a token-entropy gate, then one
   cosine probe. The screens only ever keep a pair apart; they never drop what verbatim storage
   would have kept.
4. **Supersede**: two facts on one subject-predicate key conflict only under a rule (opposite
   polarity, a functional predicate with two values, a changed number). The later effective time
   wins; the loser's salience goes to 0 and the winner points at it. Logged at INFO as
   `superseded …`.

The read path, `recall` / `get_context(query, user_id)`:

1. **Embed the bare question** (a `[Current date: <ts>]` prefix, if you pass one, is parsed for
   the question's own time window and stripped before embedding).
2. **Select the top-k rounds by score** over an exact candidate pool:
   `score = (w_sim · relevance + w_rec · recency) · salience + w_time · time_match`. Superseded
   facts (salience 0) neither rank nor render.
3. **Render** the selected rounds oldest-first through the one renderer: a `[Session date: …]`
   header per session, `user:` / `assistant:` lines, the round's facts under a `facts:` header.

`consolidate(user_id)` re-runs the conflict pass over every active pair and then the decay pass
(`salience = max(floor, initial · 0.5^(days / half_life))` on logical days). Decay is built,
tested and **off** in the shipped configuration: the pre-registered ablation on all 500 questions
read it at −6.2 points (`mnimi__500q_gpt4o` 422 → `mnimi__500q_gpt4o_decay` 391, b=19, c=50,
p = 2.4 × 10⁻⁴; T3), because it multiplies down exactly the old evidence the benchmark asks about.
No benchmark arm calls `consolidate()` unless asked.

## API

```python
from mnimi import Memory, MemoryConfig
mem = Memory(db_path, embedder, config=MemoryConfig(), extractor=None)
```

| method | what it does | returns |
| --- | --- | --- |
| `add(messages, user_id)` | the write path above, one round at a time; `messages` is a list of `{role, content, ts}` dicts | `None` |
| `recall(query, user_id)` | the top-`k` rounds by score, best first, with their component scores | `list[ScoredRecord]` (`.score`, `.relevance`, `.recency`, `.salience`, `.record`) |
| `get_context(query, user_id)` | the recalled rounds rendered oldest-first as reader context | `str` |
| `consolidate(user_id)` | the conflict pass, then the decay pass; idempotent | `None` |
| `export(user_id)` | the whole store as readable text, superseded facts shown and marked; no model, no clock | `str` |

Five methods; depth lives behind `add` and `consolidate`. `MemoryConfig` is a frozen dataclass of
eighteen fields, every threshold the library reads; the ones a user touches:

| field | default | what it controls |
| --- | --- | --- |
| `top_k` | `10` | rounds returned by `recall` and rendered by `get_context` |
| `dedup_cosine_threshold` | `0.95` | the one cosine probe that collapses a near-duplicate round or fact |
| `dedup_scope` | `"session"` | whether the cosine probe looks at the session or the whole store |
| `conflict_resolution` | `True` | the screens, the entropy gate and supersession; `False` is dedup only |
| `time_weight` | `0.05` | the time-aware term, read only when a query carries a `[Current date: …]` prefix |
| `decay_half_life_days` | `30.0` | the decay pass's half-life in logical days |
| `decay_floor` | `0.15` | the salience a decayed record never falls below (it is down-ranked, never excluded) |
| `render_unit` | `"round+facts"` | what a rendered round carries: `round+facts`, `turns` (no facts) or `facts` |

The other fields (`ranking`, `active_only`, `recall_min_relevance`, `salience_weights`,
`rerank_pool`, `dedup_entropy_gate`, `query_instruction`, `chunk_tokens`, `chunk_overlap`,
`render_format`) are documented in [`src/mnimi/config.py`](src/mnimi/config.py) and sit at the
values the benchmark was run with.

## CLI

| command | description |
| --- | --- |
| `mnimi export <db> <user_id> [-o FILE] [--format md\|text]` | dump one user's store as Markdown (default) or as `export()`'s text; no model is loaded |
| `mnimi-mcp` | the MCP server over one store, installed by the `mnimi-mcp` package (§ Use it from an agent) |

## Read your memory

```
mnimi export agent.db u1 -o memory.md
```

One heading per session, each round as a quoted block, its facts as bullets, a superseded fact
struck through with the id that replaced it. The quickstart's store exports as:

```markdown
# memory export

- user_id: u1
- now_logical: 2026-10-08T00:00:00
- records: 5 (rounds 2, facts 3)

## 2026-10-01T00:00:00

> [Session date: 2026-10-01T00:00:00]
> user: I moved to Boston last month. My dentist is Dr. Lee.
> assistant: Noted: you live in Boston and Dr. Lee is your dentist.

- ~~The user moved to Boston last month. (**2026-09**)~~ (superseded by #5)
- The user's dentist is Dr. Lee.

## 2026-10-08T00:00:00

> [Session date: 2026-10-08T00:00:00]
> user: I live in Cambridge now.
> assistant: Updated: you live in Cambridge.

- The user lives in Cambridge now. → supersedes #2
```

## Use it from an agent

A chat loop with memory, one timestamp per launch (a day is a session), the library's INFO lines
on, and `/recall`, `/facts`, `/export`, `/consolidate` commands:

```
python examples/chat.py                                   # Ollama, BGE, the CPU extractor
python examples/chat.py --embedder hashing --extractor none   # nothing beyond mnimi and a chat backend
```

The MCP server (`integrations/mcp/`, the package `mnimi-mcp`) exposes `remember`, `recall`,
`context`, `export` and `consolidate` as tools and `memory://export` as a resource over stdio. It
stamps every stored message with the server's calendar day, so the clock stays in the application;
one store, one user, from the environment:

```
pip install mnimi-mcp --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
claude mcp add --scope user mnimi -e MNIMI_DB=/absolute/path/to/agent.db -- mnimi-mcp
```

`MNIMI_USER_ID` (default `me`), `MNIMI_EXTRACTOR` (`cpu` | `gpu` | `none`, default `cpu`) and
`MNIMI_EMBEDDER` (`bge` | `hashing`, default `bge`) are the other settings;
[`integrations/mcp/README.md`](integrations/mcp/README.md) has the tool table.

## How it compares

A feature table, source-verified, no numbers (the numbers are in § The number and in
[`results/published/README.md`](results/published/README.md) § Phase 8). Each cell names where it
was read.

| feature | mnimi | naive RAG (the harness's `naive_rag` arm) | Mem0 OSS 2.2.1 | OMEGA 1.5.x |
| --- | --- | --- | --- | --- |
| input | raw chat turns, both roles (`src/mnimi/memory.py`) | raw chat turns (`evals/systems/naive_rag.py`) | raw chat turns; the LLM condenses them to facts framed around the user (`mem0/memory/main.py:881`) | agent-curated typed records; "fact extraction" is regex over distilled content (`docs/SPEC.md` § Extraction, `bridge.py:652`) |
| LLM at write | one pinned local 1.7B model, grammar-constrained, cached (`src/mnimi/extract/llama.py`) | none | yes: an API model decides ADD / UPDATE / DELETE / NONE per fact (`mem0/configs/prompts.py:205-320`; gpt-4o-mini in our run, `configs/mem0_competitor.json`) | none (`docs/SPEC.md` § Extraction) |
| LLM at read | none (`CLAUDE.md`, `src/mnimi/ranking.py`) | none | none by default; an optional reranker (`mem0/memory/main.py:505-516`) | none; an optional ONNX cross-encoder reranker (`docs/SPEC.md` § Deferred, `reranker.py`) |
| dedup | exact collapse + one cosine probe at 0.95, per round and per fact (`src/mnimi/memory.py`) | none | the write LLM's NONE / UPDATE decision (`prompts.py`) | exact `content_hash` at write, offline lexical Jaccard 0.6 compaction (`docs/SPEC.md` § Dedup, `schema.py:309`) |
| contradictions | three deterministic rules over normalized triples; loser superseded, both kept (`src/mnimi/conflict/`) | none | the write LLM's UPDATE / DELETE decision (`prompts.py:246-290`) | a four-signal heuristic (`docs/SPEC.md` § Dedup, `contradictions.py`) |
| decay | built, measured at −6.2 points on LongMemEval, off (`src/mnimi/decay.py`; T3) | none | platform-only; the OSS build raises on `decay=True` (`mem0/memory/main.py:467-470`) | on, wall-clock days since access, floors 0.15 / 0.35 (`docs/SPEC.md` § Logical time, `_base.py:226`) |
| time | logical: the message `ts` is the only clock; relative dates resolved against it (`src/mnimi/decay.py`, `extract/resolver.py`) | session date as a rendered header | `created_at` from the wall clock, not read by search; the extraction prompt grounds relative dates on the machine date (`results/published/README.md` § Phase 8) | wall-clock `created_at` and access times (`results/published/README.md` § Phase 8) |
| storage | one SQLite file with `sqlite-vec`; a `memory_meta` table that refuses a store built under other pins (`src/mnimi/store.py`) | the same file format | a vector store (Qdrant in our run) plus a SQLite history DB (`configs/mem0_competitor.json`) | SQLite with vector, FTS5 and RRF in-process (`docs/SPEC.md` § Deferred) |
| export | `export()` and `mnimi export`: the whole store as text or Markdown (`src/mnimi/export.py`, `src/mnimi_cli/`) | — | `get_all()` returns the memories; no file export (`mem0/memory/main.py:1269`) | not read |

OMEGA's cells are SPEC's source-verified notes of 2026-07-20 (v1.5.5); the harness ran 1.5.17.
Both third-party systems were run through this harness under its pins, paired against the same
rows: readings, never a rank, and not replications of their published numbers.

## Architecture

```
your application                      (holds the clock: hands every message its ts)
      │  add(messages, user_id)                 get_context(query, user_id)
      ▼                                                 ▲
┌───────────────────────────── mnimi.Memory ─────────────────────────────┐
│  extract/   the pinned model + cache, prefilter, resolver (write only)  │
│  conflict/  lexicon, normalize, screens, supersede (deterministic)      │
│  ranking.py exact top-k by score · decay.py logical time, the pass     │
│  memory.py  the one renderer: RENDER_TEMPLATE, round+facts             │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
                 store.py ── one SQLite file (sqlite-vec, memory_meta)
```

## Storage and footprint

- **The DB** is the path you pass; one file, WAL mode. `memory_meta` holds seventeen rows written
  once at creation (embedder name, revision and dimension; the embed-template hashes; the chunk
  pair; the five extractor pins or `"none"`; the frozen lexicon, conflict, prefilter, resolver and
  decay hashes) and checked at every open. A mismatch raises `MemoryMetaError` before any query
  runs, so two configurations can never share a store by accident.
- **The extraction cache** is a second SQLite file keyed by the extractor's pins: one row per
  round the model has seen, so a round is extracted once. `examples/chat.py` and `mnimi-mcp`
  put it beside the DB; the harness keeps it under `.cache/extract/`. The LongMemEval cache,
  attached to the `paper-v1` GitHub release, is 95,846 rows, 140,038,144 bytes, sha256
  `ce4a1ab33d661c2c1128974c5d4ea2f69bd238b49085691c600b07a82db4e39e`.
- **Sizes.** The quickstart's two rounds make a 3.2 MB store (384-dim float32
  vectors, the round records and their facts); the models are 130 MB (BGE) and 1.8 GB (Qwen3
  Q8_0) in the Hugging Face cache, downloaded once by revision.

## Verify the number

Every published row re-grades from its committed predictions, which carry the question and the
gold answer inline. No GPU, no dataset, no model weights; an OpenAI key for the judge, cents per
arm:

```
pip install -e ".[eval]"
python -m evals --stage judge --predictions results/published/mnimi__500q_gpt4o_p6time/predictions.jsonl
```

It recomputes the score and prints `MATCHES`, `WITHIN RE-GRADE` or `DIVERGES` against the
committed result. This is **Tier 1, auditable**: it proves the scoring, not that the predictions
came from the pipeline the pins claim. **Tier 2** is a fresh run of the shipped configuration
(`python -m evals --config configs/<run_id>.json`, about $4 of API): on the gpt-4o family it is
"score reproducible within 12/500 flips; text not reproducible" — measured 277/500 answer texts
changed, prompt tokens changed on 0/500, 429 both times, b=6, c=6
(`results/published/mnimi__500q_gpt4o_p6time_drift_2026-09-26`). The judge itself flips about 1 %
of rows between re-grades, so no absolute difference below that is interpretable; paired
comparisons are unaffected. Every command of the paper's harness is in
[`results/published/README.md`](results/published/README.md).

## Run your system through the harness

A system is a class with four methods, in [`evals/base.py`](evals/base.py):

```python
class MemorySystem(ABC):
    name: str
    def reset(self) -> None: ...                       # a fresh store per question
    def add(self, messages: list[dict]) -> None: ...   # one history, {role, content, ts} dicts
    def get_context(self, query: str) -> str: ...      # what the reader sees
    def set_question_date(self, question_date: str | None) -> None: ...  # optional
```

Render through `mnimi.memory.render_turns` so the reader sees the one format every arm uses
(date granularity and speaker labels were a measured confound), register it in
`evals/__main__.py`, and run `python -m evals --system <name> --limit 500 --reader-transport
openai --batch --purpose "..." --claim none`. Third-party packages go through the contract in
`evals/systems/competitor.py` (their own environment, a 5,364-token context budget, their own ids
as `retrieved_ids`); the Mem0 and OMEGA adapters are the worked examples. Adapters welcome: open a
pull request with the adapter and its config, and the run is paired against the same rows.

## Paper

The library the paper describes is tag `paper-v1`: `src/mnimi/` is that tag's bytes, byte for
byte, in every release of the 3.x line (a digest test enforces it). The preprint lands
2026-11-06; its title, arXiv id and BibTeX go here then, and `CITATION.cff` carries the software
citation now.

## Troubleshooting

**`MemoryMetaError: memory_meta mismatch - this store was created under a different pin`.** The
DB was built with another embedder, extractor profile or library era (the message names the row).
Point at a fresh file, or open it with the pins it was built with; the library never upgrades a
store in place.

**`ValueError: record has no timestamp; pass the message ts as created_at`.** A message lacked
`ts`. Every message needs `{"role", "content", "ts"}`: the library has no clock, so the
application must hand it one (a session's start, the calendar day).

**`RuntimeError: this llama-cpp-python build cannot offload to the GPU, but the decode pins ask
for n_gpu_layers>0`.** You built the extractor with the default `DECODE` on a CPU build. Use the
CPU profile, `QwenLlamaExtractor(decode={**DECODE, "n_gpu_layers": 0})`, or install the CUDA
build of 0.3.35. The two profiles write different `extractor_decode_hash` rows, so their stores
are not interchangeable.

**`sqlite3.OperationalError` at `enable_load_extension`.** Your Python's SQLite was built
without loadable-extension support (some macOS and conda builds). Use a Python whose `sqlite3`
module allows `enable_load_extension(True)`, for example the python.org or `uv` builds.

## Development

```
pip install -e ".[dev]"
PYTHONPATH="src;." python -m pytest -q          # 630 tests; 4 need the [embed]/[extract] extras, 10 the mcp package
python -m ruff check .
```

`src/mnimi/` is frozen at `paper-v1`'s bytes: `evals/freeze.py` holds the digest and
`tests/test_paper_freeze.py` fails on any edit, so release work lives in `src/mnimi_cli/`,
`examples/`, `integrations/` and `evals/`. Lifting the freeze is a dated `docs/DECISIONS.md` entry
and a new digest, and a library fix the paper needs is `paper-v1.1` with every paper arm rerun.
[`CLAUDE.md`](CLAUDE.md) is the working contract (the invariants that silently invalidate a
number); `docs/SPEC.md` § "v1 as built" is the exact shipped state.

## Status and scope

- v3.0.0 is the developer release: the paper's library, the export CLI, the chat example, the CPU
  recipe and the MCP server. Alpha.
- In scope is write-side policy: extraction, screens, supersession, decay, ranking, retrieval.
  Out of scope: a UI, a chat frontend, a hosted service.
- Deliberately not built, each with its re-open trigger in [`docs/FUTURE.md`](docs/FUTURE.md): a
  context token budget, `raw` spans in the rendered block, hybrid FTS5 retrieval, a recency weight
  above 0, pinned records.
- The contract is [`docs/SPEC.md`](docs/SPEC.md); the dated rulings are
  [`docs/DECISIONS.md`](docs/DECISIONS.md).

## License

[Apache-2.0](LICENSE).
