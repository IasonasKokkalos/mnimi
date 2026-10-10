# mnimi-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server over one
[mnimi](https://github.com/IasonasKokkalos/mnimi) store: five tools and one resource, stdio
transport, one SQLite file. Give Claude Code (or any MCP host) a long-term memory that lives
on your disk, writes facts with a small local model, never calls a model when it reads, and
can be read whole as Markdown.

The server is its own package so the library keeps its two dependencies (`sqlite-vec`,
`numpy`). It depends on `mnimi[embed,extract]` (the BGE embedder and the Qwen3-1.7B
extractor) and on the `mcp` SDK (written and tested against `mcp` 2.3.0, 2026-10-02).

## Install

`mnimi[extract]` pins `llama-cpp-python==0.3.35`, which PyPI ships only as a source
distribution (it needs a C++ compiler). The project's own index carries a prebuilt CPU wheel,
so pass it on the install line and nothing is compiled:

```
pip install mnimi-mcp --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Before `mnimi` 3.0.0 is on PyPI, install both from a checkout of the repository (the
`release/v3` branch), into a venv of its own:

```
pip install -e . -e integrations/mcp --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Either line puts a `mnimi-mcp` console script on the venv's `Scripts`/`bin` path. The GPU
profile (`MNIMI_EXTRACTOR=gpu`) needs a CUDA build of the same version instead
(`docs/DECISIONS.md` "Extractor runtime"); the CPU profile is the default and measured at
21.0 s per round on an 8-core laptop CPU (`examples/README.md`).

## Register it with Claude Code

```
claude mcp add --scope user mnimi -e MNIMI_DB=/absolute/path/to/mnimi/agent.db -- mnimi-mcp
```

Give `MNIMI_DB` an absolute path: outside bash a `~` reaches the server literally (the server
does expand `~` and `$VAR` itself, and makes a relative path absolute against *its* working
directory, which under Claude Code is the project folder, not your home). `--scope user` makes
the memory follow you across projects; the default scope is one project. `mnimi-mcp` must
resolve on the PATH Claude Code starts the server with; give the venv's full path to the script
otherwise. Then `claude mcp list` shows the server connected and `/mcp` inside a session lists the
five tools. On the first tool call the server opens the DB,
loads the embedder (one 130 MB download, pinned by revision) and, under `MNIMI_EXTRACTOR=cpu`
or `gpu`, the extractor (one 1.8 GB download, pinned by revision). Listing the tools loads
nothing.

## Configuration (environment)

| variable | values | default | what it is |
| --- | --- | --- | --- |
| `MNIMI_DB` | a path | **required** | the SQLite store; created with its `memory_meta` on first use |
| `MNIMI_USER_ID` | a string | `me` | the `user_id` every record is stored and recalled under |
| `MNIMI_EXTRACTOR` | `cpu` · `gpu` · `none` | `cpu` | `cpu` = the measured recipe (`{**DECODE, "n_gpu_layers": 0}`); `gpu` = the paper's pin (CUDA build); `none` = rounds only, no fact records |
| `MNIMI_EMBEDDER` | `bge` · `hashing` | `bge` | `bge` = `BAAI/bge-small-en-v1.5` (the real one); `hashing` = numpy only, for tests |

A store remembers these pins in `memory_meta`. Opening it later under other pins (another
embedder, the GPU profile on a CPU store) is refused with `MemoryMetaError`, and the tool call
returns that message as its error text (so the model, and you, read it): re-point `MNIMI_DB` at
a fresh file or use the pins the store was built with. A model that cannot load
(`MNIMI_EXTRACTOR=gpu` on a CPU build) and a bad setting come back the same way.

## Tools

| tool | arguments | returns |
| --- | --- | --- |
| `remember` | `messages: [{role: user\|assistant, content}]` (at least one) | `{round, fact, superseded, ts}`: the records this call added by kind (a duplicate round adds 0), the earlier facts it superseded, and the session timestamp they were stamped with |
| `recall` | `query`, `k` (default 10) | `[{score, salience, kind, text, session_date}]`, best first; the returned memories count as accessed today, which protects them from the next decay pass |
| `context` | `query` | the context block the library hands a reader: session-date headers, `user:`/`assistant:` lines, the facts |
| `export` | — | the whole memory as Markdown (the same text `mnimi export --format md` writes) |
| `consolidate` | — | `{conflict, decay, now_logical}`: what this call changed (the conflict pass's counters; the decay pass's `passes`, `decayed`, `at_floor`) and the logical now; a second call changes nothing |

Resource: `memory://export` (`text/markdown`), the same Markdown as the `export` tool.

**The clock lives here, not in the library.** Every message `remember` stores is stamped
`ts` = the server's calendar day at the call (`YYYY-MM-DDT00:00:00`): a day is a session, the
shape the benchmark has. A `ts` the model puts in a message is ignored. The same day reaches
`recall` and `context` as the library's documented query prefix (`[Current date: <ts>]`, parsed
for the question's own time window and stripped before embedding), so the shipped
configuration's time-aware term fires on "last week" or "three days ago". The library reads
only what it is handed and never a clock.

**stdout belongs to the transport.** Everything the server and the library log goes to
stderr, including `mnimi.memory`'s INFO lines (`superseded …`, `routed to conflict …`,
`decayed …`); Claude Code shows them in its MCP log.

## Try it

- Tell it three things about yourself over a conversation, then ask about one of them
  indirectly in a new session: `context` should carry the right round.
- Contradict one of them another day: the `superseded …` line, and `export` shows the old
  fact struck through.
- Give it a relative date ("dentist next Thursday") and later ask when: the fact's
  `valid_time` is resolved from the session date.
- `consolidate` twice: the second pass changes nothing.
- Read `memory://export` as a resource.

## Development

```
pip install -e integrations/mcp --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pytest integrations/mcp/tests            # also run by the root pytest; skipped when mcp is absent
```

(The `mcp` package is not in the root `[dev]` extra; the index is the CPU wheel's, as above.)

The tests drive the server through the SDK's in-memory client (`Client(server.mcp)`) under
`MNIMI_EMBEDDER=hashing MNIMI_EXTRACTOR=none` over a temporary DB. The server imports
`mnimi.extract.llama` for the decode dict and `mnimi_cli.main` for the Markdown export; it
changes nothing under `src/mnimi/`, which is frozen at the paper's bytes (`paper-v1`).

Apache-2.0, like the library.
