# SPEC

The contract. Signatures here are locked; changing them is a breaking change.

## Public API — `agentmem.Memory`

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # raw retrieval -> list[MemoryRecord]
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
```

- `add` — ingest messages for a user. Today: naive store of message text. Later:
  extraction, salience, dedup.
- `recall` — raw vector retrieval. Returns `MemoryRecord`s, no assembly.
- `get_context` — assemble retrieved records into a single context string for a reader.
- `consolidate` — merge duplicates, resolve conflicts, decay stale memories. Stub today.

## `MemoryRecord` (storage shape)

`id`, `user_id`, `content`, `embedding`, `created_at`, `salience` (float),
`source` (str), `supersedes` (id | None), `pinned` (bool). This shape constrains
the whole write path; it is the schema of record.

## `Embedder` protocol

`embed(texts: list[str]) -> list[list[float]]` and a `dim` property. One default
implementation ships with the library and pulls in no dependency beyond `numpy`.

## Benchmark contract — `evals.base.MemorySystem`

Separate from `Memory`. Any system under test implements `reset()`,
`add(messages)`, `get_context(query) -> str`. The harness drives it identically
for every system so results are comparable.
