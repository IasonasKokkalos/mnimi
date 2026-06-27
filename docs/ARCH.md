# ARCH

How the pieces fit. Placeholder; fills in as the write path lands.

## Layers

- **`store.py`** — sqlite-vec persistence. Schema, insert, vector KNN search. The
  only module that talks to SQLite.
- **`embeddings.py`** — `Embedder` protocol + a dependency-light default.
- **`memory.py`** — the `Memory` facade. Orchestrates store + embedder behind the
  four-method API. Write/consolidate policy will accrete here (or in helpers it calls).
- **`models.py`** — `MemoryRecord`, the storage shape.

## Data flow (today)

`add` → embed message text → `store.insert`. `recall` → embed query →
`store.search` (KNN) → `MemoryRecord`s. `get_context` → `recall` → join into a string.

## Eval harness (`evals/`)

Standalone. `dataset` loads LongMemEval → `runner` drives a `MemorySystem` per
question (reset → add sessions → get_context → reader → judge) → `report` aggregates.
TBD: retrieval index design, consolidation strategy, decay model.
