# CLAUDE.md — agentmem

Project-level instructions for working in this repo. Read before changing code.

## What this is

An embeddable, local-first agent-memory library. Thin public API, deep write-side
policy. Zero infra: one SQLite file via `sqlite-vec`, no server, no external
services. The library must import and run with only its two core deps.

## Stack

- **Language:** Python (>=3.10), src-layout (`src/agentmem/`).
- **Storage:** SQLite + `sqlite-vec` for vector search. One file on disk.
- **Core deps:** `sqlite-vec`, `numpy`. Nothing else. Adding a third is a
  decision, not a convenience — justify it in `docs/DECISIONS.md`.
- **Eval-only deps** (`[eval]` extra): `anthropic`, `huggingface-hub`. Never
  imported by the library.
- **Build:** hatchling. **Lint:** ruff. **Test:** pytest.

## API contract (locked — keep these signatures)

```python
class Memory:
    def __init__(self, db_path: str, embedder: Embedder): ...
    def add(self, messages, user_id: str) -> None          # write path
    def recall(self, query: str, user_id: str) -> list      # raw retrieval
    def get_context(self, query: str, user_id: str) -> str  # assembled context
    def consolidate(self, user_id: str) -> None             # merge / conflict / decay
```

Four methods. Do not widen the surface. Depth goes behind `add` and
`consolidate`, not into new public methods.

## Locked decisions

- **Language = Python.**
- **Storage = sqlite-vec** (local-first, zero-infra). Not a hosted vector DB.
- **Benchmark = LongMemEval** (`longmemeval_s`, ~500 questions). The harness in
  `evals/` is the source of truth.
- **Baselines = no-memory (floor), full-history (ceiling), naive-RAG.** agentmem
  must beat naive-RAG and approach full-history at a fraction of the tokens.

## Kill gates

- **W3 (Jul 17, 2026):** the read/write loop must beat the naive baselines by a
  repeatable margin. If it does not, stop and rethink the approach.
- **W6 (Aug 7, 2026):** either a competitive LongMemEval number, or reframe the
  project as a zero-infra usability play (the "one file, no server" pitch) rather
  than a state-of-the-art accuracy play.

## Scope rule

- **In scope (library):** write-side policy — extraction, consolidation, conflict
  resolution, decay, salience, retrieval. This is the product.
- **Out of scope:** UI, rendering, chat frontends, a docs site. The library
  produces context strings; what renders them is someone else's problem.

## Convention

Every change is validated by whether it moves the benchmark number. New code that
does not change a baseline, a system, or the score is suspect. When in doubt, run:

```bash
python -m evals --system <name> --limit 10
```

## Layout

- `src/agentmem/` — the library. `models.py` (MemoryRecord), `store.py` (the
  sqlite-vec store), `embeddings.py` (Embedder protocol + default), `memory.py`
  (the Memory facade).
- `evals/` — the benchmark harness. Separate from the library. `base.py` defines
  the `MemorySystem` ABC that any system (including competitors) implements.
- `tests/` — pytest. `docs/` — SPEC / ARCH / DECISIONS.

## Do not

- Do not import eval deps from `src/agentmem/`.
- Do not add public methods to `Memory`.
- Do not commit downloaded eval data or `*.db` files (see `.gitignore`).
