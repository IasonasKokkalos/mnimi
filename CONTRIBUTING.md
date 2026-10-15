# Contributing

Thanks for looking. mnimi is a small library with a strict harness around it; the rules below
are what keeps the published numbers meaningful. [`CLAUDE.md`](CLAUDE.md) is the long version,
[`docs/SPEC.md`](docs/SPEC.md) § "v1 as built" the exact shipped state, and
[`docs/DECISIONS.md`](docs/DECISIONS.md) the dated reasons.

## The library and the published numbers

`src/mnimi/` is digest-pinned: `evals/freeze.py` holds a SHA-256 over the tree and
`tests/test_library_freeze.py` fails on any edit that does not recompute it in the same commit. A
library change is therefore a dated `docs/DECISIONS.md` entry plus a new digest, never an edit
that makes the test pass. The published n=500 rows were measured on the library at tag
`v2.14.0`; release work lives on `release/*` branches, and a change there is a developer-release
change — the published arms are never rerun from it, and its own LongMemEval reading is reported
as a developer-release reading beside them. Work around the library — `src/mnimi_cli/`,
`examples/`, `integrations/`, `evals/` — needs no digest.

## Running a benchmark

- Every `python -m evals` run names its `--purpose`; a `--claim` needs a `--rule-commit`.
- A run writes `runs/<run_id>/manifest.json` and a row in `runs/INDEX.md`; both are tracked, so
  commit them before the next run of record (`freeze.check` refuses a dirty tree).
- `predictions.jsonl` is never overwritten; a re-grade is `judge_replay_N.json`.
- Paired comparisons come from rows (`python -m evals.stats a b`, saved under `analyses/`),
  never from summary scores; a quoted number cites its `run_id`.
- A number is quotable only after `python -m evals.publish runs/<run_id>` (clean tree, complete
  manifest) puts it under `results/published/`.

Two things are prohibited for every system, including yours: selecting a threshold against
LongMemEval (the 0.95 dedup selection cannot be regenerated), and a reader prompt tuned per
question type (one prompt, `mnimi-con-v1`, for every question and every system).

## Tests and lint

```
pip install -e ".[dev]"
PYTHONPATH="src;." python -m pytest -q      # the MCP tests need `pip install -e integrations/mcp`
python -m ruff check .
```

CI installs the `[dev]` extra only, so nothing on a test path may import `openai`, `ollama` or
`python-dotenv`; the `[embed]`/`[extract]` tests skip without their extras. A second CI job runs
the suite from `uv.lock` — run `uv lock` after any version bump.

## Adding a system to the harness

A system under test implements `evals.base.MemorySystem`:

```python
class MemorySystem:
    def reset(self) -> None: ...                       # a fresh store per question
    def add(self, messages: list[dict]) -> None: ...   # one history, {role, content, ts} dicts
    def get_context(self, query: str) -> str: ...      # what the reader sees
    def set_question_date(self, question_date: str | None) -> None: ...  # optional
```

Render through `mnimi.memory.render_turns` (one renderer for every arm: date granularity and
speaker labels were a measured confound), ingest per round with the same `k` as every other
arm, register it in `evals/__main__.py`, and run it at n=500 with `--purpose`. A third-party
package goes through `evals/systems/competitor.py` (its own environment under `mnimi-envs/`,
the 5,364-token context budget, its own ids as `retrieved_ids`); the Mem0 and OMEGA adapters
are the worked examples. Open a pull request with the adapter and its `configs/` entry; the
run is paired against the same rows and reported as "through this harness under these pins",
never as a rank.

## House style

- Commit messages in the present imperative, one functional unit per commit, the reason in the
  body; the commit log and DECISIONS are the changelog (there is no `CHANGELOG.md`).
- The repo is checked out with `core.autocrlf` on Windows; the freeze digest is taken over
  LF-normalised bytes, so line endings never move it. Keep files LF.
- Version of record: `pyproject.toml`; SPEC's § "v1 as built" header tracks it.
- No downloaded eval data, no `*.db`, no `runs/` output in a commit (see `.gitignore`).
