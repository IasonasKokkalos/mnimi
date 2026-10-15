# PT1 compute evidence (2026-10-15, run `mnimi__500q_gpt4o_pt1`)

The README's "Storage and footprint" compute table and the compute paragraph of DECISIONS "PT1 read" are derived
from these two files. The run replays the extraction cache, so they measure the library's ingest + retrieval for
500 stores plus the Batch API wait, not the extractor.

- `resources.csv`: `python tools/sample_resources.py --match evals --interval 5`, started three seconds after the
  arm (00:09:57 local), one row every 5 s until the harness process exited. Columns: `t_s` (seconds since the
  sampler started), `proc_cpu_pct` (100 = one core; the machine has 16 logical cores), `proc_rss_mb` and
  `sys_avail_mb` (MiB), `proc_threads`, `sys_cpu_pct`. Ingest ends at `t_s` ≈ 12,267 (the first store at
  00:10:01 plus the manifest's `ingest_s` 12,262.5); the rows after it are the Batch API wait.
- `stores.csv`: the run's scratch stores (`%TEMP%\mnimi-eval-x5qoykrx\qN.db`), watched once a second by a script
  that only listed the directory and stat-ed the files. It never opened a store, so the harness's unlink of the
  previous store at each reset could not fail. Per store: the first and last second it was seen (since the
  watcher found the directory) and its last-seen `db_bytes` and `wal_bytes`. `q1.db` is the empty store built
  at system construction (4 KB) and is left out; `q2`–`q501` are the 500 questions in predict order. The last
  store stays open until the process exits; its `closed_*` columns were read five seconds after the exit, with
  the WAL checkpointed.

Derivations: CPU mean and p95 per phase, split at `t_s` 12,267, with p95 = `sorted(x)[int(0.95 * (n - 1))]`
(the tool's own method); the per-store size is the median of `db_bytes` over `q2`–`q501`; seconds per round are
`ingest_s` 12,262.5 ÷ 124,362 rounds (the retrieval probe's count over the same haystacks). Two vec0 tables
holding one 384-dim vector each already take 3,219,456 B, because their first 1,024-vector chunks are
preallocated, so 3.07 MiB of any non-empty store is that fixed cost; the harness process's RSS also includes
the parsed 500-question dataset (`evals.dataset.load(500)` adds ≈ 342 MiB in a fresh process).
