# configs/ — one committed argument list per run of record

`python -m evals --config configs/<run_id>.json [...]` splices the file's `args` where `--config`
stood, so a later command-line flag overrides one from the file (PHASE8 D5, PLAN A3). The schema:

```json
{"config_schema": "mnimi-run-config/1", "args": ["--system", "mnimi", "--limit", "500", "..."]}
```

Rules: the file is named `<run_id>.json` and committed **before** its run; the manifest records
`arm.config_file`, `arm.config_sha256` and `arm.config_committed` (whether `git show HEAD:<file>`
holds the same bytes); a `--limit >= 100` predict without a committed config is refused unless
`--allow-unfrozen`, which marks the run provisional.
