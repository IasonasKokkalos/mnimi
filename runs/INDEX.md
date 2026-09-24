# Run registry

One row per run, including failed and aborted ones. Append-only: only the `status` column may change (and a `pending` score is filled once, when the judge stage lands). Never delete a row. Status is one of published / provisional / aborted / incomplete / complete. See `evals/manifest.py`.

| run_id | date | arm | reader | n | score | clean | status | claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_memory__500q_gpt4o` | 2026-09-21 | no_memory | gpt-4o-2024-08-06 | 500 | 31/500 | Y | published | none |
| `mnimi__500q_gpt4o` | 2026-09-22 | mnimi | gpt-4o-2024-08-06 | 500 | 422/500 | Y | published | none |
| `mnimi__500q_gpt4o_decay` | 2026-09-22 | mnimi | gpt-4o-2024-08-06 | 500 | 391/500 | Y | published | none |
| `naive_rag__500q_gpt4o` | 2026-09-22 | naive_rag | gpt-4o-2024-08-06 | 500 | 373/500 | Y | published | none |
| `oracle__500q_gpt4o` | 2026-09-22 | oracle | gpt-4o-2024-08-06 | 500 | 459/500 | Y | published | none |
| `mnimi__500q_gpt4o_p6time` | 2026-09-24 | mnimi | gpt-4o-2024-08-06 | 500 | 429/500 | Y | published | none |
| `mnimi__500q_gpt4o_p6turns` | 2026-09-24 | mnimi | gpt-4o-2024-08-06 | 500 | 424/500 | Y | published | none |
| `mnimi__500q_gpt4o_p6combo` | 2026-09-24 | mnimi | gpt-4o-2024-08-06 | 500 | 426/500 | Y | published | none |
