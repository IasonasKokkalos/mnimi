# The paper's tables

Exported by `python -m evals.paper_tables` from `results/published/` and `analyses/`; never edited by hand.

## T1 — accuracy, n=500, reader = judge = gpt-4o-2024-08-06

| arm | role | correct / n | % | Wilson 95 % | under gpt-4.1-2025-04-14 |
| --- | --- | --- | --- | --- | --- |
| no_memory__500q_gpt4o | floor | 31/500 | 6.2 % | [4.4, 8.7] | 31/500 |
| naive_rag__500q_gpt4o | strong K=V baseline | 373/500 | 74.6 % | [70.6, 78.2] | 368/500 |
| mnimi__500q_gpt4o | the Phase 5 verdict arm | 422/500 | 84.4 % | [81.0, 87.3] | 415/500 |
| mnimi__500q_gpt4o_decay | SPEC's decay ablation | 391/500 | 78.2 % | [74.4, 81.6] | 391/500 |
| mnimi__500q_gpt4o_p6time | the shipped configuration (L1 alone) | 429/500 | 85.8 % | [82.5, 88.6] | 417/500 |
| mnimi__500q_gpt4o_p6turns | L3: turns, extractor on | 424/500 | 84.8 % | [81.4, 87.7] | 420/500 |
| mnimi__500q_gpt4o_p6combo | the Phase 6 headline | 426/500 | 85.2 % | [81.8, 88.0] | 423/500 |
| oracle__500q_gpt4o | evidence-availability bound | 459/500 | 91.8 % | [89.1, 93.9] | 451/500 |
| mem0__500q_gpt4o | third-party, LLM-routed writes | 336/500 | 67.2 % | [63.0, 71.2] | 322/500 |
| omega__500q_gpt4o | third-party retrieval over verbatim rounds | 365/500 | 73.0 % | [68.9, 76.7] | 356/500 |

## T2 — per category (correct / n)

| category | n | no_memory | naive_rag | mnimi P5 | mnimi+decay | mnimi L1 | mnimi L3 | mnimi L1+L3 | oracle | Mem0 OSS | OMEGA retrieval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single-session-user | 70 | 6/70 | 64/70 | 68/70 | 68/70 | 66/70 | 67/70 | 67/70 | 67/70 | 64/70 | 66/70 |
| single-session-assistant | 56 | 0/56 | 55/56 | 55/56 | 55/56 | 56/56 | 56/56 | 56/56 | 56/56 | 19/56 | 55/56 |
| single-session-preference | 30 | 0/30 | 18/30 | 23/30 | 21/30 | 25/30 | 21/30 | 21/30 | 22/30 | 23/30 | 16/30 |
| multi-session | 133 | 12/133 | 83/133 | 95/133 | 91/133 | 98/133 | 102/133 | 101/133 | 120/133 | 90/133 | 69/133 |
| knowledge-update | 78 | 6/78 | 67/78 | 70/78 | 60/78 | 71/78 | 72/78 | 72/78 | 73/78 | 70/78 | 66/78 |
| temporal-reasoning | 133 | 7/133 | 86/133 | 111/133 | 96/133 | 113/133 | 106/133 | 109/133 | 121/133 | 70/133 | 93/133 |

## T3 — the pairs of record (b = the second arm's wins; exact McNemar)

| comparison | kind | b | c | discordant | p | b₂ | c₂ | p₂ | holds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_rag__500q_gpt4o → mnimi__500q_gpt4o | primary | 75 | 26 | 101 | 1.1e-06 | 75 | 28 | 4e-06 | yes |
| no_memory__500q_gpt4o → mnimi__500q_gpt4o | secondary | 393 | 2 | 395 | 1.9e-114 | 386 | 2 | 2.4e-112 | yes |
| oracle__500q_gpt4o → mnimi__500q_gpt4o | secondary | 17 | 54 | 71 | 1.3e-05 | 20 | 56 | 4.4e-05 | yes |
| mnimi__500q_gpt4o → mnimi__500q_gpt4o_decay | decay ablation | 19 | 50 | 69 | 0.00024 | 23 | 47 | 0.0056 | yes |
| mnimi__500q_gpt4o → mnimi__500q_gpt4o_p6time | Phase 6 L1 | 15 | 8 | 23 | 0.21 | 13 | 11 | 0.84 | yes |
| mnimi__500q_gpt4o → mnimi__500q_gpt4o_p6turns | Phase 6 L3 | 22 | 20 | 42 | 0.88 | 27 | 22 | 0.57 | yes |
| mnimi__500q_gpt4o → mnimi__500q_gpt4o_p6combo | Phase 6 L1+L3 | 18 | 14 | 32 | 0.6 | 25 | 17 | 0.28 | yes |
| naive_rag__500q_gpt4o → mnimi__500q_gpt4o_p6time | the paper system vs the bar | 77 | 21 | 98 | 1.1e-08 | 76 | 27 | 1.4e-06 | yes |
| mnimi__500q_gpt4o_p6time → mem0__500q_gpt4o | competitor | 28 | 121 | 149 | 5.6e-15 | 28 | 123 | 2.1e-15 | yes |
| mnimi__500q_gpt4o_p6time → omega__500q_gpt4o | competitor | 19 | 83 | 102 | 1e-10 | 24 | 85 | 3.5e-09 | yes |
| naive_rag__500q_gpt4o → mem0__500q_gpt4o | competitor vs the bar | 67 | 104 | 171 | 0.0057 | 67 | 113 | 0.00075 | yes |
| naive_rag__500q_gpt4o → omega__500q_gpt4o | competitor vs the bar | 45 | 53 | 98 | 0.48 | 47 | 59 | 0.29 | yes |

b₂, c₂, p₂: the same pair under gpt-4.1-2025-04-14; a pair holds iff sign(b − c) is the same under both judges (Phase 7 D7).

## T4a — the instrument, Tier 1: the cold audits

| arm | published | recomputed | flips to wrong / to right | fresh judge calls | verdict |
| --- | --- | --- | --- | --- | --- |
| no_memory__500q_gpt4o | 31/500 | 31/500 | 0 / 0 | 500 | MATCHES |
| naive_rag__500q_gpt4o | 373/500 | 370/500 | 3 / 0 | 500 | WITHIN RE-GRADE |
| mnimi__500q_gpt4o | 422/500 | 419/500 | 3 / 0 | 498 | WITHIN RE-GRADE |
| mnimi__500q_gpt4o_decay | 391/500 | 389/500 | 2 / 0 | 393 | WITHIN RE-GRADE |
| mnimi__500q_gpt4o_p6time | 429/500 | 426/500 | 3 / 0 | 292 | WITHIN RE-GRADE |
| mnimi__500q_gpt4o_p6turns | 424/500 | 424/500 | 1 / 1 | 442 | WITHIN RE-GRADE |
| mnimi__500q_gpt4o_p6combo | 426/500 | 427/500 | 0 / 1 | 442 | WITHIN RE-GRADE |
| oracle__500q_gpt4o | 459/500 | 456/500 | 3 / 0 | 498 | WITHIN RE-GRADE |

## T4b — the instrument, Tier 2: the drift pair of the shipped configuration

| reference → fresh | texts changed | prompt tokens changed | b / c / p (the null pair) | b / c / p under gpt-4.1-2025-04-14 |
| --- | --- | --- | --- | --- |
| mnimi__500q_gpt4o_p6time → mnimi__500q_gpt4o_p6time_drift_2026-09-26 | 277/500 | 0/500 | 6 / 6 / 1 | 7 / 7 / 1 |

## T4c — the instrument, the two judges (gpt-4o-2024-08-06 vs gpt-4.1-2025-04-14)

| arm | agree / n | rate | Wilson 95 % | κ | yy / yn / ny / nn |
| --- | --- | --- | --- | --- | --- |
| no_memory__500q_gpt4o | 500/500 | 100.0 % | [99.2, 100.0] | 1.000 | 31 / 0 / 0 / 469 |
| naive_rag__500q_gpt4o | 485/500 | 97.0 % | [95.1, 98.2] | 0.922 | 363 / 10 / 5 / 122 |
| mnimi__500q_gpt4o | 491/500 | 98.2 % | [96.6, 99.1] | 0.934 | 414 / 8 / 1 / 77 |
| mnimi__500q_gpt4o_decay | 494/500 | 98.8 % | [97.4, 99.4] | 0.965 | 388 / 3 / 3 / 106 |
| mnimi__500q_gpt4o_p6time | 486/500 | 97.2 % | [95.4, 98.3] | 0.893 | 416 / 13 / 1 / 70 |
| mnimi__500q_gpt4o_p6turns | 488/500 | 97.6 % | [95.9, 98.6] | 0.909 | 416 / 8 / 4 / 72 |
| mnimi__500q_gpt4o_p6combo | 491/500 | 98.2 % | [96.6, 99.1] | 0.930 | 420 / 6 / 3 / 71 |
| oracle__500q_gpt4o | 490/500 | 98.0 % | [96.4, 98.9] | 0.878 | 450 / 9 / 1 / 40 |
| pooled | 3925/4000 | 98.1 % | [97.7, 98.5] | 0.952 | 2898 / 57 / 18 / 1027 |

yy = both yes, yn = the first judge yes and the second no, ny = the reverse, nn = both no.

## T4d — the instrument, judge test-retest (C2): three cache-off replays by the first judge

| arm | judge-unstable rows / n | Wilson 95 % | score by grading (first, then the replays) | flips to wrong / to right |
| --- | --- | --- | --- | --- |
| no_memory__500q_gpt4o | 0/500 | [0.0, 0.8] | 31, 31, 31, 31 | 0 / 0 |
| naive_rag__500q_gpt4o | 4/500 | [0.3, 2.0] | 373, 371, 371, 371 | 7 / 1 |
| mnimi__500q_gpt4o_p6time | 6/500 | [0.6, 2.6] | 429, 426, 426, 426 | 10 / 1 |
| oracle__500q_gpt4o | 3/500 | [0.2, 1.7] | 459, 459, 458, 458 | 5 / 3 |

## T4e — the instrument, the human labels (60 rows, blind)

| rows where | n | human agrees with the first judge | Wilson 95 % | human agrees with the second judge | Wilson 95 % |
| --- | --- | --- | --- | --- | --- |
| the judges disagree | 50 | 32/50 | [50.1, 75.9] | 18/50 | [24.1, 49.9] |
| the judges agree | 10 | 10/10 | [72.2, 100.0] | 10/10 | [72.2, 100.0] |

## T5 — cited, not run

| system | reader | score | source |
| --- | --- | --- | --- |
| full_history | GPT-4o + Chain-of-Note | 64.0 % | LongMemEval (Wu et al., 2024), Fig. 3b, LongMemEval-S; cited, never run |

## T6a — the accounting (C1): mnimi__500q_gpt4o_p6time against oracle__500q_gpt4o, judge-stable rows

| category | n stable | both right | oracle right, system wrong | both wrong | system right, oracle wrong | judge-unstable |
| --- | --- | --- | --- | --- | --- | --- |
| single-session-user | 62 | 58 | 2 | 1 | 1 | 2 |
| single-session-assistant | 55 | 55 | 0 | 0 | 0 | 1 |
| single-session-preference | 29 | 21 | 0 | 5 | 3 | 1 |
| multi-session | 120 | 84 | 25 | 9 | 2 | 1 |
| knowledge-update | 70 | 63 | 2 | 3 | 2 | 2 |
| temporal-reasoning | 126 | 102 | 14 | 6 | 4 | 1 |
| abstention | 30 | 26 | 1 | 1 | 2 | 0 |
| overall | 492 | 409 | 44 | 25 | 14 | 8 |

The unanswerable (abstention) rows are counted in the abstention row and not in their category's row, so the rows above overall partition the benchmark and a category's n here is smaller than in T2.

## T6b — the primary per category (F2): naive_rag__500q_gpt4o → mnimi__500q_gpt4o_p6time

| category | n | b | c | discordant | p | p (Holm, 6 tests) |
| --- | --- | --- | --- | --- | --- | --- |
| single-session-user | 70 | 4 | 2 | 6 | 0.69 | 1 |
| single-session-assistant | 56 | 1 | 0 | 1 | 1 | 1 |
| single-session-preference | 30 | 8 | 1 | 9 | 0.039 | 0.16 |
| multi-session | 133 | 25 | 10 | 35 | 0.017 | 0.083 |
| knowledge-update | 78 | 7 | 3 | 10 | 0.34 | 1 |
| temporal-reasoning | 133 | 32 | 5 | 37 | 7.4e-06 | 4.5e-05 |

## T7 — the third-party systems, through this harness under these pins

| arm | version | write-side LLM | unit | embedder | wall clock | correct / n | Wilson 95 % | b / c / p vs mnimi__500q_gpt4o_p6time | b / c / p vs naive_rag__500q_gpt4o |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mem0__500q_gpt4o | mem0ai==2.2.1 | openai:gpt-4o-mini-2024-07-18 | the system's own memories | BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | created_at stamped from the wall clock, not read by search; the extraction prompt grounds relative dates on the machine date (evals/systems/mem0_oss.py) | 336/500 | [63.0, 71.2] | 28 / 121 / 5.6e-15 | 67 / 104 / 0.0057 |
| omega__500q_gpt4o | omega-memory==1.5.17 | none | the system's own memories | BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a | created_at, access times and decay read the wall clock; created_at backdated to the session date (evals/systems/omega_retrieval.py) | 365/500 | [68.9, 76.7] | 19 / 83 / 1e-10 | 45 / 53 / 0.48 |
| agentmemory v4 @ 3aa3b83 | — | — | — | — | deferred: its smoke measured 20-28 min per history (DECISIONS 2026-09-27); no n=500 arm | — | — | — | — |

b = the third-party arm's wins. Paired readings, never a rank. unit is the manifest's chunk_unit, the units the system itself stores; in the OMEGA arm those are naive_rag's rounds (its role in T1).

## T8 — provenance: one row per run the tables read

| run_id | commit | tag | config sha256 | reader served models | judge | replays | machine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory__500q_gpt4o | f07c24d | — | pre-rule | UNKNOWN | gpt-4o-2024-08-06 | 4 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| naive_rag__500q_gpt4o | f07c24d | — | pre-rule | UNKNOWN | gpt-4o-2024-08-06 | 4 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o | f07c24d | — | pre-rule | UNKNOWN | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o_decay | f07c24d | — | pre-rule | UNKNOWN | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o_p6time | f0a7de3 | — | pre-rule | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 4 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o_p6turns | 3845c4e | — | pre-rule | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o_p6combo | 606e110 | — | pre-rule | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| oracle__500q_gpt4o | f07c24d | — | pre-rule | UNKNOWN | gpt-4o-2024-08-06 | 4 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mem0__500q_gpt4o | 2939867 | — | d57410e8ed6a | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| omega__500q_gpt4o | 4aaeb39 | — | f333f7a1ec1c | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |
| mnimi__500q_gpt4o_p6time_drift_2026-09-26 | 685e16b | — | pre-rule | gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | 1 | NVIDIA RTX 1000 Ada Generation Laptop GPU |

tag is the exact tag on the run's commit (— when the commit carries none); pre-rule = run before a committed config was required; UNKNOWN is what the manifest itself records; the reader and the judge are served over the API, the machine is where the contexts were built.
