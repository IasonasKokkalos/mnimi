# DECISIONS

Locked choices and why. Append-only; supersede, don't delete.

- **Language = Python.** Embeddable, ubiquitous in the agent ecosystem.
- **Storage = sqlite-vec.** Local-first, zero-infra, one file on disk. No hosted
  vector DB, no server process.
- **Core deps = `sqlite-vec` + `numpy`, and only those.** Every core dep is a
  liability the library forces on its users. The default embedder is therefore a
  numpy-only hashing embedder — no `sentence-transformers`, no API calls — so the
  library imports clean.
- **Benchmark = LongMemEval (`longmemeval_s`, ~500 questions).** The harness is
  the source of truth for every claim.
- **Baselines = no-memory, full-history, naive-RAG.** Floor, ceiling, and the bar
  mnimi must clear.
- **Eval loader uses `huggingface-hub`, not `datasets`.** The dataset's nested
  `haystack_sessions` breaks the HF Arrow viewer; we download the raw
  `longmemeval_s_cleaned.json` via `hf_hub_download` and parse with stdlib `json`.
  Leaner extra, no Arrow dependency, no schema-casting surprises.
- **Reader + judge = Anthropic API.** Key from `ANTHROPIC_API_KEY`. The judge
  follows the LongMemEval paper's per-question-type prompts (incl. abstention).
