"""``python -m evals`` — run a baseline against LongMemEval and print the table."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Phase A pins. Reader is a local Ollama model; judge is the paper's validated
# OpenAI snapshot. Split from the old single DEFAULT_MODEL so the two swap
# independently — they are different instruments with different jobs.
READER_MODEL = "qwen2.5:1.5b-instruct-q4_0"
JUDGE_MODEL = "gpt-4o-2024-08-06"
# Verify against `ollama show`; preflight prints the model's declared ceiling.
DEFAULT_NUM_CTX = 32768

OLLAMA_HOST = "http://localhost:11434"


def build_system(name: str):
    if name == "no_memory":
        from .systems.no_memory import NoMemorySystem

        return NoMemorySystem()
    if name == "full_history":
        from .systems.full_history import FullHistorySystem

        return FullHistorySystem()
    raise SystemExit(f"unknown system: {name}")


def _ollama_get(path: str):
    with urllib.request.urlopen(f"{OLLAMA_HOST}{path}", timeout=5) as resp:
        return json.load(resp)


def _ollama_post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.load(resp)


def ollama_preflight(model: str) -> tuple[str | None, str | None]:
    """Return ``(error, digest)``. Fails fast so the run never dies mid-loop."""
    try:
        tags = _ollama_get("/api/tags")
    except (urllib.error.URLError, OSError) as exc:
        return (
            f"Ollama daemon not reachable at {OLLAMA_HOST} ({exc}). Start it with "
            "`ollama serve` (install: https://ollama.com/download).",
            None,
        )
    models = tags.get("models", [])
    match = next((m for m in models if m.get("name") == model), None)
    if match is None:
        present = sorted(m.get("name", "") for m in models)
        return (
            f"Reader model '{model}' not present locally. Pull it with "
            f"`ollama pull {model}`. Present tags: {present}",
            None,
        )
    return None, match.get("digest")


def ollama_context_length(model: str) -> int | None:
    """Best-effort declared context length from /api/show, for visibility only."""
    try:
        info = _ollama_post("/api/show", {"model": model})
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for key, val in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            return val
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run a memory system against LongMemEval and print a score table.",
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=["no_memory", "full_history"],
        help="which baseline to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="run only the first N questions (default 10; a cheap smoke run)",
    )
    parser.add_argument(
        "--model", default=READER_MODEL, help="reader model tag (local Ollama)"
    )
    parser.add_argument(
        "--judge-model", default=JUDGE_MODEL, help="judge model id (OpenAI snapshot)"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="reader context window passed to Ollama (default 32768)",
    )
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="override the LongMemEval file (HF name or local path); "
        "e.g. longmemeval_oracle.json for a smaller download",
    )
    args = parser.parse_args(argv)

    # Judge transport: OpenAI. No Anthropic key is used on this path.
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. The judge "
            f"({args.judge_model}) runs on the OpenAI API; export your key and re-run.",
            file=sys.stderr,
        )
        return 2

    # Reader transport: local Ollama. Fail fast if the daemon or model is missing.
    err, digest = ollama_preflight(args.model)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    declared_ctx = ollama_context_length(args.model)

    # Capture both pins for visibility (full reproducibility header is Phase B).
    print("--- pins ---", file=sys.stderr)
    print(f"reader (Ollama):  {args.model}  digest={digest}", file=sys.stderr)
    print(
        f"reader num_ctx:   {args.num_ctx}"
        + (f"  (model declares {declared_ctx})" if declared_ctx else ""),
        file=sys.stderr,
    )
    print(f"judge (literal sent): {args.judge_model}", file=sys.stderr)
    if args.judge_model in {"gpt-4o", "gpt-4o-mini"}:
        print(
            "WARNING: judge model looks like a rolling alias, not a pinned snapshot.",
            file=sys.stderr,
        )
    print("------------", file=sys.stderr)

    from .judge_cache import JudgeCache
    from .report import print_report
    from .runner import run

    system = build_system(args.system)
    cache = JudgeCache()

    def progress(done: int, total: int, q, correct: bool) -> None:
        mark = "PASS" if correct else "FAIL"
        print(
            f"[{done}/{total}] {mark}  {q.question_id} ({q.question_type})",
            file=sys.stderr,
        )

    started = time.perf_counter()
    results = run(
        system,
        reader_model=args.model,
        judge_model=args.judge_model,
        limit=args.limit,
        num_ctx=args.num_ctx,
        dataset_file=args.dataset_file,
        judge_cache=cache,
        progress=progress,
    )
    elapsed = time.perf_counter() - started

    print_report(args.system, results)

    # Run-level stats (kept out of report.py, which is category-table only).
    fed = [r.reader_prompt_tokens for r in results if r.reader_prompt_tokens is not None]
    mean_fed = f"{sum(fed) / len(fed):,.0f}" if fed else "n/a"
    truncated_n = sum(1 for r in results if r.truncated)
    dropped_total = sum(r.tokens_dropped for r in results)
    print(
        f"\nwall-clock: {elapsed:,.1f}s over {len(results)} q"
        f"  |  mean reader prompt tokens: {mean_fed}"
        f"  |  truncated: {truncated_n}/{len(results)} (~{dropped_total:,} tok dropped)"
        f"  |  judge cache: {cache.hits} hit / {cache.misses} miss",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
