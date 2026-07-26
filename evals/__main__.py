"""``python -m evals`` — run a baseline against LongMemEval and print the table."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Stdlib-only, so importing these here keeps `python -m evals --help` from
# pulling in ollama/openai/huggingface_hub (those stay lazy inside main()).
# dataset's own heavy dep (huggingface_hub) is lazy inside download().
from . import artifacts
from .dataset import DEFAULT_SAMPLE_SEED, SAMPLE_FILE_ORDER, SAMPLE_STRATIFIED

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
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help="GPU layers for the reader. Default 99 (full offload), measured "
        "reproducible with num_batch pinned and ~23x faster than CPU. Pass 0 "
        "for CPU-only; any value other than the pin marks the run provisional.",
    )
    parser.add_argument(
        "--sample",
        default=SAMPLE_STRATIFIED,
        choices=[SAMPLE_STRATIFIED, SAMPLE_FILE_ORDER],
        help="how --limit selects questions. Default stratified: the dataset is "
        "clustered by category, so file-order slices are single-category and "
        "not comparable across systems.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="seed for stratified sampling (recorded in pins)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "predict", "judge"],
        help="'predict' ingests + reads and writes predictions.jsonl (needs Ollama, "
        "no OpenAI key); 'judge' grades an existing predictions.jsonl (needs an "
        "OpenAI key, no Ollama); 'all' does both (default)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="where staged artifacts live (default runs/<system>__<limit>q). "
        "Pass the same value to --stage judge that --stage predict used.",
    )
    args = parser.parse_args(argv)

    # Load .env (repo root) before any environ.get() below reads a key from it.
    from dotenv import load_dotenv

    load_dotenv()

    from .dataset import file_sha256, resolve_path
    from .judge import JUDGE_PROMPT_VERSION, judge_prompt_hash
    from .judge_cache import JudgeCache
    from .report import print_report
    from .runner import (
        READER_NUM_BATCH,
        READER_NUM_GPU,
        READER_NUM_THREAD,
        READER_PROMPT_VERSION,
        READER_SEED,
        READER_TOP_K,
        Prediction,
        judge_predictions,
        predict,
        reader_prompt_hash,
    )

    num_gpu = READER_NUM_GPU if args.num_gpu is None else args.num_gpu

    do_predict = args.stage in {"all", "predict"}
    do_judge = args.stage in {"all", "judge"}
    directory = Path(args.run_dir) if args.run_dir else artifacts.run_dir(
        args.system, args.limit
    )

    # Guards are stage-scoped: the predict stage never touches OpenAI, and the
    # judge stage never touches Ollama. Demanding both for either would make
    # re-judging on a machine without a reader impossible for no reason.
    if do_judge and not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. The judge "
            f"({args.judge_model}) runs on the OpenAI API; set it in .env and re-run.",
            file=sys.stderr,
        )
        return 2

    digest = None
    declared_ctx = None
    if do_predict:
        # Reader transport: local Ollama. Fail fast if daemon or model is missing.
        err, digest = ollama_preflight(args.model)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
        declared_ctx = ollama_context_length(args.model)

    started = time.perf_counter()

    if do_predict:
        dataset_path = resolve_path(args.dataset_file)
        pins = artifacts.build_pins(
            dataset_file=str(dataset_path),
            dataset_sha256=file_sha256(dataset_path),
            system=args.system,
            limit=args.limit,
            sample_strategy=args.sample,
            sample_seed=args.sample_seed,
            reader_model=args.model,
            reader_digest=digest,
            reader_num_ctx=args.num_ctx,
            reader_seed=READER_SEED,
            reader_top_k=READER_TOP_K,
            reader_num_gpu=num_gpu,
            reader_num_thread=READER_NUM_THREAD,
            reader_num_batch=READER_NUM_BATCH,
            reader_prompt_version=READER_PROMPT_VERSION,
            reader_prompt_hash=reader_prompt_hash(),
            judge_model=args.judge_model,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            judge_prompt_hash=judge_prompt_hash(),
        )
        _print_pins(pins, declared_ctx)

        def predict_progress(done: int, total: int, q, truncated: bool) -> None:
            mark = "TRUNC" if truncated else "  ok "
            print(
                f"[{done}/{total}] read {mark}  {q.question_id} ({q.question_type})",
                file=sys.stderr,
            )

        predictions = predict(
            build_system(args.system),
            reader_model=args.model,
            limit=args.limit,
            num_ctx=args.num_ctx,
            num_gpu=num_gpu,
            dataset_file=args.dataset_file,
            strategy=args.sample,
            sample_seed=args.sample_seed,
            progress=predict_progress,
        )
        artifacts.write_pins(directory, pins)
        artifacts.write_predictions(directory, predictions)
        print(f"wrote {directory / 'predictions.jsonl'}", file=sys.stderr)
    else:
        # Judge-only replay: the header and the rows both come off disk.
        try:
            pins = artifacts.read_pins(directory)
            predictions = artifacts.read_predictions(directory, Prediction)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        _print_pins(pins, None)
        if pins.get("judge_model") != args.judge_model:
            print(
                f"WARNING: judging with {args.judge_model} but predictions were "
                f"pinned to {pins.get('judge_model')}; recording the judge actually used.",
                file=sys.stderr,
            )
            pins = {**pins, "judge_model": args.judge_model}

    if not do_judge:
        elapsed = time.perf_counter() - started
        print(
            f"\npredict stage only: {len(predictions)} predictions in {elapsed:,.1f}s. "
            f"Grade them with:  python -m evals --system {args.system} "
            f"--limit {args.limit} --stage judge",
            file=sys.stderr,
        )
        return 0

    cache = JudgeCache()

    def judge_progress(done: int, total: int, p, correct: bool) -> None:
        mark = "PASS" if correct else "FAIL"
        print(f"[{done}/{total}] {mark}  {p.question_id} ({p.category})", file=sys.stderr)

    results = judge_predictions(
        predictions,
        judge_model=args.judge_model,
        judge_cache=cache,
        progress=judge_progress,
    )
    elapsed = time.perf_counter() - started

    print_report(args.system, results)

    # Run-level stats (kept out of report.py, which is category-table only).
    fed = [r.reader_prompt_tokens for r in results if r.reader_prompt_tokens is not None]
    truncated_n = sum(1 for r in results if r.truncated)
    dropped_total = sum(r.tokens_dropped for r in results)
    run_meta = {
        "stage": args.stage,
        "elapsed_s": round(elapsed, 1),
        "n": len(results),
        "reader_mean_prompt_tokens": round(sum(fed) / len(fed)) if fed else None,
        "truncated": truncated_n,
        "tokens_dropped_estimated": dropped_total,
        "judge_cache_hits": cache.hits,
        "judge_cache_misses": cache.misses,
        "abstention_questions": sum(1 for r in results if r.is_abstention),
        # Diagnostic only — never folded into pins_hash (see capture_environment).
        "environment": artifacts.capture_environment(
            ollama_host=OLLAMA_HOST, serve_log=os.environ.get("OLLAMA_SERVE_LOG")
        ),
    }
    provisional = _provisional_reasons(pins)
    results_path = artifacts.write_results(directory, pins, results, run_meta, provisional)

    mean_fed = f"{run_meta['reader_mean_prompt_tokens']:,}" if fed else "n/a"
    print(
        f"\nwall-clock: {elapsed:,.1f}s over {len(results)} q"
        f"  |  mean reader prompt tokens: {mean_fed}"
        f"  |  truncated: {truncated_n}/{len(results)} (~{dropped_total:,} tok dropped)"
        f"  |  judge cache: {cache.hits} hit / {cache.misses} miss",
        file=sys.stderr,
    )
    print(f"wrote {results_path}", file=sys.stderr)
    if provisional:
        print(
            "PROVISIONAL - not publishable: " + "; ".join(provisional),
            file=sys.stderr,
        )
    return 0


def _print_pins(pins: dict, declared_ctx: int | None) -> None:
    """Echo the header to stderr so a run is self-describing in the terminal."""
    print("--- pins ---", file=sys.stderr)
    print(f"harness git:      {pins.get('harness_git_sha')}", file=sys.stderr)
    print(f"dataset:          {pins.get('dataset_file')}", file=sys.stderr)
    print(f"dataset sha256:   {pins.get('dataset_sha256')}", file=sys.stderr)
    print(
        f"reader (Ollama):  {pins.get('reader_model')}  digest={pins.get('reader_digest')}",
        file=sys.stderr,
    )
    print(
        f"reader num_ctx:   {pins.get('reader_num_ctx')}"
        + (f"  (model declares {declared_ctx})" if declared_ctx else ""),
        file=sys.stderr,
    )
    print(
        f"reader prompt:    {pins.get('reader_prompt_version')} "
        f"({str(pins.get('reader_prompt_hash'))[:12]}...)",
        file=sys.stderr,
    )
    print(
        f"judge (literal):  {pins.get('judge_model')}  prompts="
        f"{pins.get('judge_prompt_version')} ({str(pins.get('judge_prompt_hash'))[:12]}...)",
        file=sys.stderr,
    )
    if pins.get("judge_model") in {"gpt-4o", "gpt-4o-mini"}:
        print(
            "WARNING: judge model looks like a rolling alias, not a pinned snapshot.",
            file=sys.stderr,
        )
    print(f"pins_hash:        {artifacts.pins_hash(pins)}", file=sys.stderr)
    print("------------", file=sys.stderr)


def _provisional_reasons(pins: dict) -> list[str]:
    """Why this artifact is not yet a publishable number. Empty list = it is."""
    from .runner import READER_NUM_BATCH, READER_NUM_GPU

    reasons = []
    if pins.get("reader_prompt_version") != "json-con-v1":
        reasons.append(
            f"reader prompt is {pins.get('reader_prompt_version')} "
            "(JSON + Chain-of-Note pending Phase D)"
        )
    if str(pins.get("harness_git_sha", "")).endswith("-dirty"):
        reasons.append("harness tree was dirty at run time")
    # A deviated decode pin means this run is not the configuration the
    # determinism evidence was gathered under.
    if pins.get("reader_num_gpu") != READER_NUM_GPU:
        reasons.append(
            f"reader_num_gpu={pins.get('reader_num_gpu')} overrides the pin "
            f"({READER_NUM_GPU}) — not the measured-reproducible configuration"
        )
    if pins.get("reader_num_batch") != READER_NUM_BATCH:
        reasons.append(
            f"reader_num_batch={pins.get('reader_num_batch')} overrides the pin "
            f"({READER_NUM_BATCH}) — unpinned batch size caused observed drift"
        )
    if pins.get("sample_strategy") != "stratified-round-robin":
        reasons.append(
            f"sample_strategy={pins.get('sample_strategy')} — a file-order slice "
            "is category-clustered and not comparable across systems"
        )
    return reasons


if __name__ == "__main__":
    raise SystemExit(main())
