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


# The five systems of the W3 artifact: floor, truncated-context baseline,
# ceiling, bar, mnimi. Declared once so `--system` choices and every error
# message quoting them cannot drift apart — they already had, listing two
# systems in a message the parser had also enumerated.
SYSTEMS = ("no_memory", "full_history", "oracle", "naive_rag", "mnimi")


def build_system(name: str):
    """Construct a system by name. Imports are lazy — only mnimi and naive_rag
    need the embedder, and the other three must stay runnable without it."""
    if name == "no_memory":
        from .systems.no_memory import NoMemorySystem

        return NoMemorySystem()
    if name == "full_history":
        from .systems.full_history import FullHistorySystem

        return FullHistorySystem()
    if name == "oracle":
        from .systems.oracle import OracleSystem

        return OracleSystem()
    if name == "naive_rag":
        from .systems.naive_rag import NaiveRagSystem

        return NaiveRagSystem()
    if name == "mnimi":
        from .systems.mnimi import MnimiSystem

        return MnimiSystem()
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


def _force_model_load(model: str, num_ctx: int, num_gpu: int) -> None:
    """Load the model so the daemon logs its resolved settings.

    LOAD-BEARING FOR REPRODUCIBILITY — do not delete this as "just a preflight
    helper". The first inference after a model load runs against a cold CUDA
    graph cache (measured: 555 graphs reused vs 1,608 once warm) and produces a
    different answer from the same input. This call absorbs that first-inference
    state so question 1 never lands in it.

    Note what actually makes the run deterministic: not a saturated graph cache,
    but an *identical request sequence*. Two runs match because both begin with
    this same fixed 30-token generate, so the graph state at every subsequent
    question is the same (verified: reuse counts 1, 95, 205, 458 in both runs of
    a restart pair, 0/20 predictions changed). Change or remove this call and
    question 1 drifts again.

    Sends the run's FULL decode pin set, not just num_ctx/num_gpu. llama.cpp
    fixes ``n_batch`` when the context is created, so a load that omits
    ``num_batch`` builds an n_batch=1024 context — which Ollama then tears down
    and reloads at 512 when the first real question arrives. That wasted a load
    per run and, worse, meant preflight validated a context the run never used.
    """
    from .runner import READER_NUM_BATCH, READER_NUM_THREAD

    try:
        _ollama_post(
            "/api/generate",
            {
                "model": model,
                "prompt": "ready",
                "stream": False,
                "options": {
                    "num_predict": 1,
                    "num_ctx": num_ctx,
                    "num_gpu": num_gpu,
                    "num_batch": READER_NUM_BATCH,
                    "num_thread": READER_NUM_THREAD,
                },
            },
        )
    except (urllib.error.URLError, OSError, ValueError):
        pass  # preflight_reader_env reports the missing log far more usefully


def _recent_model_load_log() -> str:
    """Tail of the daemon's serve log covering the most recent model load."""
    path = artifacts.default_serve_log()
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-400_000:]
    except OSError:
        return ""


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
        choices=list(SYSTEMS),
        help="which baseline to evaluate. Required to predict; NOT required to "
        "judge, because predictions.jsonl carries everything the judge reads.",
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
    parser.add_argument(
        "--predictions",
        default=None,
        help="TIER 1 AUDIT: grade this predictions.jsonl directly. Implies "
        "--stage judge and needs no --system, no dataset, no reader and no "
        "Ollama. Read-only: it writes nothing, so auditing a published artifact "
        "cannot modify it.",
    )
    args = parser.parse_args(argv)

    # Load .env (repo root) before any environ.get() below reads a key from it.
    from dotenv import load_dotenv

    load_dotenv()

    from .dataset import file_sha256, resolve_path
    from .judge import JUDGE_PROMPT_VERSION, judge_prompt_hash
    from .judge_cache import JudgeCache
    from .report import print_report
    from .report import summary as report_summary
    from .runner import (
        READER_CACHE_RAM,
        READER_FLASH_ATTENTION,
        READER_NUM_BATCH,
        READER_NUM_GPU,
        READER_NUM_THREAD,
        READER_PROMPT_VERSION,
        READER_SEED,
        READER_TOP_K,
        Prediction,
        ReaderEnvError,
        judge_predictions,
        predict,
        preflight_reader_env,
        reader_prompt_hash,
        reader_trim_pins,
    )

    num_gpu = READER_NUM_GPU if args.num_gpu is None else args.num_gpu

    # A Tier 1 audit points at one predictions file: judge only, no system, no
    # dataset, no reader — and no writes into the artifact being audited.
    auditing = args.predictions is not None
    if auditing and args.stage == "predict":
        print(
            "ERROR: --predictions grades an existing file; it cannot predict.",
            file=sys.stderr,
        )
        return 2

    do_predict = args.stage in {"all", "predict"} and not auditing
    do_judge = args.stage in {"all", "judge"} or auditing
    if do_predict and not args.system:
        print(
            f"ERROR: --system is required for the predict stage (choices: "
            f"{', '.join(SYSTEMS)}).",
            file=sys.stderr,
        )
        return 2
    if not do_predict and not auditing and not args.system and not args.run_dir:
        print(
            "ERROR: judging needs somewhere to read from - pass --predictions "
            "<file>, --run-dir <dir>, or --system to use the default run dir.",
            file=sys.stderr,
        )
        return 2

    predictions_path = Path(args.predictions) if auditing else None
    # source_dir is where artifacts are READ from; directory is where they are
    # written. They differ only when auditing into a separate run dir.
    source_dir = (
        predictions_path.parent
        if auditing
        else (Path(args.run_dir) if args.run_dir else artifacts.run_dir(args.system, args.limit))
    )
    directory = Path(args.run_dir) if args.run_dir else source_dir

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

        # The daemon only reports its resolved flash-attention and prompt-cache
        # settings when it loads a model, so force the load the run needs anyway
        # and then assert the serving daemon is the one the pins describe. A run
        # served by the tray app's daemon carries `flash_attn = auto` and a live
        # prompt cache, which is a different configuration wearing this
        # configuration's pins_hash.
        try:
            _force_model_load(args.model, args.num_ctx, num_gpu)
            preflight_reader_env(_recent_model_load_log())
        except ReaderEnvError as exc:
            print(f"ERROR: reader environment does not match pins: {exc}", file=sys.stderr)
            return 2

    started = time.perf_counter()

    if do_predict:
        dataset_path = resolve_path(args.dataset_file)
        # Built before the pins so the header describes the system that will
        # actually run — a retrieving system contributes its own pins below.
        # It also means a missing [embed] extra fails here, before the header,
        # rather than after it.
        system = build_system(args.system)
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
            reader_flash_attention=READER_FLASH_ATTENTION,
            reader_cache_ram=READER_CACHE_RAM,
            reader_prompt_version=READER_PROMPT_VERSION,
            reader_prompt_hash=reader_prompt_hash(),
            judge_model=args.judge_model,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
            judge_prompt_hash=judge_prompt_hash(),
            **reader_trim_pins(),
            # A retrieving system declares the knobs that move its score;
            # everything else returns {} and the fields stay None.
            **system.retrieval_pins(),
        )
        _print_pins(pins, declared_ctx)

        def predict_progress(done: int, total: int, q, truncated: bool) -> None:
            mark = "TRUNC" if truncated else "  ok "
            print(
                f"[{done}/{total}] read {mark}  {q.question_id} ({q.question_type})",
                file=sys.stderr,
            )

        predictions = predict(
            system,
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
            if auditing:
                pins = artifacts.read_pins_optional(source_dir)
                predictions = artifacts.read_predictions_file(predictions_path, Prediction)
            else:
                pins = artifacts.read_pins(source_dir)
                predictions = artifacts.read_predictions(source_dir, Prediction)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if pins:
            _print_pins(pins, None)
        else:
            print(
                "--- pins ---\nno pins.json or results.json beside the predictions "
                "file: grading rows without provenance.\n------------",
                file=sys.stderr,
            )
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

    # The fingerprint scopes cached verdicts to the judge that produced them.
    # Without it, changing the judge model or prompt replays stale verdicts and
    # reports "nothing changed" — a false null, not a cheap re-grade.
    cache = JudgeCache(
        judge_fingerprint=f"{args.judge_model}:{judge_prompt_hash()}",
    )

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

    print_report(args.system or source_dir.name, results)

    if auditing:
        _report_audit(source_dir, results, cache)
        return 0

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
            ollama_host=OLLAMA_HOST,
            serve_log=artifacts.default_serve_log(),
            save_log_to=directory / "model_load.log",
        ),
    }
    provisional = _provisional_reasons(pins)
    results_path = artifacts.write_results(
        directory, pins, results, run_meta, provisional, summary=report_summary(results)
    )

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


def _report_audit(source_dir: Path, results, cache) -> None:
    """Tier 1 verdict: does re-judging these predictions reproduce the score?

    Checks the recomputed score against the ``results.json`` published beside
    the predictions. This is the whole Tier 1 claim and its whole limit — it
    proves the SCORING step, not that the predictions came from the pipeline
    the pins header names. Auditable, never reproducible.
    """
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    published = artifacts.read_published_score(source_dir)
    if not total:
        print("\nTier 1 audit — no rows in the predictions file.", file=sys.stderr)
        return
    print(
        f"\nTier 1 audit - recomputed {correct}/{total} ({correct / total:.1%})",
        file=sys.stderr,
    )
    if published is None:
        print(
            f"no results.json beside {source_dir} to compare against - "
            "recomputed score printed above, nothing verified.",
            file=sys.stderr,
        )
    elif published == (correct, total):
        print(
            f"MATCHES the published {published[0]}/{published[1]}. The judge "
            "reproduces the published score from the published predictions.",
            file=sys.stderr,
        )
    else:
        print(
            f"DIVERGES from the published {published[0]}/{published[1]}. Either "
            "the judge moved (model, prompt templates, or their hash) or the "
            "predictions file was edited — both are bugs worth locating.",
            file=sys.stderr,
        )
    print(
        f"judge cache: {cache.hits} hit / {cache.misses} miss "
        "(a full-hit audit costs nothing; misses cost cents)",
        file=sys.stderr,
    )
    print(
        "This verifies scoring only. It does NOT verify that the predictions "
        "were generated by the pipeline the pins claim - that is Tier 2.",
        file=sys.stderr,
    )


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
        f"reader daemon:    flash_attn={pins.get('reader_flash_attention')} "
        f"cache_ram={pins.get('reader_cache_ram')} (verified resolved at preflight)",
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
    from .runner import READER_CACHE_RAM, READER_NUM_BATCH, READER_NUM_GPU

    reasons = []
    if pins.get("reader_cache_ram") != READER_CACHE_RAM:
        reasons.append(
            f"reader_cache_ram={pins.get('reader_cache_ram')} overrides the pin "
            f"({READER_CACHE_RAM}) — a live prompt cache makes a prediction "
            "depend on which request preceded it"
        )
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
