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
from . import artifacts, pricing
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
# evidence-availability bound, bar, mnimi. Declared once so `--system` choices and every error
# message quoting them cannot drift apart — they already had, listing two
# systems in a message the parser had also enumerated.
SYSTEMS = ("no_memory", "full_history", "oracle", "naive_rag", "mnimi")


def build_system(
    name: str,
    render_format: str = "text",
    dedup_scope: str = "store",
    query_instruction: str = "",
):
    """Construct a system by name. Imports are lazy — only mnimi and naive_rag
    need the embedder, and the other three must stay runnable without it.

    ``render_format`` is handed to every context-bearing arm — the harness
    pins ``render_template_hash(render_format)``, and one value for all arms
    is what keeps that pin true (SPEC: one renderer). ``dedup_scope`` reaches
    the two retrieval arms through one ``MemoryConfig``; only mnimi reads it
    and only mnimi pins it. ``query_instruction`` reaches both retrieval arms
    (their query path must match); ``"bge"`` names the model-card text."""
    if query_instruction == "bge":
        from mnimi.embeddings import BGE_QUERY_INSTRUCTION

        query_instruction = BGE_QUERY_INSTRUCTION
    if name == "no_memory":
        from .systems.no_memory import NoMemorySystem

        return NoMemorySystem()
    if name == "full_history":
        from .systems.full_history import FullHistorySystem

        return FullHistorySystem(render_format=render_format)
    if name == "oracle":
        from .systems.oracle import OracleSystem

        return OracleSystem(render_format=render_format)
    if name == "naive_rag":
        from mnimi import MemoryConfig

        from .systems.naive_rag import NaiveRagSystem

        return NaiveRagSystem(
            config=MemoryConfig(
                render_format=render_format,
                dedup_scope=dedup_scope,
                query_instruction=query_instruction,
            )
        )
    if name == "mnimi":
        from mnimi import MemoryConfig

        from .systems.mnimi import MnimiSystem

        return MnimiSystem(
            config=MemoryConfig(
                render_format=render_format,
                dedup_scope=dedup_scope,
                query_instruction=query_instruction,
            )
        )
    raise SystemExit(f"unknown system: {name}")


def _ollama_get(path: str):
    with urllib.request.urlopen(f"{OLLAMA_HOST}{path}", timeout=5) as resp:
        return json.load(resp)


def _ollama_post(path: str, payload: dict, timeout: float = 5):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# The warmup probe must outlive a model load. Ollama 0.32.13 ABORTS a load when
# the requesting client disconnects ("client connection closed before
# llama-server finished loading, aborting load" — measured 2026-08-16), so the
# old behaviour of firing the probe on a 5s timeout and letting the daemon
# finish loading anyway (which 0.32.5 did) now cancels the load outright and the
# load-bearing warmup never happens. 9.04s measured warm; cold loads and slower
# disks need real headroom.
_MODEL_LOAD_TIMEOUT_S = 300


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
            timeout=_MODEL_LOAD_TIMEOUT_S,
        )
    except (urllib.error.URLError, OSError, ValueError):
        pass  # preflight_reader_env reports the missing log far more usefully


def _live_ollama_version() -> str | None:
    """The serving daemon's build from ``GET /api/version``, or ``None``.

    ``None`` means "could not confirm" — unreachable endpoint, non-JSON body,
    or no ``version`` field — and preflight_reader_transport refuses on it.
    Fail closed here, never guess: the build is a pin.
    """
    try:
        return _ollama_get("/api/version")["version"]
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


def _recent_model_load_log() -> str:
    """The current runner's output from the daemon's serve log.

    Anchored at the runner-start marker, NOT sliced from the end of the file.
    The settings preflight checks are written once per spawn and never restated
    while the model stays warm, so a fixed-size tail silently stops covering
    them as a run gets longer — see ``artifacts.recent_model_load_block`` for
    the measured failure and for why reading the whole file is not the fix.
    """
    path = artifacts.default_serve_log()
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return artifacts.recent_model_load_block(text)


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


def _resume_extras(args) -> str:
    """The non-default flags a batch resume should repeat.

    The resume refuses on a pins-hash mismatch, so a hint that dropped a
    pin-bearing flag (``--render-format``) would send the user into that
    refusal. ``--verify-drift`` is not a pin, but it only fires when the
    predictions land, which on a no-wait submission is the resume.
    """
    extras = []
    if args.render_format != "text":
        extras.append(f"--render-format {args.render_format}")
    if args.dedup_scope != "store":
        extras.append(f"--dedup-scope {args.dedup_scope}")
    if args.query_instruction:
        extras.append(f'--query-instruction "{args.query_instruction}"')
    if args.verify_drift:
        extras.append(f"--verify-drift {args.verify_drift}")
    return "".join(f"{flag} " for flag in extras)


def _verify_drift(reference: str, directory: Path) -> int:
    """``--verify-drift``: compare the fresh predictions, leave drift.json, 0 or 2."""
    from . import drift

    try:
        drift.verify(Path(reference), directory)
    except (drift.DriftPairError, FileNotFoundError) as exc:
        print(f"ERROR: --verify-drift: {exc}", file=sys.stderr)
        return 2
    return 0


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
        "--reader-transport",
        default="ollama",
        choices=["ollama", "openai"],
        help="which reader family: 'ollama' (the local, bit-reproducible "
        "family; default) or 'openai' (the gpt-4o era). The two are never "
        "paired against each other.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"reader model. Default per transport: {READER_MODEL} (ollama) "
        "or gpt-4o-2024-08-06 (openai).",
    )
    parser.add_argument(
        "--api-budget-usd",
        type=float,
        default=None,
        help=f"override the programme's API budget (${pricing.API_BUDGET_USD:.2f}) "
        "for this run. Every run projects its cost from the real request bodies "
        "before the first call and refuses above the remaining budget; the "
        "override is echoed loudly and recorded in the ledger.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="openai transport only: send the predict stage through the Batch "
        "API (half price, 24h window). Resumable: re-run the same command with "
        "the same --run-dir to poll a submitted batch instead of paying again.",
    )
    parser.add_argument(
        "--batch-no-wait",
        action="store_true",
        help="with --batch: submit, write batch_state.json, print the resume "
        "command and exit 0 instead of polling.",
    )
    parser.add_argument(
        "--batch-enqueued-tokens",
        type=int,
        default=None,
        help="with --batch: the organization's per-model enqueued-token cap the "
        "run is chunked under (default 90,000, this key's usage tier; a batch "
        "over it fails validation). Sub-batches are submitted one at a time. "
        "Recorded in reader_resolved.json, never a pin.",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=60.0,
        help="with --batch: seconds between status polls (default 60).",
    )
    parser.add_argument(
        "--judge-model", default=JUDGE_MODEL, help="judge model id (OpenAI snapshot)"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help=f"reader context window. Default per transport: {DEFAULT_NUM_CTX} "
        "(ollama) or 128000 (openai, the model's window).",
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
        "--verify-drift",
        default=None,
        metavar="REFERENCE",
        help="predict stage: after predictions.jsonl is written (sync, batch, "
        "or a batch resume), compare it row by row against REFERENCE — a run "
        "directory or a predictions.jsonl of the same pins — print N/n changed "
        "with the first divergent byte of each row, and write drift.json beside "
        "the fresh predictions. The count is the Tier 2 statement. "
        "`python -m evals.drift A B` does the same for two existing artifacts.",
    )
    parser.add_argument(
        "--dedup-scope",
        default="store",
        choices=["store", "session"],
        help="mnimi's cosine dedup scope (MemoryConfig.dedup_scope): 'store' drops "
        "an incoming round whose nearest stored neighbour clears the threshold "
        "(v1 as shipped); 'session' only when that neighbour carries the same "
        "session timestamp. Pinned (schema /7). R3, DECISIONS 2026-09-12.",
    )
    parser.add_argument(
        "--query-instruction",
        default="",
        help="text prepended to the query before it is embedded, for both retrieval "
        "arms (MemoryConfig.query_instruction); '' as shipped, 'bge' for the "
        "BAAI/bge model-card retrieval instruction, or a literal. Pinned "
        "(schema /7). R5, DECISIONS 2026-09-12.",
    )
    parser.add_argument(
        "--render-format",
        default="text",
        choices=["text", "json"],
        help="how every context-bearing arm frames the reader's context: 'text' "
        "(the default: dated header + role-labelled lines) or 'json' (the same "
        "blocks as a JSON array, LongMemEval §5.5). Pinned as "
        "render_template_hash, so the two never pair. The gpt-4o era's "
        "presentation pair decides the era's value (DECISIONS 2026-09-12).",
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
    # python-dotenv ships with the [eval] extra; without it (CI installs the
    # dev extra alone) the environment is taken as-is.
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover — exercised by CI's dev-only install
        pass
    else:
        load_dotenv()

    # Library import, core deps only (no [embed] extra): the render template
    # is library-owned and its hash is a mandatory harness pin.
    from mnimi.memory import render_template_hash

    from .dataset import file_sha256, resolve_path
    from .judge import Judge, judge_block
    from .judge_cache import JudgeCache
    from .report import print_report
    from .report import summary as report_summary
    from .runner import (
        OPENAI_NUM_CTX,
        OPENAI_READER_MODEL,
        READER_CACHE_RAM,
        READER_FLASH_ATTENTION,
        READER_NUM_BATCH,
        READER_NUM_GPU,
        READER_NUM_THREAD,
        READER_PROMPT_VERSION,
        READER_SEED,
        READER_TOP_K,
        READER_TRANSPORT_OPENAI,
        READER_TRANSPORT_VERSION,
        Prediction,
        PredictStats,
        ReaderEnvError,
        judge_predictions,
        predict,
        preflight_reader_env,
        preflight_reader_transport,
        reader_prompt_hash,
        reader_trim_pins,
    )

    num_gpu = READER_NUM_GPU if args.num_gpu is None else args.num_gpu
    # Per-transport defaults, resolved here rather than in argparse so the
    # help text stays honest about both families.
    is_api = args.reader_transport == READER_TRANSPORT_OPENAI
    if args.model is None:
        args.model = OPENAI_READER_MODEL if is_api else READER_MODEL
    if args.num_ctx is None:
        args.num_ctx = OPENAI_NUM_CTX if is_api else DEFAULT_NUM_CTX
    if args.batch and not is_api:
        print(
            "ERROR: --batch is the OpenAI Batch API; it needs --reader-transport openai.",
            file=sys.stderr,
        )
        return 2

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
    if do_predict and is_api and args.system == "full_history":
        # Decided 2026-09-12 (DECISIONS, "Pre-registration for the gpt-4o
        # era"): the gpt-4o family cites the paper's full-context row instead
        # of running one. At the 128K window an n=100 sitting is ~$15 of the
        # $50 programme budget for a number the paper already reports.
        print(
            "ERROR: full_history is not run on the openai transport. The gpt-4o "
            "family cites the paper's untruncated full-context row instead "
            "(GPT-4o, Chain-of-Note, LongMemEval-S: 64.0%, Fig. 3b) - see "
            "docs/DECISIONS.md, 'Pre-registration for the gpt-4o era' (2026-09-12).",
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
    reader_client = None
    if do_predict and is_api:
        # The gpt-4o family. No daemon, no model load, no serve log: the
        # reader is a dated API snapshot. Two guards before any call.
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "ERROR: OPENAI_API_KEY is not set. The openai reader transport "
                f"({args.model}) runs on the OpenAI API; set it in .env and re-run.",
                file=sys.stderr,
            )
            return 2
        # A model without a price cannot be projected, and a run that cannot
        # be projected cannot spend (the budget gate below needs the number).
        try:
            pricing.price(args.model)
        except pricing.UnpricedModelError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        reader_client = build_openai_reader_client()
        # The snapshot must be served for this key before anything is built:
        # a retired snapshot found out at question 1 would already have cost
        # the ingest of the whole slice.
        snapshot_error = _preflight_snapshot(reader_client, args.model)
        if snapshot_error:
            print(f"ERROR: {snapshot_error}", file=sys.stderr)
            return 2
    elif do_predict:
        # Reader transport: local Ollama. Fail fast if daemon or model is missing.
        err, digest = ollama_preflight(args.model)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
        declared_ctx = ollama_context_length(args.model)

        # The server build is checked FIRST, before the warm-up load: it is
        # readable without loading anything, and a daemon on the wrong build
        # must be refused without touching VRAM or writing a run directory.
        # Then the daemon only reports its resolved flash-attention and
        # prompt-cache settings when it loads a model, so force the load the
        # run needs anyway and assert the serving daemon is the one the pins
        # describe. A run served by the tray app's daemon carries
        # `flash_attn = auto`, a live prompt cache and whatever build the
        # auto-updater last installed — a different configuration wearing this
        # configuration's pins_hash.
        try:
            preflight_reader_transport(_live_ollama_version())
            _force_model_load(args.model, args.num_ctx, num_gpu)
            preflight_reader_env(_recent_model_load_log())
        except ReaderEnvError as exc:
            print(f"ERROR: reader environment does not match pins: {exc}", file=sys.stderr)
            return 2

    started = time.perf_counter()
    budget_usd = pricing.API_BUDGET_USD if args.api_budget_usd is None else args.api_budget_usd
    budget_override = args.api_budget_usd is not None
    if budget_override:
        print(
            f"BUDGET OVERRIDE: --api-budget-usd {budget_usd:.2f} replaces the programme "
            f"cap of ${pricing.API_BUDGET_USD:.2f} for this run (recorded in the ledger).",
            file=sys.stderr,
        )
    # One ledger line per invocation: filled by the reader stage, the judge
    # stage, or both, and appended before main() returns. Money never enters
    # pins_hash; it is a fact about this key, kept in the gitignored .cache/.
    ledger_entry = {
        "ts": _ledger_ts(),
        "run_dir": str(directory),
        "system": args.system,
        "limit": args.limit,
        "stage": args.stage,
        "reader_transport": args.reader_transport if do_predict else None,
        "reader_model": args.model if (do_predict and is_api) else None,
        "batch": bool(args.batch and do_predict),
        "judge_model": args.judge_model if do_judge else None,
        "budget_usd": budget_usd,
        "budget_override": budget_override,
        "prices_as_of": pricing.PRICES_AS_OF,
        "status": "done",
    }

    if do_predict:
        dataset_path = resolve_path(args.dataset_file)
        # Built before the pins so the header describes the system that will
        # actually run — a retrieving system contributes its own pins below.
        # It also means a missing [embed] extra fails here, before the header,
        # rather than after it.
        system = build_system(
            args.system,
            render_format=args.render_format,
            dedup_scope=args.dedup_scope,
            query_instruction=args.query_instruction,
        )
        pins = artifacts.build_pins(
            dataset_file=str(dataset_path),
            dataset_sha256=file_sha256(dataset_path),
            system=args.system,
            limit=args.limit,
            sample_strategy=args.sample,
            sample_seed=args.sample_seed,
            reader_transport=args.reader_transport,
            reader_model=args.model,
            reader_digest=digest,
            reader_num_ctx=args.num_ctx,
            reader_seed=READER_SEED,
            # The six Ollama-only pins do not exist on the API; for that
            # family the "build that served" is the dated snapshot itself.
            reader_top_k=None if is_api else READER_TOP_K,
            reader_num_gpu=None if is_api else num_gpu,
            reader_num_thread=None if is_api else READER_NUM_THREAD,
            reader_num_batch=None if is_api else READER_NUM_BATCH,
            reader_flash_attention=None if is_api else READER_FLASH_ATTENTION,
            reader_cache_ram=None if is_api else READER_CACHE_RAM,
            reader_transport_version=args.model if is_api else READER_TRANSPORT_VERSION,
            reader_prompt_version=READER_PROMPT_VERSION,
            reader_prompt_hash=reader_prompt_hash(),
            # The canonical context format every arm renders through. Pinned
            # from the library constant so a format edit is loud in the header.
            render_template_hash=render_template_hash(args.render_format),
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

        if is_api:
            judge_calls_planned = 0
            if do_judge:
                # Upper bound: every row graded, no cache hit assumed.
                judge_calls_planned = args.limit if args.limit is not None else 0
            outcome = _predict_openai(
                args, directory, system, pins, reader_client, predict_progress,
                budget_usd=budget_usd, judge_calls_planned=judge_calls_planned,
                entry=ledger_entry,
            )
            if isinstance(outcome, int):
                return outcome
            predictions = outcome
        else:
            predict_stats = PredictStats()
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
                reader_transport=args.reader_transport,
                reader_client=reader_client,
                stats=predict_stats,
            )
            artifacts.write_pins(directory, pins)
            artifacts.write_predictions(directory, predictions)
            if is_api:
                # The API family's counterpart of model_load.log: what served.
                artifacts.write_reader_resolved(
                    directory, predict_stats.as_resolved(args.reader_transport, args.model)
                )
        print(f"wrote {directory / 'predictions.jsonl'}", file=sys.stderr)
        if args.verify_drift:
            rc = _verify_drift(args.verify_drift, directory)
            if rc:
                return rc
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
        # Schema /2 artifacts pinned a judge at predict time; /3 pins carry no
        # judge fields at all. Either way pins are never rewritten here — the
        # judge that actually grades is recorded in the results.json judge
        # block, refreshed below from the judge about to run.
        pinned_judge = pins.get("judge_model")
        if pinned_judge is not None and pinned_judge != args.judge_model:
            print(
                f"WARNING: judging with {args.judge_model} but this /2-era header "
                f"named {pinned_judge}; the judge block records the one used.",
                file=sys.stderr,
            )

    if not do_judge:
        elapsed = time.perf_counter() - started
        print(
            f"\npredict stage only: {len(predictions)} predictions in {elapsed:,.1f}s. "
            f"Grade them with:  python -m evals --system {args.system} "
            f"--limit {args.limit} --stage judge",
            file=sys.stderr,
        )
        if is_api:
            _record_spend(ledger_entry, budget_usd)
        return 0

    # Judge identity, built once from the judge about to run. It feeds both the
    # verdict-cache fingerprint and the results.json judge block, so what
    # invalidates the cache and what is recorded beside the verdicts cannot
    # drift apart.
    judge_info = judge_block(args.judge_model)
    _print_judge(judge_info)

    # The judge is API spend on every family. Same discipline as the reader:
    # priced, served, projected and gated before the first verdict is asked.
    try:
        pricing.price(args.judge_model)
    except pricing.UnpricedModelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    judge_client = build_openai_reader_client()
    snapshot_error = _preflight_snapshot(judge_client, args.judge_model)
    if snapshot_error:
        print(f"ERROR: {snapshot_error}", file=sys.stderr)
        return 2
    if not do_predict:
        # Judge-only invocation: the reader stage did not project, so this one
        # does. With the reader stage, the judge calls were inside its bound.
        judge_usd = len(predictions) * pricing.estimate_usd(
            args.judge_model, pricing.JUDGE_PROMPT_TOKENS_EST, pricing.JUDGE_COMPLETION_TOKENS_EST
        )
        print(
            f"projected: judge ${judge_usd:.4f} ({len(predictions)} calls, upper bound: "
            f"no cache hits assumed) | {pricing.status_line(budget_usd)}",
            file=sys.stderr,
        )
        refusal = pricing.gate(judge_usd, budget_usd, "judge stage")
        if refusal:
            print(f"ERROR: {refusal}", file=sys.stderr)
            return 2
        ledger_entry["projected_usd"] = judge_usd

    # The fingerprint scopes cached verdicts to the judge that produced them.
    # Without it, changing the judge model, prompt or decode config replays
    # stale verdicts and reports "nothing changed" — a false null, not a cheap
    # re-grade.
    cache = JudgeCache(
        judge_fingerprint=f"{judge_info['judge_model']}:{judge_info['judge_prompt_hash']}",
    )

    def judge_progress(done: int, total: int, p, correct: bool) -> None:
        mark = "PASS" if correct else "FAIL"
        print(f"[{done}/{total}] {mark}  {p.question_id} ({p.category})", file=sys.stderr)

    judge = Judge(args.judge_model, client=judge_client, cache=cache)
    results = judge_predictions(
        predictions,
        judge_model=args.judge_model,
        judge_cache=cache,
        progress=judge_progress,
        judge=judge,
    )
    elapsed = time.perf_counter() - started
    judge_usd_actual = pricing.estimate_usd(
        args.judge_model, judge.prompt_tokens, judge.completion_tokens
    )
    ledger_entry.update(
        judge_calls=judge.calls,
        judge_prompt_tokens=judge.prompt_tokens,
        judge_completion_tokens=judge.completion_tokens,
        judge_actual_usd=judge_usd_actual,
    )

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
        # Which framing the pinned render_template_hash names, for a human
        # reading results.json; the hash in pins.json is the pin.
        "render_format": args.render_format if do_predict else None,
        "elapsed_s": round(elapsed, 1),
        "n": len(results),
        "reader_mean_prompt_tokens": round(sum(fed) / len(fed)) if fed else None,
        "truncated": truncated_n,
        "tokens_dropped_estimated": dropped_total,
        "judge_cache_hits": getattr(cache, "hits", 0),
        "judge_cache_misses": getattr(cache, "misses", 0),
        "abstention_questions": sum(1 for r in results if r.is_abstention),
        # Diagnostic only — never folded into pins_hash (see capture_environment).
        # The family comes from the pins on disk, not the CLI flag, so a
        # judge-only replay of an API-era run never reads an Ollama daemon.
        "environment": _capture_environment_for(
            pins.get("reader_transport", args.reader_transport), directory
        ),
    }
    provisional = _provisional_reasons(pins)
    results_path = artifacts.write_results(
        directory,
        pins,
        results,
        run_meta,
        provisional,
        summary=report_summary(results),
        judge=judge_info,
    )

    mean_fed = f"{run_meta['reader_mean_prompt_tokens']:,}" if fed else "n/a"
    print(
        f"\nwall-clock: {elapsed:,.1f}s over {len(results)} q"
        f"  |  mean reader prompt tokens: {mean_fed}"
        f"  |  truncated: {truncated_n}/{len(results)} (~{dropped_total:,} tok dropped)"
        f"  |  judge cache: {getattr(cache, 'hits', 0)} hit / {getattr(cache, 'misses', 0)} miss",
        file=sys.stderr,
    )
    print(f"wrote {results_path}", file=sys.stderr)
    if provisional:
        print(
            "PROVISIONAL - not publishable: " + "; ".join(provisional),
            file=sys.stderr,
        )
    _record_spend(ledger_entry, budget_usd)
    return 0


def build_openai_reader_client():
    """The OpenAI client for the reader — the judge's shape, one seam for tests."""
    import openai

    return openai.OpenAI(max_retries=4)


def _capture_environment_for(reader_transport: str, directory: Path) -> dict:
    """The resolved-environment block for one family.

    Ollama: the daemon's resolved state from its serve log, unchanged. OpenAI:
    the machine (the embedder and, later, the extractor still run here) plus
    what the transport resolved to at predict time (``reader_resolved.json``).
    """
    if reader_transport != "openai":
        return artifacts.capture_environment(
            ollama_host=OLLAMA_HOST,
            serve_log=artifacts.default_serve_log(),
            save_log_to=directory / "model_load.log",
        )
    machine = artifacts.capture_environment(ollama_host="http://127.0.0.1:1", serve_log=None)
    env = {
        "reader_transport": "openai",
        "gpu_model": machine.get("gpu_model"),
        "driver_version": machine.get("driver_version"),
        "cuda_version": machine.get("cuda_version"),
        "reader_resolved": artifacts.read_reader_resolved_optional(directory),
    }
    return env


def _ledger_ts() -> str:
    """A per-invocation id for ledger lines (microsecond resolution so two
    quick invocations never share one)."""
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1e6):06d}"


def _record_spend(entry: dict, budget_usd: float) -> None:
    """Append the invocation's ledger line and say where the budget stands."""
    reader = entry.get("reader_actual_usd") or 0.0
    judge = entry.get("judge_actual_usd") or 0.0
    entry["actual_usd"] = reader + judge
    pricing.append(entry)
    print(
        f"actual API spend this run: ${entry['actual_usd']:.4f} "
        f"(reader ${reader:.4f}, judge ${judge:.4f}) | {pricing.status_line(budget_usd)}",
        file=sys.stderr,
    )


def _preflight_snapshot(client, model: str) -> str | None:
    """Confirm the dated snapshot is served for this key, before any spend.

    Classified on the SDK error's ``status_code`` rather than on its classes,
    so this module never imports ``openai`` at call time: the harness tests
    run under the dev extra alone (CI's rule that the core needs no eval deps),
    and a fake client only has to raise something with a status code.
    """
    try:
        client.models.retrieve(model)
    except Exception as exc:  # noqa: BLE001 — every failure is a refusal
        status = getattr(exc, "status_code", None)
        if status == 404:
            return (
                f"snapshot {model!r} is not served for this API key "
                "(models.retrieve → 404). A retired or mistyped snapshot cannot be "
                "the pinned reader; see docs/DECISIONS.md for the pinned family."
            )
        if status == 401:
            return f"OpenAI rejected the API key while checking snapshot {model!r}: {exc}"
        # Fail closed: an unknown state is not a pass.
        return f"could not confirm snapshot {model!r} is served: {exc}"
    return None


def _ctx_progress(done: int, total: int, q, truncated: bool) -> None:
    """Progress for the context-building pass (no reader call yet)."""
    mark = "TRUNC" if truncated else "  ok "
    print(f"[{done}/{total}] ctx  {mark}  {q.question_id} ({q.category})", file=sys.stderr)


def _announce_and_gate(projection, budget_usd: float) -> str | None:
    """Print the projection and the budget position; return a refusal or None."""
    print(f"{projection.line()} | {pricing.status_line(budget_usd)}", file=sys.stderr)
    return pricing.gate(projection.total_usd, budget_usd, projection.basis)


def _predict_openai(
    args, directory: Path, system, pins: dict, client, progress, *,
    budget_usd: float, judge_calls_planned: int, entry: dict,
):
    """The gpt-4o family's predict stage: build every request first, project
    and gate, then spend — synchronously or through the Batch API."""
    from .runner import (
        OpenAIReader,
        PredictStats,
        answer_items_sync,
        build_batch_items,
        predictions_from_batch,
    )

    if args.batch:
        return _predict_batch(
            args, directory, system, pins, client, progress,
            budget_usd=budget_usd, judge_calls_planned=judge_calls_planned, entry=entry,
        )
    reader = OpenAIReader(args.model, num_ctx=args.num_ctx, client=client)
    items = build_batch_items(
        system, reader, limit=args.limit, dataset_file=args.dataset_file,
        strategy=args.sample, sample_seed=args.sample_seed, progress=_ctx_progress,
    )
    projection = pricing.project(
        model=args.model, items=items, batch=False,
        judge_model=args.judge_model if judge_calls_planned else None,
        judge_calls=judge_calls_planned,
    )
    refusal = _announce_and_gate(projection, budget_usd)
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 2
    entry["projected_usd"] = projection.total_usd

    def read_progress(done: int, total: int, item) -> None:
        mark = "TRUNC" if item.truncated else "  ok "
        print(f"[{done}/{total}] read {mark}  {item.custom_id} ({item.category})", file=sys.stderr)

    outputs = answer_items_sync(client, items, progress=read_progress)
    stats = PredictStats()
    predictions = predictions_from_batch(items, outputs, stats=stats)
    artifacts.write_pins(directory, pins)
    artifacts.write_predictions(directory, predictions)
    if args.verify_drift:
        rc = _verify_drift(args.verify_drift, directory)
        if rc:
            return rc
    reader_usd = pricing.estimate_usd(args.model, stats.prompt_tokens, stats.completion_tokens)
    resolved = stats.as_resolved(args.reader_transport, args.model)
    resolved["projected_usd"] = projection.total_usd
    resolved["actual_usd"] = reader_usd
    artifacts.write_reader_resolved(directory, resolved)
    entry.update(
        reader_prompt_tokens=stats.prompt_tokens,
        reader_completion_tokens=stats.completion_tokens,
        reader_actual_usd=reader_usd,
    )
    return predictions


def _predict_batch(
    args, directory: Path, system, pins: dict, client, progress, *,
    budget_usd: float, judge_calls_planned: int, entry: dict,
):
    """The predict stage through the Batch API. Returns the predictions, or an
    exit code.

    Resumable at every step: the requests file and ``pins.json`` are written
    before submission, ``batch_state.json`` right after it, and a later run of
    the same command in the same ``--run-dir`` polls the recorded batch
    instead of resubmitting. Items the batch did not answer (an error line,
    an expired window) are filled with ONE synchronous call each, using the
    item's own body, and listed in ``reader_resolved.json``.
    """
    from . import batch as batch_mod
    from .runner import (
        _CHARS_PER_TOKEN,
        BatchItem,
        OpenAIReader,
        PredictStats,
        build_batch_items,
        predictions_from_batch,
    )

    pins_hash = artifacts.pins_hash(pins)
    state = artifacts.read_batch_state_optional(directory)
    resuming = state is not None and not (directory / "predictions.jsonl").exists()
    enqueued_limit = (
        args.batch_enqueued_tokens
        if args.batch_enqueued_tokens is not None
        else batch_mod.ENQUEUED_TOKEN_LIMIT
    )

    if resuming:
        if state.get("pins_hash") != pins_hash:
            print(
                f"ERROR: {directory} holds batch {state.get('batch_id')} submitted under "
                f"pins_hash {state.get('pins_hash')}, but this command's pins hash to "
                f"{pins_hash}. A resume must be the same configuration; use another "
                "--run-dir for a different one.",
                file=sys.stderr,
            )
            return 2
        state = batch_mod.upgrade_state(state)
        enqueued_limit = state.get("enqueued_token_limit", enqueued_limit)
        bodies = batch_mod.read_requests_jsonl(directory / artifacts.BATCH_REQUESTS_FILE)
        items = [BatchItem(body=bodies[meta["custom_id"]], **meta) for meta in state["items"]]
        submitted_n = sum(1 for c in state["chunks"] if c.get("batch_id"))
        print(
            f"resuming {directory}: {submitted_n}/{len(state['chunks'])} chunk(s) submitted, "
            f"last seen {state.get('status')}",
            file=sys.stderr,
        )
    else:
        reader = OpenAIReader(args.model, num_ctx=args.num_ctx, client=client)
        items = build_batch_items(
            system,
            reader,
            limit=args.limit,
            dataset_file=args.dataset_file,
            strategy=args.sample,
            sample_seed=args.sample_seed,
            progress=_ctx_progress,
        )
        projection = pricing.project(
            model=args.model, items=items, batch=True,
            judge_model=args.judge_model if judge_calls_planned else None,
            judge_calls=judge_calls_planned,
        )
        refusal = _announce_and_gate(projection, budget_usd)
        if refusal:
            print(f"ERROR: {refusal}", file=sys.stderr)
            return 2
        entry["projected_usd"] = projection.total_usd
        batch_mod.write_requests_jsonl(directory / artifacts.BATCH_REQUESTS_FILE, items)
        artifacts.write_pins(directory, pins)
        chunks = batch_mod.plan_chunks(items, enqueued_limit, _CHARS_PER_TOKEN)
        print(
            f"planned {len(items)} requests as {len(chunks)} sub-batch(es) under the "
            f"{enqueued_limit:,} enqueued-token cap, submitted one at a time",
            file=sys.stderr,
        )
        state = {
            "batch_state_schema": batch_mod.BATCH_STATE_SCHEMA,
            "batch_id": None,
            "status": "planned",
            "pins_hash": pins_hash,
            "reader_model": args.model,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "enqueued_token_limit": enqueued_limit,
            "items": [_batch_item_meta(item) for item in items],
            "chunks": batch_mod.chunk_states(chunks),
            # The ledger line this submission is booked under, so the line the
            # resume writes can supersede it instead of double counting.
            "ledger_ts": entry["ts"],
        }
        artifacts.write_batch_state(directory, state)

    by_id = {item.custom_id: item for item in items}
    n_chunks = len(state["chunks"])

    def submit_chunk(chunk: dict) -> None:
        chunk_items = [by_id[cid] for cid in chunk["custom_ids"]]
        path = batch_mod.write_requests_jsonl(
            directory / batch_mod.chunk_file(chunk["index"]), chunk_items
        )
        submitted = batch_mod.submit(
            client,
            path,
            metadata={
                "run_dir": str(directory)[:512],
                "system": str(args.system),
                "pins_hash": pins_hash[:64],
                "chunk": f"{chunk['index'] + 1}/{n_chunks}",
            },
        )
        chunk.update(
            batch_id=submitted.id,
            status=submitted.status,
            input_file_id=getattr(submitted, "input_file_id", None),
            attempts=chunk.get("attempts", 0) + 1,
        )
        state["batch_id"] = ",".join(c["batch_id"] for c in state["chunks"] if c.get("batch_id"))
        # The run-level status mirrors the chunk in flight; "done" at the end.
        state["status"] = submitted.status
        artifacts.write_batch_state(directory, state)
        tokens = sum(
            batch_mod.enqueued_tokens(item.body, _CHARS_PER_TOKEN) for item in chunk_items
        )
        print(
            f"submitted chunk {chunk['index'] + 1}/{n_chunks} as batch {submitted.id}: "
            f"{len(chunk_items)} requests (~{tokens:,} enqueued tokens), status "
            f"{submitted.status}",
            file=sys.stderr,
        )

    def on_change(batch_obj) -> None:
        counts = batch_mod.request_counts(batch_obj)
        print(
            f"batch {batch_obj.id}: {batch_obj.status} "
            f"{counts.get('completed')}/{counts.get('total')} completed",
            file=sys.stderr,
        )

    finals: list = []
    outputs: dict = {}
    errors: dict = {}
    for chunk in state["chunks"]:
        while True:
            if not chunk.get("batch_id"):
                if chunk.get("attempts", 0) >= batch_mod.CHUNK_RETRIES:
                    print(
                        f"ERROR: chunk {chunk['index'] + 1}/{n_chunks} failed validation "
                        f"{chunk['attempts']} times; nothing written. The state file is kept; "
                        "re-run the same command later (the cap is shared by every batch "
                        "in progress on this key).",
                        file=sys.stderr,
                    )
                    return 2
                submit_chunk(chunk)
                if args.batch_no_wait:
                    print(
                        "Resume later (polls the batch, submits the remaining chunks, then "
                        "writes predictions.jsonl) with:\n"
                        f"  python -m evals --system {args.system} --limit {args.limit} "
                        f"--stage {args.stage} --reader-transport openai --batch "
                        f"{_resume_extras(args)}--run-dir {directory}",
                        file=sys.stderr,
                    )
                    if not resuming:
                        # Booked at its projection until the resume replaces this line.
                        entry["status"] = "submitted"
                        pricing.append(entry)
                        print(pricing.status_line(budget_usd), file=sys.stderr)
                    return 0
            final = batch_mod.wait(
                client, chunk["batch_id"], poll_seconds=args.batch_poll_seconds,
                progress=on_change,
            )
            chunk["status"] = state["status"] = final.status
            artifacts.write_batch_state(directory, state)
            if final.status in batch_mod.WITH_OUTPUT:
                break
            reasons = batch_mod.summary(final)["batch_errors"]
            if batch_mod.never_ran(final):
                # Validation refused it and nothing was enqueued or charged —
                # typically the enqueued-token cap while an earlier batch was
                # still in progress. Resubmit the same chunk after a pause.
                print(
                    f"batch {final.id} ended {final.status} before running "
                    f"({reasons}); resubmitting chunk {chunk['index'] + 1}/{n_chunks} "
                    f"(attempt {chunk.get('attempts', 0) + 1}/{batch_mod.CHUNK_RETRIES})",
                    file=sys.stderr,
                )
                chunk["batch_id"] = None
                artifacts.write_batch_state(directory, state)
                if args.batch_poll_seconds > 0:
                    time.sleep(args.batch_poll_seconds)
                continue
            print(
                f"ERROR: batch {final.id} ended {final.status}; nothing written."
                + (f" Batch errors: {reasons}." if reasons else "")
                + " The state file is kept for the record; resubmit into a fresh --run-dir.",
                file=sys.stderr,
            )
            return 2
        finals.append(final)
        chunk_outputs, chunk_errors = batch_mod.fetch_outputs(client, final)
        outputs.update(chunk_outputs)
        errors.update(chunk_errors)

    outputs, fallbacks = batch_mod.complete_outputs(client, items, outputs)
    if fallbacks:
        print(
            f"{len(fallbacks)} request(s) had no batch output and were answered "
            f"synchronously: {fallbacks}",
            file=sys.stderr,
        )
    stats = PredictStats()
    predictions = predictions_from_batch(items, outputs, stats=stats)
    artifacts.write_predictions(directory, predictions)
    if args.verify_drift:
        rc = _verify_drift(args.verify_drift, directory)
        if rc:
            return rc
    # Batch rate for the batch rows; the fallbacks were synchronous calls.
    # Their usage is inside the same totals, so bill the fallback share at
    # standard rate and the rest at batch rate.
    fallback_prompt = fallback_completion = 0
    for cid in fallbacks:
        p_tok, c_tok = _usage_of(outputs[cid])
        fallback_prompt += p_tok or 0
        fallback_completion += c_tok or 0
    reader_usd = pricing.estimate_usd(
        args.model, stats.prompt_tokens - fallback_prompt,
        stats.completion_tokens - fallback_completion, batch=True,
    ) + pricing.estimate_usd(args.model, fallback_prompt, fallback_completion)
    resolved = stats.as_resolved(args.reader_transport, args.model)
    resolved.update(batch_mod.merged_summary(finals, enqueued_token_limit=enqueued_limit))
    resolved["sync_fallbacks"] = fallbacks
    resolved["batch_errors_by_id"] = errors
    resolved["projected_usd"] = entry.get("projected_usd")
    resolved["actual_usd"] = reader_usd
    artifacts.write_reader_resolved(directory, resolved)
    state["status"] = "done"
    artifacts.write_batch_state(directory, state)
    if resuming and state.get("ledger_ts"):
        entry["supersedes_ts"] = state["ledger_ts"]
        entry.setdefault("projected_usd", None)
    entry.update(
        reader_prompt_tokens=stats.prompt_tokens,
        reader_completion_tokens=stats.completion_tokens,
        reader_actual_usd=reader_usd,
        batch_id=resolved["batch_id"],
    )
    return predictions


def _usage_of(completion) -> tuple[int | None, int | None]:
    from .runner import completion_usage

    return completion_usage(completion)


def _batch_item_meta(item) -> dict:
    """The item without its body (the body lives in batch_requests.jsonl)."""
    return {
        "custom_id": item.custom_id,
        "category": item.category,
        "is_abstention": item.is_abstention,
        "question": item.question,
        "answer": item.answer,
        "truncated": item.truncated,
        "tokens_dropped": item.tokens_dropped,
    }


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
    transport = pins.get("reader_transport", "ollama")
    digest = "" if transport == "openai" else f"  digest={pins.get('reader_digest')}"
    print(f"reader ({transport}):  {pins.get('reader_model')}{digest}", file=sys.stderr)
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
    if pins.get("reader_transport") == "openai":
        print(
            f"reader transport: openai {pins.get('reader_model')} (snapshot; "
            "fingerprints recorded per call, seed/temperature/max_tokens pinned)",
            file=sys.stderr,
        )
    else:
        print(
            f"reader daemon:    flash_attn={pins.get('reader_flash_attention')} "
            f"cache_ram={pins.get('reader_cache_ram')} (verified resolved at preflight)",
            file=sys.stderr,
        )
        print(
            f"reader transport: ollama {pins.get('reader_transport_version')} "
            "(verified at preflight)",
            file=sys.stderr,
        )
    print(
        f"render template:  {str(pins.get('render_template_hash'))[:12]}...",
        file=sys.stderr,
    )
    print(f"pins_hash:        {artifacts.pins_hash(pins)}", file=sys.stderr)
    print("------------", file=sys.stderr)


def _print_judge(judge: dict) -> None:
    """Echo the judge block — grading provenance, separate from predict pins."""
    print("--- judge ---", file=sys.stderr)
    print(
        f"{judge.get('judge_model')}  prompts={judge.get('judge_prompt_version')} "
        f"({str(judge.get('judge_prompt_hash'))[:12]}...)  "
        f"temperature={judge.get('judge_temperature')} "
        f"max_tokens={judge.get('judge_max_tokens')}",
        file=sys.stderr,
    )
    if judge.get("judge_model") in {"gpt-4o", "gpt-4o-mini"}:
        print(
            "WARNING: judge model looks like a rolling alias, not a pinned snapshot.",
            file=sys.stderr,
        )
    print("-------------", file=sys.stderr)


def _provisional_reasons(pins: dict) -> list[str]:
    """Why this artifact is not yet a publishable number. Empty list = it is."""
    from .runner import READER_CACHE_RAM, READER_NUM_BATCH, READER_NUM_GPU

    reasons = []
    # The daemon and decode-pin checks describe the Ollama family; on the API
    # family those pins are None by construction and carry no meaning.
    is_api = pins.get("reader_transport") == "openai"
    if not is_api and pins.get("reader_cache_ram") != READER_CACHE_RAM:
        reasons.append(
            f"reader_cache_ram={pins.get('reader_cache_ram')} overrides the pin "
            f"({READER_CACHE_RAM}) — a live prompt cache makes a prediction "
            "depend on which request preceded it"
        )
    # mnimi-con-v1 is the pinned Phase D prompt. "json-con-paper-v1" stays
    # RESERVED for a future byte-exact reproduction of the paper's prompt with
    # zero deviations; it is not this gate's target and must not be reused.
    if pins.get("reader_prompt_version") != "mnimi-con-v1":
        reasons.append(
            f"reader prompt is {pins.get('reader_prompt_version')} "
            "(the pinned Phase D prompt is mnimi-con-v1)"
        )
    if str(pins.get("harness_git_sha", "")).endswith("-dirty"):
        reasons.append("harness tree was dirty at run time")
    # A deviated decode pin means this run is not the configuration the
    # determinism evidence was gathered under.
    if not is_api and pins.get("reader_num_gpu") != READER_NUM_GPU:
        reasons.append(
            f"reader_num_gpu={pins.get('reader_num_gpu')} overrides the pin "
            f"({READER_NUM_GPU}) — not the measured-reproducible configuration"
        )
    if not is_api and pins.get("reader_num_batch") != READER_NUM_BATCH:
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
