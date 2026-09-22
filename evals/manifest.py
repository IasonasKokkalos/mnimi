"""The run manifest, the run registry and the promotion gate (the run-documentation rule).

Rule set by the maintainer on 2026-09-22, so that every run is usable in the
paper without reconstruction. Three files and one gate:

* ``runs/<run_id>/manifest.json`` — one document per run holding what the
  three staged artifacts (``pins.json``, ``predictions.jsonl``,
  ``results.json``) spread across themselves plus what they never held:
  purpose, claim, the rule commit, the served model strings, the question-id
  list digest, the environment, the cost and the timings. Every required
  field is listed in :data:`REQUIRED_FIELDS`; a run whose manifest is missing
  one is ``incomplete`` and the missing names are printed.
* ``runs/INDEX.md`` — the registry: one row per run including failed and
  aborted ones, append-only; only the ``status`` column may change (and a
  ``pending`` score may be filled once, when the judge stage lands).
* ``analyses/<a>__vs__<b>.json`` — every paired comparison, written by
  ``python -m evals.stats``, computed from per-question rows and never from
  summary scores (see :mod:`evals.stats`).
* Promotion (``python -m evals.publish``): only a clean-tree run with a
  complete manifest may be copied to ``results/published/``.

``UNKNOWN`` is a value, not an absence: a backfilled manifest writes it where a
field cannot be recovered from the artifacts on disk, and :func:`check`
reports those separately from fields that are absent altogether. Nothing here
guesses.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import artifacts

MANIFEST_SCHEMA = "mnimi-run-manifest/1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
INDEX_FILE = "INDEX.md"
UNKNOWN = "UNKNOWN"
ANALYSES_DIR = "analyses"

STATUSES = ("published", "provisional", "aborted", "incomplete", "complete")

#: Dotted paths that must be present (not ``None``) for a manifest to be
#: complete. ``UNKNOWN`` counts as present-but-unrecovered and is listed
#: separately by :func:`check`.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id", "created_utc", "purpose", "claim",
    "code.commit", "code.clean_tree", "code.mnimi_version", "code.harness_version",
    "arm.system",
    "arm.config.top_k", "arm.config.token_budget", "arm.config.chunk_unit",
    "arm.config.dedup", "arm.config.consolidate", "arm.config.embedder",
    "arm.config.embedder_revision", "arm.config.render_template_hash",
    "arm.config.embed_template_hash",
    "reader.requested_model", "reader.served_models", "reader.temperature",
    "reader.max_tokens", "reader.seed", "reader.prompt_id",
    "judge.model", "judge.prompt_source", "judge.replay_count",
    "dataset.name", "dataset.n", "dataset.question_ids_sha256",
    "environment.python", "environment.lockfile_hash", "environment.hardware",
    "environment.daemon_flags",
    "cost.reader_usd", "cost.judge_usd", "cost.tokens_in", "cost.tokens_out",
    "cost.wall_clock_s", "cost.ingest_s", "cost.llm_calls_at_write",
)

INDEX_COLUMNS = ("run_id", "date", "arm", "reader", "n", "score", "clean", "status", "claim")
INDEX_HEADER = (
    "# Run registry\n\n"
    "One row per run, including failed and aborted ones. Append-only: only the "
    "`status` column may change (and a `pending` score is filled once, when the judge "
    "stage lands). Never delete a row. Status is one of published / provisional / "
    "aborted / incomplete / complete. See `evals/manifest.py`.\n\n"
    "| " + " | ".join(INDEX_COLUMNS) + " |\n"
    "| " + " | ".join("---" for _ in INDEX_COLUMNS) + " |\n"
)


def utc_now() -> str:
    """Wall-clock, harness side only (the library never reads a clock)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_status_porcelain() -> str | None:
    """``git status --porcelain`` verbatim, ``None`` when git is unavailable."""
    return artifacts._git("status", "--porcelain")


def mnimi_version(root: Path | None = None) -> str | None:
    """The version of record: ``pyproject.toml``, not the (possibly stale) install metadata."""
    root = root or Path(__file__).resolve().parents[1]
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def harness_version() -> dict:
    return {"artifact_schema": artifacts.ARTIFACT_SCHEMA, "manifest_schema": MANIFEST_SCHEMA}


def environment_digest() -> dict:
    """The installed distributions, digested, in place of a lockfile the repo does not have.

    Hashed as ``name==version`` lines sorted, so two environments with the same
    packages at the same versions share a digest wherever they were installed.
    """
    dists = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata and d.metadata.get("Name")
    )
    return {
        "lockfile_hash": artifacts.fingerprint("\n".join(dists)),
        "lockfile_source": "sha256 of the installed distributions (name==version, sorted); "
        "the repo has no lockfile",
        "distributions": len(dists),
    }


def hardware_block(environment: dict | None) -> dict:
    """GPU/driver/CUDA from the run's environment block plus the host's CPU."""
    env = environment or {}
    return {
        "gpu_model": env.get("gpu_model"),
        "driver_version": env.get("driver_version"),
        "cuda_version": env.get("cuda_version"),
        "processor": platform.processor() or platform.machine() or None,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }


def daemon_flags(pins: dict, environment: dict | None) -> dict | None:
    """The local reader's daemon-level settings; a statement, not a blank, on the API family."""
    if pins.get("reader_transport") == "openai":
        return {"daemon": "none (API family: a dated snapshot served by the provider)"}
    env = environment or {}
    return {
        "ollama_version": env.get("ollama_version"),
        "flash_attention_reported": env.get("flash_attention_reported"),
        "prompt_cache_reported": env.get("prompt_cache_reported"),
        "kv_cache_type": env.get("kv_cache_type"),
        "ollama_env_daemon": env.get("ollama_env_daemon"),
        "reader_num_gpu": pins.get("reader_num_gpu"),
        "reader_num_batch": pins.get("reader_num_batch"),
        "reader_num_thread": pins.get("reader_num_thread"),
    }


def question_ids_sha256(question_ids: list[str]) -> str:
    """Digest of the question-id list in order — which questions, and in what order."""
    return artifacts.fingerprint(artifacts.canonical(list(question_ids)))


def arm_config(pins: dict) -> dict:
    """The number-determining configuration, named the way the rule names it."""
    dedup = None
    if pins.get("dedup_cosine_threshold") is not None:
        dedup = {
            "on": pins.get("system") == "mnimi",
            "cosine_threshold": pins.get("dedup_cosine_threshold"),
            "scope": pins.get("dedup_scope"),
            "entropy_gate": pins.get("dedup_entropy_gate"),
        }
    elif pins.get("system") in {"no_memory", "full_history", "oracle", "naive_rag"}:
        dedup = {"on": False}
    return {
        "top_k": pins.get("k") if pins.get("k") is not None else (
            0 if pins.get("system") in {"no_memory", "full_history", "oracle"} else None
        ),
        "token_budget": pins.get("reader_num_ctx"),
        "answer_reserve": pins.get("reader_answer_reserve"),
        "chunk_unit": (
            "round" if pins.get("chunk_tokens") in (0, None) else
            f"window:{pins.get('chunk_tokens')}/{pins.get('chunk_overlap')}"
        ) if pins.get("system") in {"mnimi", "naive_rag"} else (
            "session" if pins.get("system") in {"full_history", "oracle"} else "none"
        ),
        "dedup": dedup,
        "consolidate": bool(pins.get("consolidate")) if pins.get("system") == "mnimi" else False,
        "embedder": pins.get("embedder_name") or "none",
        "embedder_revision": pins.get("embedder_revision") or "none",
        "embedder_dim": pins.get("embedder_dim"),
        "render_template_hash": pins.get("render_template_hash"),
        "embed_template_hash": pins.get("embed_template_hash") or "none",
        "fact_embed_template_hash": pins.get("fact_embed_template_hash"),
        "render_unit": pins.get("render_unit"),
        "extractor_model": pins.get("extractor_model"),
        "ranking": pins.get("ranking"),
        "active_only": pins.get("active_only"),
        "salience_weights": pins.get("salience_weights"),
        "conflict_resolution": pins.get("conflict_resolution"),
        "query_instruction": pins.get("query_instruction"),
    }


def build(
    *,
    run_id: str,
    purpose: str | None,
    claim: str | None,
    rule_commit: str | None,
    pins: dict,
    porcelain: str | None,
    served_models: dict | None,
    served_fingerprints: dict | None,
    judge: dict | None,
    replay_count: int | None,
    question_ids: list[str] | None,
    environment: dict | None,
    cost: dict,
    outputs: dict,
    created_utc: str | None = None,
    env_digest: dict | None = None,
) -> dict:
    """Assemble a manifest from the pieces the harness has at the end of a stage.

    Every value comes from something recorded — the pins, the reader's
    resolved block, the judge's counters, the dataset — never from a default
    that would pass as a measurement. A field the caller does not have is
    ``None`` and shows up in :func:`check`.
    """
    sha = pins.get("harness_git_sha")
    clean = None if sha is None else not str(sha).endswith("-dirty")
    digest = env_digest or environment_digest()
    manifest = {
        "manifest_schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_utc": created_utc or utc_now(),
        "updated_utc": utc_now(),
        "purpose": purpose,
        "claim": claim,
        "rule_commit": rule_commit,
        "code": {
            "commit": None if sha is None else str(sha).removesuffix("-dirty"),
            "clean_tree": clean,
            "git_status_porcelain": porcelain,
            "mnimi_version": mnimi_version(),
            "harness_version": harness_version(),
        },
        "arm": {
            "system": pins.get("system"),
            "config": arm_config(pins),
            "pins_hash": artifacts.pins_hash(pins) if pins else None,
        },
        "reader": {
            "transport": pins.get("reader_transport"),
            "requested_model": pins.get("reader_model"),
            "served_models": served_models,
            "served_fingerprints": served_fingerprints,
            "transport_version": pins.get("reader_transport_version"),
            "temperature": 0 if pins.get("reader_model") else None,
            "max_tokens": pins.get("reader_answer_reserve"),
            "seed": pins.get("reader_seed"),
            "num_ctx": pins.get("reader_num_ctx"),
            "prompt_id": pins.get("reader_prompt_version"),
            "prompt_hash": pins.get("reader_prompt_hash"),
        },
        "judge": {
            "model": (judge or {}).get("judge_model"),
            "prompt_id": (judge or {}).get("judge_prompt_version"),
            "prompt_hash": (judge or {}).get("judge_prompt_hash"),
            "prompt_source": (
                f"evals/judge.py ({(judge or {}).get('judge_prompt_version')}), "
                "LongMemEval's paper judge templates"
                if judge else None
            ),
            "temperature": (judge or {}).get("judge_temperature"),
            "max_tokens": (judge or {}).get("judge_max_tokens"),
            "replay_count": replay_count,
        },
        "dataset": {
            "name": Path(str(pins.get("dataset_file"))).name if pins.get("dataset_file") else None,
            "file": pins.get("dataset_file"),
            "sha256": pins.get("dataset_sha256"),
            "n": len(question_ids) if question_ids is not None else pins.get("limit"),
            "question_ids_sha256": (
                question_ids_sha256(question_ids) if question_ids is not None else None
            ),
            "sample_strategy": pins.get("sample_strategy"),
            "sample_seed": pins.get("sample_seed"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "lockfile_hash": digest.get("lockfile_hash"),
            "lockfile_source": digest.get("lockfile_source"),
            "hardware": hardware_block(environment),
            "daemon_flags": daemon_flags(pins, environment),
        },
        "cost": {
            "reader_usd": cost.get("reader_usd"),
            "judge_usd": cost.get("judge_usd"),
            "tokens_in": cost.get("tokens_in"),
            "tokens_out": cost.get("tokens_out"),
            "reader_prompt_tokens": cost.get("reader_prompt_tokens"),
            "reader_completion_tokens": cost.get("reader_completion_tokens"),
            "judge_prompt_tokens": cost.get("judge_prompt_tokens"),
            "judge_completion_tokens": cost.get("judge_completion_tokens"),
            "wall_clock_s": cost.get("wall_clock_s"),
            "predict_wall_s": cost.get("predict_wall_s"),
            "judge_wall_s": cost.get("judge_wall_s"),
            "ingest_s": cost.get("ingest_s"),
            "llm_calls_at_write": cost.get("llm_calls_at_write"),
        },
        "outputs": outputs,
        "status": None,
        "missing_fields": [],
        "unknown_fields": [],
    }
    return manifest


def _get(manifest: dict, dotted: str):
    node = manifest
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check(manifest: dict) -> tuple[list[str], list[str]]:
    """``(missing, unknown)``: fields ``None`` or absent, and fields written ``UNKNOWN``."""
    missing, unknown = [], []
    for field in REQUIRED_FIELDS:
        value = _get(manifest, field)
        if value is None:
            missing.append(field)
        elif value == UNKNOWN:
            unknown.append(field)
    return missing, unknown


def finalize(manifest: dict, provisional: list[str] | None, aborted: bool = False) -> dict:
    """Set ``status`` from the checks and the provisional reasons; record what is missing."""
    missing, unknown = check(manifest)
    manifest["missing_fields"] = missing
    manifest["unknown_fields"] = unknown
    if aborted:
        manifest["status"] = "aborted"
    elif missing:
        manifest["status"] = "incomplete"
    elif provisional or manifest["code"].get("clean_tree") is False:
        manifest["status"] = "provisional"
    else:
        manifest["status"] = "complete"
    manifest["provisional_reasons"] = list(provisional or [])
    manifest["updated_utc"] = utc_now()
    return manifest


def write(directory: Path, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_optional(directory: Path) -> dict | None:
    path = directory / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def merge_into(existing: dict | None, fresh: dict) -> dict:
    """A later stage adds to an earlier stage's manifest and never blanks a recorded value."""
    if existing is None:
        return fresh
    merged = json.loads(json.dumps(existing))
    for key, value in fresh.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            for sub, sub_value in value.items():
                if sub_value is not None or sub not in merged[key]:
                    merged[key][sub] = sub_value
        elif value is not None or key not in merged:
            merged[key] = value
    merged["created_utc"] = existing.get("created_utc") or fresh.get("created_utc")
    return merged


def format_missing(manifest: dict) -> str | None:
    """The line printed at the end of every run: what is missing, if anything."""
    missing, unknown = manifest.get("missing_fields", []), manifest.get("unknown_fields", [])
    if not missing and not unknown:
        return None
    parts = []
    if missing:
        parts.append(f"INCOMPLETE manifest — missing: {', '.join(missing)}")
    if unknown:
        parts.append(f"UNKNOWN (not recoverable): {', '.join(unknown)}")
    return "; ".join(parts)


# -- the registry ---------------------------------------------------------------


def index_path(runs_dir: Path | str = artifacts.DEFAULT_RUNS_DIR) -> Path:
    return Path(runs_dir) / INDEX_FILE


def _index_rows(path: Path) -> tuple[str, list[list[str]]]:
    if not path.exists():
        return INDEX_HEADER, []
    text = path.read_text(encoding="utf-8")
    head, rows = [], []
    for line in text.splitlines():
        cells = None
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and len(cells) == len(INDEX_COLUMNS) and cells[0] not in ("run_id", "---"):
            rows.append(cells)
        elif not rows:
            head.append(line)
    return "\n".join(head) + "\n", rows


def _write_index(path: Path, head: str, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows)
    path.write_text(head.rstrip("\n") + "\n" + body, encoding="utf-8")


def index_row(manifest: dict, score: str | None) -> dict:
    """The registry row for a manifest; ``score`` is ``"422/500"`` or ``"pending"``."""
    return {
        "run_id": manifest["run_id"],
        "date": (manifest.get("created_utc") or utc_now())[:10],
        "arm": manifest["arm"].get("system") or UNKNOWN,
        "reader": manifest["reader"].get("requested_model") or UNKNOWN,
        "n": str(manifest["dataset"].get("n") or UNKNOWN),
        "score": score or "pending",
        "clean": {True: "Y", False: "N"}.get(manifest["code"].get("clean_tree"), UNKNOWN),
        "status": manifest.get("status") or "incomplete",
        "claim": manifest.get("claim") or "none",
    }


def index_append(row: dict, runs_dir: Path | str = artifacts.DEFAULT_RUNS_DIR) -> Path:
    """Append one row. A ``run_id`` already present is updated through :func:`index_update`."""
    path = index_path(runs_dir)
    head, rows = _index_rows(path)
    if any(r[0] == f"`{row['run_id']}`" or r[0] == row["run_id"] for r in rows):
        return index_update(
            row["run_id"], status=row["status"], score=row["score"], runs_dir=runs_dir
        )
    rows.append([str(row[c]) if c != "run_id" else f"`{row['run_id']}`" for c in INDEX_COLUMNS])
    _write_index(path, head, rows)
    return path


def index_update(
    run_id: str,
    *,
    status: str | None = None,
    score: str | None = None,
    runs_dir: Path | str = artifacts.DEFAULT_RUNS_DIR,
) -> Path:
    """Change a row's ``status``; fill a ``pending`` score once. Nothing else moves."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    path = index_path(runs_dir)
    head, rows = _index_rows(path)
    status_col, score_col = INDEX_COLUMNS.index("status"), INDEX_COLUMNS.index("score")
    for r in rows:
        if r[0] in (f"`{run_id}`", run_id):
            if status is not None:
                r[status_col] = status
            if score is not None and score != "pending" and r[score_col] == "pending":
                r[score_col] = score
            break
    else:
        raise KeyError(f"{run_id} is not in {path}")
    _write_index(path, head, rows)
    return path


def index_rows(runs_dir: Path | str = artifacts.DEFAULT_RUNS_DIR) -> list[dict]:
    _head, rows = _index_rows(index_path(runs_dir))
    return [dict(zip(INDEX_COLUMNS, [r[0].strip("`"), *r[1:]], strict=True)) for r in rows]


# -- the promotion gate -----------------------------------------------------------


def promotable(manifest: dict | None) -> list[str]:
    """Why a run may NOT be promoted to ``results/published/``; empty = it may."""
    if manifest is None:
        return ["no manifest.json in the run directory"]
    reasons = []
    if manifest["code"].get("clean_tree") is not True:
        reasons.append("the harness tree was not clean at run time")
    missing, _unknown = check(manifest)
    if missing:
        reasons.append("manifest incomplete: missing " + ", ".join(missing))
    if manifest.get("provisional_reasons"):
        reasons.append("provisional: " + "; ".join(manifest["provisional_reasons"]))
    if manifest.get("status") == "aborted":
        reasons.append("the run was aborted")
    return reasons
