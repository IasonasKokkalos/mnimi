"""Staged run artifacts and the reproducibility header.

A score is only a result if you can say what produced it. Every run writes:

    runs/<name>/pins.json          the reproducibility header
    runs/<name>/predictions.jsonl  one reader answer per question
    runs/<name>/results.json       header + run stats + graded results

The stage split is the point of the layout: ``predictions.jsonl`` carries the
gold answer and question text alongside each prediction, so the judge stage is
self-sufficient — re-grading never re-ingests, never re-reads, and never needs
the dataset or a reader.

**Digest algorithm lives in exactly one place** (:func:`fingerprint`). Nothing
else in the harness calls ``hashlib`` directly, so changing the algorithm is a
one-line edit rather than an archaeology exercise. It is stable across
processes by construction — Python's builtin ``hash()`` is salted per process
and cannot key anything persisted to disk.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

DEFAULT_RUNS_DIR = "runs"

# Schema version for the artifact layout itself, so a future reader can tell a
# v0.2 artifact from whatever replaces it.
ARTIFACT_SCHEMA = "mnimi-eval-artifact/1"


def fingerprint(text: str) -> str:
    """Stable digest of a string. The harness's single hashing boundary."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _git(*args: str) -> str | None:
    """Run a git command, or return ``None`` if git/the repo is unavailable."""
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def harness_git_sha() -> str | None:
    """Commit of the harness, suffixed ``-dirty`` when the tree has edits.

    An uncommitted tree means the code that produced the number cannot be
    recovered from the sha alone; the suffix says so out loud instead of
    implying a clean provenance the artifact does not have.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None
    return f"{sha}-dirty" if _git("status", "--porcelain") else sha


def build_pins(
    *,
    dataset_file: str,
    dataset_sha256: str,
    system: str,
    limit: int | None,
    reader_model: str,
    reader_digest: str | None,
    reader_num_ctx: int,
    reader_seed: int,
    reader_top_k: int,
    reader_num_gpu: int,
    reader_num_thread: int,
    reader_num_batch: int,
    reader_prompt_version: str,
    reader_prompt_hash: str,
    judge_model: str,
    judge_prompt_version: str,
    judge_prompt_hash: str,
    embedder_name: str | None = None,
    embedder_dim: int | None = None,
    extractor_model: str | None = None,
    k: int | None = None,
) -> dict:
    """Everything that determines the number, and nothing that does not.

    Wall-clock, elapsed time and counts are deliberately excluded: they belong
    to run metadata. Two runs are comparable exactly when their pins match, so
    anything that varies between identical re-runs must stay out of here.

    Retrieval and extraction fields are present but ``None`` for the baselines —
    the schema is stable from v0.2 on, and a system that retrieves fills them.
    """
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "harness_git_sha": harness_git_sha(),
        "dataset_file": dataset_file,
        "dataset_sha256": dataset_sha256,
        "system": system,
        "limit": limit,
        "reader_model": reader_model,
        "reader_digest": reader_digest,
        "reader_num_ctx": reader_num_ctx,
        "reader_seed": reader_seed,
        "reader_top_k": reader_top_k,
        "reader_num_gpu": reader_num_gpu,
        "reader_num_thread": reader_num_thread,
        "reader_num_batch": reader_num_batch,
        "reader_prompt_version": reader_prompt_version,
        "reader_prompt_hash": reader_prompt_hash,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "judge_prompt_hash": judge_prompt_hash,
        "embedder_name": embedder_name,
        "embedder_dim": embedder_dim,
        "extractor_model": extractor_model,
        "k": k,
    }


def pins_hash(pins: dict) -> str:
    """One string that answers "is this the same configuration?"."""
    return fingerprint(canonical(pins))


def run_dir(system: str, limit: int | None, base: str = DEFAULT_RUNS_DIR) -> Path:
    """Stable directory per (system, slice) so predict → judge chains by default."""
    return Path(base) / f"{system}__{limit}q"


def write_pins(directory: Path, pins: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pins.json"
    payload = {"pins": pins, "pins_hash": pins_hash(pins)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_pins(directory: Path) -> dict:
    path = directory / "pins.json"
    if not path.exists():
        raise FileNotFoundError(f"no pins.json in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))["pins"]


def write_predictions(directory: Path, predictions) -> Path:
    """One JSON object per line, so a long run is inspectable while it streams."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(canonical(asdict(pred)) + "\n")
    return path


def read_predictions(directory: Path, cls):
    """Rehydrate predictions written by :func:`write_predictions`."""
    path = directory / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions.jsonl in {directory} — run the predict stage first"
        )
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(cls(**json.loads(line)))
    return out


def write_results(
    directory: Path, pins: dict, results, run_meta: dict, provisional: list[str]
) -> Path:
    """Final graded artifact: header, provisionality, run stats, per-question rows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    payload = {
        "pins": pins,
        "pins_hash": pins_hash(pins),
        # Empty list means publishable; anything in it names what is not yet locked.
        "provisional": provisional,
        "run": run_meta,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
