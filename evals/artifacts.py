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
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

DEFAULT_RUNS_DIR = "runs"

# Schema version for the artifact layout itself, so a future reader can tell a
# v0.2 artifact from whatever replaces it.
ARTIFACT_SCHEMA = "mnimi-eval-artifact/4"


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


def default_serve_log() -> str | None:
    """Best guess at the Ollama server log, if one is being written."""
    explicit = os.environ.get("OLLAMA_SERVE_LOG")
    if explicit:
        return explicit
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Ollama" / "server.log"
        if candidate.exists():
            return str(candidate)
    home = Path.home() / ".ollama" / "logs" / "server.log"
    return str(home) if home.exists() else None


def _parse_daemon_env(log_text: str) -> dict:
    """OLLAMA_* settings the *daemon* actually resolved.

    Ollama logs its effective configuration once at startup as
    ``msg="server config" env="map[K:V K:V ...]"``. That is the daemon's view,
    which is the one that matters — the client process can hold entirely
    different values and frequently does (a variable set after the daemon
    started, or a process-scoped override, is invisible to the running server).
    """
    matches = re.findall(r'env="map\[(.*?)\]"', log_text, re.S)
    if not matches:
        return {}
    out: dict = {}
    # Space-separated K:V pairs; values may be empty, and may contain ':'.
    for token in re.findall(r"(\w+):((?:[^\s]|\s(?!\w+:))*)", matches[-1]):
        key, value = token[0], token[1].strip()
        if key.startswith("OLLAMA_"):
            # The log escapes backslashes, so a Windows path arrives doubled.
            # Unescaped here so the recorded value is the real path and a
            # comparison against the client's value is not pure noise.
            out[key] = value.replace("\\\\", "\\")
    return out


def _norm_env_value(value: str) -> str:
    """Comparison form for an env value: escaping, separators and case folded.

    Without this, every Windows path looks like a client/daemon mismatch and
    the mismatch field is worthless precisely where it matters.
    """
    return value.replace("\\\\", "\\").replace("/", "\\").rstrip("\\").casefold()


# Ollama writes this once per runner spawn, immediately before the runner prints
# the settings it resolved. It is the only line that delimits one runner's output
# from the next one's — which is exactly what "the daemon serving this run" means.
_RUNNER_START = re.compile(r"starting llama server")


def recent_model_load_block(log_text: str) -> str:
    """Everything the CURRENT runner wrote, anchored at its start marker.

    The settings preflight validates — resolved `flash_attn`, prompt-cache state
    — are stated once, when a runner spawns, and are never restated while the
    model stays warm. Request logging meanwhile appends for the whole length of
    a run, so the load block's distance from the end of the file grows without
    bound. Selecting it by byte offset therefore fails on exactly the long runs
    that matter: measured 2026-07-30, a ~1.2 MB log with the resolved line at
    byte 16,862 fell outside a 400,000-byte tail and preflight refused three
    arms whose daemon was verifiably in the pinned configuration. Anchoring on
    the marker makes the result independent of both log size and model warmth.

    Deliberately NOT the whole file. `preflight_reader_env` takes the FIRST
    resolution it finds, so handing it every block would grade the oldest runner
    in the log — waving a restarted, mis-launched daemon through on the strength
    of a correct block written hours earlier. That converts a false refusal into
    a false pass, and a false pass is the one outcome a preflight must never
    return.

    Returns "" when no marker is present: without one there is nothing to
    attribute the settings below it to, and unattributable evidence is not
    evidence. The caller fails closed on the empty string.
    """
    starts = [m.start() for m in _RUNNER_START.finditer(log_text)]
    return log_text[starts[-1] :] if starts else ""


def _tail_model_load(log_text: str) -> str:
    """The most recent model-load block, verbatim.

    Bounded to the last load so the artifact records what produced *this* run
    rather than the whole file's history.
    """
    starts = [m.start() for m in re.finditer(r"starting llama server|load_tensors:", log_text)]
    if not starts:
        return ""
    block = log_text[starts[-1] if len(starts) < 2 else starts[-2] :]
    return block[:200_000]


def capture_environment(
    ollama_host: str = "http://localhost:11434",
    serve_log: str | None = None,
    save_log_to: Path | None = None,
) -> dict:
    """Diagnostic snapshot of the machine that produced a run.

    **Deliberately NOT part of pins.** These fields describe the environment,
    not the configuration: folding them into ``pins_hash`` would make every run
    on a different machine — or after a driver update — look like a different
    configuration, which destroys the ability to compare runs at all. They are
    recorded so a surprising number can be investigated, not so it can be
    invalidated. ``tests/test_artifacts.py`` asserts they stay out of the hash.

    The load-time toggles below exist because of a real drift: an identical
    request under an identical ``pins_hash`` and identical model digest produced
    different output after the model store moved and the daemon restarted. That
    drift is now RESOLVED — flash attention and the llama.cpp prompt cache, both
    since promoted to pins (see ``runner.READER_FLASH_ATTENTION`` /
    ``READER_CACHE_RAM``) — and these fields are what made the diagnosis
    possible, so they stay. They remain diagnostics rather than pins: they
    describe the machine, and the two settings that describe the *configuration*
    now live in ``build_pins`` where a mismatch changes ``pins_hash``.

    ``prompt_cache_reported`` follows the same requested-vs-resolved discipline
    that caught flash attention: the daemon prints its own cache limit, and that
    printed limit — not the env var that asked for it — is what gets recorded.
    """
    env: dict = {
        "gpu_model": None,
        "driver_version": None,
        "cuda_version": None,
        "ollama_version": None,
        "offloaded_layers": None,
        "offloaded_layers_source": None,
        # Client vs daemon are recorded separately and never merged: when they
        # disagree, the disagreement is itself the finding.
        "ollama_env_client": {},
        "ollama_env_daemon": {},
        "ollama_env_mismatch": None,
        # Resolved, not requested: "flash_attn = auto" is a request, "Flash
        # Attention enabled" is what actually happened.
        "flash_attention_reported": None,
        # Resolved prompt-cache limit as the daemon printed it. "0.000 MiB" is
        # the disabled state this harness requires; anything else means a
        # prediction can depend on which request preceded it.
        "prompt_cache_reported": None,
        "kv_cache_type": None,
        "model_blob_path": None,
        "runner_cmd": None,
        "serve_log_path": serve_log,
        "model_load_log": None,
    }

    env["ollama_env_client"] = {
        k: v for k, v in sorted(os.environ.items()) if k.startswith("OLLAMA_")
    }

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, _, driver = proc.stdout.strip().splitlines()[0].partition(",")
            env["gpu_model"] = name.strip()
            env["driver_version"] = driver.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False
        )
        m = re.search(r"CUDA Version:\s*([0-9.]+)", proc.stdout or "")
        if m:
            env["cuda_version"] = m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        with urllib.request.urlopen(f"{ollama_host}/api/version", timeout=5) as resp:
            env["ollama_version"] = json.load(resp).get("version")
    except (urllib.error.URLError, OSError, ValueError):
        pass

    # Layer count is READ, never assumed. Preference order: the server log (it
    # states the split outright), then the runtime signal from /api/ps. A
    # manually launched `ollama serve` discards stdout, so the log is often
    # absent — in which case say which signal was used rather than inventing a
    # number.
    if serve_log and Path(serve_log).exists():
        try:
            text = Path(serve_log).read_text(encoding="utf-8", errors="ignore")

            hits = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
            if hits:
                env["offloaded_layers"] = f"{hits[-1][0]}/{hits[-1][1]}"
                env["offloaded_layers_source"] = "server_log"

            env["ollama_env_daemon"] = _parse_daemon_env(text)

            # Resolved flash-attention state. Prefer the outcome line over the
            # request line; record whichever is present so "auto" is never
            # mistaken for a resolution.
            # `flash_attn = enabled|disabled` IS a resolution; only `auto` is an
            # unresolved request, and `auto` means the daemon was launched
            # without OLLAMA_FLASH_ATTENTION — i.e. very likely the tray app.
            # Precedence: an explicit `flash_attn = enabled|disabled` is itself a
            # resolution; `auto` is not, and is resolved only by the later fused-
            # ops line. `auto` surviving with no resolution means the daemon
            # started without OLLAMA_FLASH_ATTENTION at all.
            fa = re.findall(r"flash_attn\s*=\s*(\S+)", text)
            fused = re.findall(r"Flash Attention (enabled|disabled)", text)
            if fa and fa[-1] in {"enabled", "disabled"}:
                env["flash_attention_reported"] = f"Flash Attention {fa[-1]}"
            elif fused:
                env["flash_attention_reported"] = f"Flash Attention {fused[-1]}"
            elif fa:
                env["flash_attention_reported"] = (
                    f"requested flash_attn={fa[-1]} (unresolved — daemon started "
                    "without OLLAMA_FLASH_ATTENTION)"
                )

            # Resolved prompt-cache state. The two states report themselves in
            # DIFFERENT sentences, and the disabled one emits no `cache state`
            # line at all — so matching only the limit form records None exactly
            # when the cache is off, i.e. never confirms the state we want.
            #   disabled: "prompt cache is disabled - use `--cache-ram N` ..."
            #   active:   "cache state: 0 prompts, ... (limits: 8192.000 MiB, ...)"
            if re.search(r"prompt cache is disabled", text):
                env["prompt_cache_reported"] = "prompt cache disabled"
            else:
                cache = re.findall(r"cache state:[^(]*\(limits:\s*([0-9.]+)\s*MiB", text)
                if cache:
                    env["prompt_cache_reported"] = (
                        f"prompt cache ACTIVE, limit {float(cache[-1]):.3f} MiB "
                        "— output may depend on the preceding request"
                    )

            kv = re.findall(
                r"(type_k\s*=\s*\S+.*?type_v\s*=\s*\S+|KV cache type[^\n]*|"
                r"kv_cache_type[^\n]*|K \([^)]*\), V \([^)]*\))",
                text,
            )
            if kv:
                env["kv_cache_type"] = f"{kv[-1].strip()} (from load log)"

            # Matched separately, not as an alternation: "starting llama server"
            # appears earlier on the same line and would swallow the match
            # before the cmd= capture group is ever reached.
            cmds = re.findall(r'cmd="([^"]*)"', text)
            if cmds:
                env["runner_cmd"] = cmds[-1].strip()[:4000]
            else:
                fallback = re.findall(r"starting llama server[^\n]*", text)
                if fallback:
                    env["runner_cmd"] = fallback[-1].strip()[:4000]

            blobs = re.findall(r"(--model\s+(\S+)|from\s+(\S*blobs[\\/]sha256-\S+))", text)
            if blobs:
                last_blob = blobs[-1]
                env["model_blob_path"] = (last_blob[1] or last_blob[2] or "").strip()

            # Verbatim load log: saved beside the run when a destination exists
            # (keeps results.json readable), inlined otherwise so it is never
            # simply lost.
            block = _tail_model_load(text)
            if block:
                if save_log_to is not None:
                    try:
                        save_log_to.parent.mkdir(parents=True, exist_ok=True)
                        save_log_to.write_text(block, encoding="utf-8")
                        env["model_load_log"] = f"see {save_log_to.name}"
                    except OSError:
                        env["model_load_log"] = block
                else:
                    env["model_load_log"] = block
        except OSError:
            pass

    # Not every Ollama build logs the KV cache type at load. When it does not,
    # fall back to the daemon's configured value and say so, rather than
    # recording None and losing the setting entirely.
    if env["kv_cache_type"] is None and env["ollama_env_daemon"]:
        configured = env["ollama_env_daemon"].get("OLLAMA_KV_CACHE_TYPE", "")
        env["kv_cache_type"] = (
            f"OLLAMA_KV_CACHE_TYPE={configured or 'unset (build default)'} "
            "(from daemon config; not stated in load log)"
        )

    if env["ollama_env_daemon"]:
        shared = set(env["ollama_env_client"]) & set(env["ollama_env_daemon"])
        differing = sorted(
            k for k in shared
            if _norm_env_value(str(env["ollama_env_client"][k]))
            != _norm_env_value(str(env["ollama_env_daemon"][k]))
        )
        env["ollama_env_mismatch"] = differing or "none"
    else:
        env["ollama_env_mismatch"] = "daemon env unavailable (no server config in log)"
    if env["offloaded_layers"] is None:
        try:
            with urllib.request.urlopen(f"{ollama_host}/api/ps", timeout=5) as resp:
                for m in json.load(resp).get("models", []):
                    size, vram = m.get("size", 0), m.get("size_vram", 0)
                    if size:
                        env["offloaded_layers"] = (
                            "all (size_vram == size)" if vram == size
                            else ("none (size_vram == 0)" if vram == 0 else "partial")
                        )
                        env["offloaded_layers_source"] = "api_ps_vram"
                    break
        except (urllib.error.URLError, OSError, ValueError):
            pass
    return env


def build_pins(
    *,
    dataset_file: str,
    dataset_sha256: str,
    system: str,
    limit: int | None,
    sample_strategy: str,
    sample_seed: int,
    reader_model: str,
    reader_digest: str | None,
    reader_num_ctx: int,
    reader_seed: int,
    reader_top_k: int,
    reader_num_gpu: int,
    reader_num_thread: int,
    reader_num_batch: int,
    reader_flash_attention: int,
    reader_cache_ram: int,
    reader_prompt_version: str,
    reader_prompt_hash: str,
    reader_answer_reserve: int,
    reader_scaffold_tokens: int,
    reader_chars_per_token: int,
    render_template_hash: str,
    embedder_name: str | None = None,
    embedder_dim: int | None = None,
    embedder_revision: str | None = None,
    embed_template_hash: str | None = None,
    extractor_model: str | None = None,
    k: int | None = None,
    dedup_cosine_threshold: float | None = None,
) -> dict:
    """Everything that determines the *predictions*, and nothing that does not.

    Wall-clock, elapsed time and counts are deliberately excluded: they belong
    to run metadata. Two runs are comparable exactly when their pins match, so
    anything that varies between identical re-runs must stay out of here.

    Retrieval and extraction fields are ``None`` for systems that do not
    retrieve; a retrieving system fills them via ``MemorySystem.retrieval_pins``.

    Schema /2 (2026-07-29) added ``dedup_cosine_threshold`` and the three
    reader-trim fields, and made the retrieval fields actually get filled. The
    hole they close was found by measurement, not review: two mnimi runs
    differing by 19/20 predictions and 20 accuracy points carried identical
    ``pins_hash`` under /1, because the only thing separating them — the dedup
    threshold — had no slot, and ``embedder_name``/``embedder_dim``/``k`` were
    documented as "a system that retrieves fills them" while nothing did.

    The reader-trim trio is here for the same reason. ``num_ctx`` alone does not
    determine how much context reaches the reader: the budget is
    ``num_ctx - answer_reserve - scaffold_tokens``, compared against a
    ``chars_per_token`` estimate. All three move the fed-token count, and
    full_history truncates on every question, so all three move its score.

    Schema /3 (2026-07-30) moved the judge OUT of pins and added
    ``embedder_revision``. Judge identity (model, prompt version, prompt hash,
    decode config) lives in the ``judge`` block of ``results.json``, refreshed
    at judge time from the judge that actually grades — a judge replay must
    never rewrite predict-stage provenance, and under /2 the judge fields here
    described the judge *requested* at predict time, which a later re-grade
    silently falsified. ``embedder_revision`` pins the HF commit of the
    embedding model; a bare model name is mutable and can move every vector
    without moving any header field.

    Schema /4 (2026-07-30) added the embed/render split hashes. The embedded
    text and the rendered text are distinct artifacts with distinct hashes:
    an edit to the render template changes reader context and no vectors; an
    edit to the embed template changes every vector, every retrieval and the
    dedup key, and invalidates any threshold selected under the previous form.
    ``render_template_hash`` is harness-wide (every arm renders through one
    code path); ``embed_template_hash`` is declared by the arms that embed.
    """
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "harness_git_sha": harness_git_sha(),
        "dataset_file": dataset_file,
        "dataset_sha256": dataset_sha256,
        "system": system,
        "limit": limit,
        # Which questions were selected is part of the number: a file-order
        # slice and a stratified slice of the same size are different benchmarks.
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "reader_model": reader_model,
        "reader_digest": reader_digest,
        "reader_num_ctx": reader_num_ctx,
        "reader_seed": reader_seed,
        "reader_top_k": reader_top_k,
        "reader_num_gpu": reader_num_gpu,
        "reader_num_thread": reader_num_thread,
        "reader_num_batch": reader_num_batch,
        # Daemon-level, not per-request: both are resolved when the daemon
        # starts. They are pins because they change the output — FA changed 2/2
        # probe predictions with cache state held constant, and a live prompt
        # cache makes a prediction depend on what ran before it. `preflight_
        # reader_env` asserts the serving daemon actually resolved to these.
        "reader_flash_attention": reader_flash_attention,
        "reader_cache_ram": reader_cache_ram,
        "reader_prompt_version": reader_prompt_version,
        "reader_prompt_hash": reader_prompt_hash,
        # The trim gate: budget = num_ctx - answer_reserve - scaffold_tokens,
        # measured in chars_per_token estimates. Determines what the reader sees.
        "reader_answer_reserve": reader_answer_reserve,
        "reader_scaffold_tokens": reader_scaffold_tokens,
        "reader_chars_per_token": reader_chars_per_token,
        "render_template_hash": render_template_hash,
        "embedder_name": embedder_name,
        "embedder_dim": embedder_dim,
        "embedder_revision": embedder_revision,
        "embed_template_hash": embed_template_hash,
        "extractor_model": extractor_model,
        "k": k,
        "dedup_cosine_threshold": dedup_cosine_threshold,
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
    return read_predictions_file(path, cls)


def read_predictions_file(path: Path, cls):
    """Rehydrate predictions from an explicit path — the Tier 1 audit entry point.

    A published ``predictions.jsonl`` is self-sufficient: every row carries the
    question and the gold answer, so grading it needs no dataset, no pins, no
    reader and no memory system. Auditing therefore means pointing at one file,
    not reconstructing the run that produced it.
    """
    if not path.exists():
        raise FileNotFoundError(f"no predictions file at {path}")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(cls(**json.loads(line)))
    return out


def read_pins_optional(directory: Path) -> dict:
    """Best-effort pins for a directory that may not carry all three artifacts.

    Prefers ``pins.json``, falls back to the copy embedded in ``results.json``,
    and returns ``{}`` when neither is present. Tier 1 must still grade a bare
    ``predictions.jsonl`` — a missing header costs the auditor provenance, not
    the ability to recompute the score.
    """
    try:
        return read_pins(directory)
    except (FileNotFoundError, KeyError, ValueError):
        pass
    results = directory / "results.json"
    if results.exists():
        try:
            return json.loads(results.read_text(encoding="utf-8")).get("pins", {})
        except (OSError, ValueError):
            return {}
    return {}


def read_published_score(directory: Path) -> tuple[int, int] | None:
    """``(correct, total)`` from a published ``results.json``, or ``None``.

    This is what a Tier 1 audit checks its recomputed score against: the claim
    is "given these predictions, the judge produces the published score", and
    the published score has to come from the published artifact to test it.
    """
    path = directory / "results.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError):
        return None
    return sum(1 for r in rows if r.get("correct")), len(rows)


def write_results(
    directory: Path,
    pins: dict,
    results,
    run_meta: dict,
    provisional: list[str],
    summary: dict | None = None,
    judge: dict | None = None,
) -> Path:
    """Final graded artifact: header, judge block, run stats, per-question rows.

    ``summary`` carries accuracy with Wilson intervals. It is computed by the
    caller (``report.summary``) rather than here because this module sits below
    ``runner`` in the import graph and cannot reach the aggregation code.

    ``judge`` is the identity of the judge that produced these verdicts
    (``evals.judge.judge_block``), passed in for the same layering reason. It
    is refreshed on every judge pass and hashed separately from ``pins_hash``:
    pins describe the predict stage, this block describes the grading, and a
    re-grade must be able to update one without touching the other.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    payload = {
        "pins": pins,
        "pins_hash": pins_hash(pins),
        # The judge that graded THESE verdicts — not the one pins requested.
        "judge": judge or {},
        "judge_hash": fingerprint(canonical(judge)) if judge else None,
        # Empty list means publishable; anything in it names what is not yet locked.
        "provisional": provisional,
        "run": run_meta,
        # Every rate here carries its interval: no bare percentage in an artifact.
        "summary": summary or {},
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
