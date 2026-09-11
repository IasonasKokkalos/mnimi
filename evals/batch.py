"""Batch API mode for the OpenAI reader transport.

The programme's reader budget is $50 and the Batch API halves the price, so
every sitting on the gpt-4o family goes through here. The contract, from the
official guide and the installed SDK: one JSONL of ``{custom_id, method, url,
body}`` lines uploaded with ``purpose="batch"``; ``batches.create`` with the
chat-completions endpoint and the only allowed window, ``24h``; a status walk
``validating → in_progress → finalizing → completed`` with the terminal
failures ``failed`` (validation), ``expired`` (window ran out — partial output
is still delivered) and ``cancelled``; output lines in **no particular order**,
keyed by ``custom_id``; errors in a separate file.

Batch mode is not a pin. The request bodies are the ones the synchronous path
sends (``OpenAIReader.request_body``), on the same snapshot with the same
decode pins, so a batch run and a sync run share a ``pins_hash`` and a pair of
them is a valid drift measurement. What the batch resolved to — its id,
status, counts, the served fingerprints — is recorded in
``reader_resolved.json``, never hashed.

Only the injected client touches the network; everything here is testable
with a fake.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"  # the only value the API accepts today
TERMINAL = frozenset({"completed", "expired", "failed", "cancelled"})
#: Terminal statuses that carry an output file. ``expired`` delivers whatever
#: finished inside the window; the rest of the items are filled synchronously.
WITH_OUTPUT = frozenset({"completed", "expired"})


def request_line(item) -> dict:
    """One input line, exactly as the guide specifies."""
    return {
        "custom_id": item.custom_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": item.body,
    }


def write_requests_jsonl(path: Path, items) -> Path:
    """The file that gets uploaded — kept beside the run as its own record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(request_line(item), sort_keys=True) + "\n")
    return path


def read_requests_jsonl(path: Path) -> dict[str, dict]:
    """``custom_id -> body`` from a requests file, for a resume."""
    bodies: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                bodies[row["custom_id"]] = row["body"]
    return bodies


def submit(client, requests_path: Path, *, metadata: dict | None = None):
    """Upload the requests file and create the batch. Returns the Batch."""
    data = requests_path.read_bytes()
    uploaded = client.files.create(file=(requests_path.name, data), purpose="batch")
    return client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata=metadata or {},
    )


def request_counts(batch_obj) -> dict:
    """``{completed, failed, total}`` from an SDK object or a dict."""
    counts = getattr(batch_obj, "request_counts", None)
    if counts is None and isinstance(batch_obj, dict):
        counts = batch_obj.get("request_counts")
    out = {}
    for key in ("completed", "failed", "total"):
        if isinstance(counts, dict):
            out[key] = counts.get(key)
        else:
            out[key] = getattr(counts, key, None)
    return out


def wait(
    client,
    batch_id: str,
    *,
    poll_seconds: float,
    progress: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Poll until a terminal status. ``progress(batch)`` fires on every change.

    ``poll_seconds=0`` never sleeps (tests). Interrupting the wait is safe: the
    batch id is already on disk and the same command resumes.
    """
    last_status = None
    while True:
        batch_obj = client.batches.retrieve(batch_id)
        if batch_obj.status != last_status:
            last_status = batch_obj.status
            if progress is not None:
                progress(batch_obj)
        if batch_obj.status in TERMINAL:
            return batch_obj
        if poll_seconds > 0:
            sleep(poll_seconds)


def _read_jsonl_file(client, file_id: str) -> list[dict]:
    content = client.files.content(file_id).content
    text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def fetch_outputs(client, batch_obj) -> tuple[dict[str, dict], dict[str, dict]]:
    """``(outputs, errors)`` keyed by ``custom_id``.

    Only a 200 response counts as output; anything else — an error line, a
    non-200 status, an expired request — lands in ``errors`` with whatever
    the API said, so the caller can fill it synchronously and say so.
    """
    outputs: dict[str, dict] = {}
    errors: dict[str, dict] = {}
    output_file_id = getattr(batch_obj, "output_file_id", None)
    error_file_id = getattr(batch_obj, "error_file_id", None)
    if output_file_id:
        for row in _read_jsonl_file(client, output_file_id):
            response = row.get("response") or {}
            if response.get("status_code") == 200 and response.get("body") is not None:
                outputs[row["custom_id"]] = response["body"]
            else:
                errors[row["custom_id"]] = row.get("error") or response
    if error_file_id:
        for row in _read_jsonl_file(client, error_file_id):
            errors.setdefault(row["custom_id"], row.get("error") or {})
    return outputs, errors


def complete_outputs(client, items, outputs: dict) -> tuple[dict, list[str]]:
    """Fill every item without a 200 body with ONE synchronous call.

    The body sent is the item's own batch body — same snapshot, same decode
    pins — so the row is the same configuration; only its transport differed.
    The ids are returned so ``reader_resolved.json`` can list them.
    """
    filled = dict(outputs)
    fallbacks: list[str] = []
    for item in items:
        if item.custom_id in filled:
            continue
        completion = client.chat.completions.create(**item.body)
        filled[item.custom_id] = completion
        fallbacks.append(item.custom_id)
    return filled, fallbacks


def summary(batch_obj) -> dict:
    """The batch as recorded in ``reader_resolved.json``."""
    errors = getattr(batch_obj, "errors", None)
    error_rows = []
    data = getattr(errors, "data", None) if errors is not None else None
    if data is None and isinstance(errors, dict):
        data = errors.get("data")
    for err in data or []:
        if isinstance(err, dict):
            error_rows.append({k: err.get(k) for k in ("code", "message", "line")})
        else:
            error_rows.append({k: getattr(err, k, None) for k in ("code", "message", "line")})
    return {
        "batch_id": getattr(batch_obj, "id", None),
        "batch_status": getattr(batch_obj, "status", None),
        "request_counts": request_counts(batch_obj),
        "output_file_id": getattr(batch_obj, "output_file_id", None),
        "error_file_id": getattr(batch_obj, "error_file_id", None),
        "batch_errors": error_rows,
        "completion_window": COMPLETION_WINDOW,
    }
