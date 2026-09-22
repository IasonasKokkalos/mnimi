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

**Sub-batches under the enqueued-token cap (2026-09-12).** The Batch API
enforces a per-model *enqueued tokens* limit on the organization — 90,000 for
`gpt-4o` at this key's usage tier — and a batch that would exceed it fails
validation outright (`token_limit_exceeded`, nothing charged). An n=100 arm
is ~550k tokens, so a run is planned as sub-batches (chunks) of at most
``ENQUEUED_TOKEN_LIMIT`` estimated tokens each — input at the trim gate's
chars-per-token convention plus ``max_tokens`` per request, i.e. the same
upper bound the cost gate uses — submitted one at a time, each after the
previous reached a terminal status. Chunking is not a pin: the request
bodies are unchanged and every chunk is a batch of the same configuration;
the chunk ids and the cap are recorded in ``reader_resolved.json``.

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
#: The organization's per-model enqueued-token cap for the Batch API at this
#: key's usage tier (measured 2026-09-12: four n=100 batches failed validation
#: with "Limit: 90,000 enqueued tokens"). A fact about the key, like the
#: ledger; ``--batch-enqueued-tokens`` overrides it for one run.
ENQUEUED_TOKEN_LIMIT = 90_000
#: A chunk whose batch failed validation without running anything (the cap,
#: usually because another batch was still in progress) is resubmitted this
#: many times before the run gives up.
CHUNK_RETRIES = 3
BATCH_STATE_SCHEMA = 2


def chunk_file(index: int) -> str:
    """The per-chunk requests file uploaded for chunk ``index`` (0-based)."""
    return f"batch_chunk_{index:02d}.jsonl"


def enqueued_tokens(body: dict, chars_per_token: int) -> int:
    """Upper-bound tokens one request enqueues: input at chars/N plus max_tokens."""
    chars = sum(len(m.get("content") or "") for m in body.get("messages", []))
    return -(-chars // chars_per_token) + int(body.get("max_tokens") or 0)


def plan_chunks(items, limit: int, chars_per_token: int) -> list[list]:
    """Greedy, order-preserving split of ``items`` into chunks under ``limit``.

    An item that alone exceeds the limit still gets its own chunk — the API
    will refuse it and the run reports that, rather than silently dropping it.
    """
    chunks: list[list] = []
    current: list = []
    load = 0
    for item in items:
        cost = enqueued_tokens(item.body, chars_per_token)
        if current and load + cost > limit:
            chunks.append(current)
            current, load = [], 0
        current.append(item)
        load += cost
    if current:
        chunks.append(current)
    return chunks


def chunk_states(chunks: list[list]) -> list[dict]:
    """The ``chunks`` entries of ``batch_state.json`` for freshly planned chunks."""
    return [
        {"index": i, "custom_ids": [item.custom_id for item in chunk],
         "batch_id": None, "status": None, "input_file_id": None, "attempts": 0}
        for i, chunk in enumerate(chunks)
    ]


def replan_outstanding(
    state: dict, by_id: dict, limit: int, chars_per_token: int
) -> tuple[int, int]:
    """Re-plan every chunk that has not run under a new, lower cap; ``(before, after)`` counts.

    The oracle arm of the n=500 sitting (2026-09-22): a chunk the planner put at
    86k of the 90k cap was refused three times with ``token_limit_exceeded`` --
    the chars/N estimate ran under the real count on long contexts -- and a
    resume could not recover, because it reloaded the saved plan and the saved
    cap and refused the chunk at ``CHUNK_RETRIES`` before submitting. A chunk
    counts as outstanding when it never ran: no batch id, or a terminal
    ``failed`` that enqueued nothing. Completed and in-flight chunks are kept
    exactly; the outstanding items are re-split in their original order under
    ``limit`` with ``attempts`` reset, and indices are renumbered after the kept
    chunks. No request body changes -- grouping does not alter a per-request
    Batch API result.
    """
    kept = [c for c in state["chunks"] if c.get("batch_id") and c.get("status") != "failed"]
    outstanding = [c for c in state["chunks"] if c not in kept]
    if not outstanding:
        return 0, 0
    items = [by_id[cid] for c in outstanding for cid in c["custom_ids"]]
    fresh = chunk_states(plan_chunks(items, limit, chars_per_token))
    for i, c in enumerate(fresh, start=len(kept)):
        c["index"] = i
    state["chunks"] = kept + fresh
    state["enqueued_token_limit"] = limit
    return len(outstanding), len(fresh)


def upgrade_state(state: dict) -> dict:
    """A schema-1 state (one batch for the whole run) as a one-chunk schema 2."""
    if state.get("chunks") is not None:
        return state
    ids = [meta["custom_id"] for meta in state.get("items", [])]
    state["chunks"] = [{
        "index": 0, "custom_ids": ids, "batch_id": state.get("batch_id"),
        "status": state.get("status"), "input_file_id": state.get("input_file_id"),
        "attempts": 1,
    }]
    state["batch_state_schema"] = BATCH_STATE_SCHEMA
    return state


def never_ran(batch_obj) -> bool:
    """A terminal batch that enqueued nothing (validation failed): resubmittable."""
    counts = request_counts(batch_obj)
    return batch_obj.status in TERMINAL and not counts.get("total")


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


def merged_summary(batch_objs: list, *, enqueued_token_limit: int) -> dict:
    """The run's batches as recorded in ``reader_resolved.json``.

    One chunk keeps the flat schema-1 keys exactly; several chunks carry the
    same keys aggregated (ids joined, counts summed, the worst status) plus a
    per-chunk ``batches`` list.
    """
    summaries = [summary(b) for b in batch_objs]
    if len(summaries) == 1:
        merged = dict(summaries[0])
    else:
        statuses = [s["batch_status"] for s in summaries]
        counts = {"completed": 0, "failed": 0, "total": 0}
        for s in summaries:
            for key in counts:
                counts[key] += s["request_counts"].get(key) or 0
        merged = {
            "batch_id": ",".join(str(s["batch_id"]) for s in summaries),
            "batch_status": "completed" if all(x == "completed" for x in statuses)
            else next(x for x in statuses if x != "completed"),
            "request_counts": counts,
            "output_file_id": None,
            "error_file_id": None,
            "batch_errors": [e for s in summaries for e in s["batch_errors"]],
            "completion_window": COMPLETION_WINDOW,
        }
    merged["batches"] = summaries
    merged["chunks"] = len(summaries)
    merged["enqueued_token_limit"] = enqueued_token_limit
    return merged


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
