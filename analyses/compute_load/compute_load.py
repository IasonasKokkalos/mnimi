"""Compute-load cells for the README (2026-10-17). One fresh process per cell.

python compute_load.py <cell> --scratch DIR [--profile cpu|gpu]
cells: base | hash | bge | extract

Rounds: the first 50 rounds (session order) of the LongMemEval questions outside the
100-question stratified slice (the dev pool of evals.probes.extractor_bench), each
round one Memory.add() with the session date as ts. Queries: the 20 question texts of
the 20-row stratified slice. Peak working set from K32GetProcessMemoryInfo (no psutil).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as w
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

T_IMPORT_START = time.perf_counter()

_k = ctypes.WinDLL("kernel32", use_last_error=True)


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("PageFaultCount", w.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k.GetCurrentProcess.restype = w.HANDLE
_k.K32GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(_PMC), w.DWORD]


def mem_mib() -> tuple[float, float]:
    """(working set, peak working set) in MiB."""
    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    if not _k.K32GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(p), p.cb):
        raise OSError(ctypes.get_last_error())
    return p.WorkingSetSize / 2**20, p.PeakWorkingSetSize / 2**20


def gpu_used_mib() -> dict:
    """nvidia-smi: total used on the device and this process's own usage (MiB)."""
    try:
        total = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        mine = None
        for line in apps.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0] == str(os.getpid()):
                mine = float(parts[1])
        return {"device_used_mib": float(total) if total else None, "process_used_mib": mine}
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"error": repr(exc)}


def rounds_and_queries(n_rounds: int = 50, n_queries: int = 20):
    """The first n rounds, in session order, of the questions outside the 100-row slice."""
    from evals.dataset import DEFAULT_SAMPLE_SEED, SAMPLE_STRATIFIED, load
    from evals.runner import _session_to_messages

    from mnimi.memory import _messages_to_rounds

    inside = {
        q.question_id for q in load(limit=100, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED)
    }
    rounds = []
    for q in load():
        if q.question_id in inside:
            continue
        for session in q.sessions:
            for round_ in _messages_to_rounds(_session_to_messages(session)):
                msgs = [
                    {"role": t["role"], "content": t["content"], "ts": round_.ts}
                    for t in round_.turns
                ]
                rounds.append(
                    {
                        "id": q.question_id,
                        "turns": msgs,
                        "chars": sum(len(t["content"]) for t in msgs),
                    }
                )
                if len(rounds) == n_rounds:
                    break
            if len(rounds) == n_rounds:
                break
        if len(rounds) == n_rounds:
            break
    qs = [
        q.question
        for q in load(limit=n_queries, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED)
    ]
    return rounds, qs


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "median_ms": round(1000 * statistics.median(xs), 2),
        "mean_ms": round(1000 * statistics.fmean(xs), 2),
        "p90_ms": round(1000 * xs[int(0.9 * (len(xs) - 1))], 2),
        "min_ms": round(1000 * xs[0], 2),
        "max_ms": round(1000 * xs[-1], 2),
    }


def db_bytes(path: Path) -> dict:
    out = {}
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        out[suffix or "db"] = p.stat().st_size if p.exists() else 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=["prep", "base", "hash", "bge", "extract"])
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "cell": args.cell,
        "profile": args.profile if args.cell == "extract" else None,
        "python": sys.version.split()[0],
        "exe": sys.executable,
        "pid": os.getpid(),
    }

    ws0, _ = mem_mib()
    out["ws_bare_mib"] = round(ws0, 1)

    t0 = time.perf_counter()
    import mnimi  # noqa: F401
    from mnimi.memory import Memory

    out["import_mnimi_s"] = round(time.perf_counter() - t0, 3)
    out["ws_after_import_mib"] = round(mem_mib()[0], 1)
    if args.cell == "base":
        out["peak_ws_mib"] = round(mem_mib()[1], 1)
        print(json.dumps(out, indent=1))
        return 0

    fixture = scratch / "rounds50.json"
    if args.cell == "prep":
        rounds, queries = rounds_and_queries(args.rounds)
        fixture.write_text(
            json.dumps({"rounds": rounds, "queries": queries}, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"wrote {fixture} rounds={len(rounds)} queries={len(queries)} "
            f"bytes={fixture.stat().st_size}"
        )
        return 0
    fx = json.loads(fixture.read_text(encoding="utf-8"))
    rounds, queries = fx["rounds"], fx["queries"]
    out["rounds"] = len(rounds)
    out["round_question_ids"] = sorted({r["id"] for r in rounds})
    out["round_chars_mean"] = round(statistics.fmean(r["chars"] for r in rounds), 1)
    out["ws_after_dataset_mib"] = round(mem_mib()[0], 1)
    ws_dataset = mem_mib()[0]

    db = scratch / f"{args.cell}_{args.profile}.db"
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db) + suffix)
        if p.exists():
            p.unlink()

    t0 = time.perf_counter()
    if args.cell == "hash":
        from mnimi.embeddings import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        from mnimi.embeddings import BgeSmallEmbedder

        embedder = BgeSmallEmbedder()
    out["embedder_construct_s"] = round(time.perf_counter() - t0, 3)

    extractor = None
    if args.cell == "extract":
        from mnimi.extract.cache import CachedExtractor
        from mnimi.extract.llama import DECODE, QwenLlamaExtractor

        decode = (
            DECODE if args.profile == "gpu" else {**DECODE, "n_gpu_layers": 0, "max_tokens": 2048}
        )
        cache = scratch / f"extract-cache-{args.profile}.sqlite"
        if cache.exists():
            cache.unlink()
        if args.profile == "gpu":
            out["gpu_before_load"] = gpu_used_mib()
        t0 = time.perf_counter()
        extractor = CachedExtractor(QwenLlamaExtractor(decode=decode), cache)
        out["extractor_construct_s"] = round(time.perf_counter() - t0, 3)
        out["ws_after_extractor_mib"] = round(mem_mib()[0], 1)
        if args.profile == "gpu":
            out["gpu_after_load"] = gpu_used_mib()

    t0 = time.perf_counter()
    mem = Memory(str(db), embedder, extractor=extractor)
    out["memory_construct_s"] = round(time.perf_counter() - t0, 3)
    out["ws_after_memory_mib"] = round(mem_mib()[0], 1)

    # Write: one add() per round.
    adds = []
    for r in rounds:
        t0 = time.perf_counter()
        mem.add(r["turns"], "u")
        adds.append(time.perf_counter() - t0)
    out["write_per_round"] = stats(adds)
    out["write_total_s"] = round(sum(adds), 2)
    out["ws_after_writes_mib"] = round(mem_mib()[0], 1)
    if extractor is not None:
        out["extraction_stats"] = dict(getattr(mem, "extraction_stats", {}) or {})
        out["extractor_pins"] = extractor.pins
        if args.profile == "gpu":
            out["gpu_after_writes"] = gpu_used_mib()

    # Read: one warm-up, then passes x queries of recall() and get_context().
    t0 = time.perf_counter()
    mem.recall(queries[0], "u")
    out["first_recall_s"] = round(time.perf_counter() - t0, 4)
    rec, ctx, ctx_chars = [], [], []
    for _ in range(args.passes):
        for q in queries:
            t0 = time.perf_counter()
            hits = mem.recall(q, "u")
            rec.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            s = mem.get_context(q, "u")
            ctx.append(time.perf_counter() - t0)
            ctx_chars.append(len(s))
    out["recall_per_query"] = stats(rec)
    out["get_context_per_query"] = stats(ctx)
    out["recall_k"] = len(hits)
    out["context_chars_mean"] = round(statistics.fmean(ctx_chars), 1)
    out["ws_after_reads_mib"] = round(mem_mib()[0], 1)

    n_records = mem.store.count()
    out["records_in_store"] = n_records
    mem.store.close()
    out["store_bytes"] = db_bytes(db)
    if extractor is not None:
        out["cache_bytes"] = cache.stat().st_size
    ws, peak = mem_mib()
    out["ws_end_mib"], out["peak_ws_mib"] = round(ws, 1), round(peak, 1)
    out["peak_minus_dataset_mib"] = round(peak - (ws_dataset - ws0), 1)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
