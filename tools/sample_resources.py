"""Sample a process's CPU and memory while it runs; summarise at the end (PT1-RUNBOOK Task 2a).

usage: python tools/sample_resources.py --match "evals" --out runs/<run_id>/resources.csv
           [--interval 5]
       python tools/sample_resources.py --summarise runs/<run_id>/resources.csv

Finds the newest Python process whose command line contains --match (the harness: "evals"),
then every --interval seconds writes one row: wall time, the process's CPU percent (100 = one
core busy), its RSS in MB, its thread count, the whole machine's CPU percent and free RAM. It
stops when the process exits and prints the summary. Needs psutil (not a project dependency:
``pip install psutil`` in the run environment). Diagnostic only: nothing here enters a run's
pins, manifest or score.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import psutil


def find(match: str) -> psutil.Process:
    me = os.getpid()
    best = None
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
            continue
        if match in cmd and "sample_resources" not in cmd:
            if best is None or p.info["create_time"] > best.info["create_time"]:
                best = p
    if best is None:
        sys.exit(f"no python process with {match!r} on its command line")
    return psutil.Process(best.info["pid"])


def sample(proc: psutil.Process, out: str, interval: float) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    proc.cpu_percent(None)
    psutil.cpu_percent(None)
    t0 = time.time()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "proc_cpu_pct", "proc_rss_mb", "proc_threads", "sys_cpu_pct",
                    "sys_avail_mb"])
        print(f"sampling pid {proc.pid} every {interval}s -> {out}", flush=True)
        while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            time.sleep(interval)
            try:
                with proc.oneshot():
                    row = [round(time.time() - t0, 1), proc.cpu_percent(None),
                           round(proc.memory_info().rss / 2**20, 1), proc.num_threads(),
                           psutil.cpu_percent(None),
                           round(psutil.virtual_memory().available / 2**20)]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            w.writerow(row)
            fh.flush()
    print("process ended", flush=True)
    summarise(out)


def summarise(path: str) -> None:
    with open(path, newline="") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        print("no samples")
        return
    cpu = [r["proc_cpu_pct"] for r in rows]
    rss = [r["proc_rss_mb"] for r in rows]
    cores = os.cpu_count() or 1
    print(f"samples {len(rows)} over {rows[-1]['t_s'] / 60:.1f} min on {cores} logical cores")
    p95 = sorted(cpu)[int(0.95 * (len(cpu) - 1))]
    print(f"process CPU: mean {statistics.mean(cpu):.0f} %, p95 {p95:.0f} %, "
          f"max {max(cpu):.0f} %  (100 = one core)")
    print(f"process RSS: median {statistics.median(rss):.0f} MB, max {max(rss):.0f} MB")
    print(f"machine CPU mean {statistics.mean(r['sys_cpu_pct'] for r in rows):.0f} %, "
          f"free RAM min {min(r['sys_avail_mb'] for r in rows):.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="evals")
    ap.add_argument("--out", default="resources.csv")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--summarise", metavar="CSV")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    sample(find(a.match), a.out, a.interval)


if __name__ == "__main__":
    main()
