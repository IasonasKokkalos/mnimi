"""Extractor throughput, byte-stability and prompt-development probe (Phase 2, 2.1/2.2).

Three jobs, no reader, no judge, no API:

* ``--rounds N`` — time the pinned extractor on N rounds of the n=100 slice's
  corpus, sampled stratified over ten length deciles (the same ids every run),
  and print prefill/decode tokens per second, seconds per round (p50/p90/mean)
  and the projected hours for the whole corpus (24,747 rounds). Gate 2.1:
  ≤ 24 h.
* ``--stability`` — run that sample twice in two FRESH processes, in order and
  shuffled, and diff the raw model outputs byte for byte. Gate 2.2: N/N
  identical. A round's output must not depend on the round before it (cold
  prefill per call) or on its position (one sequence) — the cache key assumes
  exactly that.
* ``--dev-set N`` — sample N rounds from the 400 questions OUTSIDE the slice,
  run the extractor and write round text + output to a JSONL for reading by
  hand. The prompt is developed here and never on the slice's evidence rounds
  (DECISIONS "Phase 2 pre-registration", protocol 1).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mnimi.memory import _messages_to_rounds

from ..dataset import DEFAULT_SAMPLE_SEED, SAMPLE_STRATIFIED, load
from ..runner import _session_to_messages

CORPUS_ROUNDS = 24_747  # the n=100 slice, PHASE1-RESULTS § Task 2
DECILES = 10


def _round_text(round_) -> str:
    return "\n".join(str(t.get("content", "")).strip() for t in round_.turns)


def slice_rounds(limit: int) -> list[dict]:
    """Every round of the slice as ``{id, question_id, session_id, index, turns}``."""
    out = []
    for q in load(limit=limit, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED):
        for session in q.sessions:
            for i, round_ in enumerate(_messages_to_rounds(_session_to_messages(session))):
                out.append(
                    {
                        "id": f"{q.question_id}/{session.session_id}/{i}",
                        "question_id": q.question_id,
                        "session_id": session.session_id,
                        "index": i,
                        "turns": round_.turns,
                        "chars": len(_round_text(round_)),
                    }
                )
    return out


def outside_rounds(limit: int) -> list[dict]:
    """Rounds of the questions NOT in the stratified slice — the development pool."""
    inside = {q.question_id for q in load(limit=limit, strategy=SAMPLE_STRATIFIED,
                                          seed=DEFAULT_SAMPLE_SEED)}
    out = []
    for q in load():
        if q.question_id in inside:
            continue
        for session in q.sessions:
            for i, round_ in enumerate(_messages_to_rounds(_session_to_messages(session))):
                out.append(
                    {
                        "id": f"{q.question_id}/{session.session_id}/{i}",
                        "question_id": q.question_id,
                        "turns": round_.turns,
                        "chars": len(_round_text(round_)),
                    }
                )
    return out


def sample_by_length(rounds: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` rounds spread over ``DECILES`` length deciles, ``n // DECILES`` each."""
    ordered = sorted(rounds, key=lambda r: (r["chars"], r["id"]))
    per = max(1, n // DECILES)
    rng = random.Random(seed)
    picked: list[dict] = []
    size = len(ordered)
    for d in range(DECILES):
        bucket = ordered[d * size // DECILES : (d + 1) * size // DECILES]
        picked.extend(rng.sample(bucket, min(per, len(bucket))))
    return picked[:n]


def _build_extractor(verbose: bool = False):
    from mnimi.extract.llama import QwenLlamaExtractor

    return QwenLlamaExtractor(verbose=verbose)


def run_rounds(rounds: list[dict], emit: Path | None, verbose: bool = False) -> dict:
    """Extract every round once; return timing stats; optionally write outputs."""
    extractor = _build_extractor(verbose=verbose)
    rows = []
    started = time.perf_counter()
    for i, r in enumerate(rounds, start=1):
        t0 = time.perf_counter()
        result = extractor.extract(r["turns"])
        seconds = time.perf_counter() - t0
        usage = getattr(extractor, "last_usage", None) or {}
        rows.append(
            {
                "id": r["id"],
                "seconds": round(seconds, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "facts": len(result.facts),
                "truncated": result.truncated,
                "truncated_input": bool(getattr(extractor, "last_truncated_input", False)),
                "raw_output": result.raw_output,
                "round": [{"role": t["role"], "content": t["content"]} for t in r["turns"]],
            }
        )
        print(
            f"[{i}/{len(rounds)}] {r['id']}  {seconds:5.2f}s  facts={len(result.facts)}"
            f"{'  TRUNC' if result.truncated else ''}  {time.perf_counter() - started:.0f}s",
            file=sys.stderr,
        )
    if emit is not None:
        emit.parent.mkdir(parents=True, exist_ok=True)
        with open(emit, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summarize(rows, pins=extractor.pins)


def summarize(rows: list[dict], pins: dict | None = None) -> dict:
    secs = [r["seconds"] for r in rows]
    prompt = [r["prompt_tokens"] for r in rows if r["prompt_tokens"]]
    completion = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    total_prompt, total_completion, total_secs = sum(prompt), sum(completion), sum(secs)
    mean = statistics.fmean(secs) if secs else 0.0
    return {
        "rounds": len(rows),
        "seconds_p50": statistics.median(secs) if secs else None,
        "seconds_p90": sorted(secs)[int(0.9 * (len(secs) - 1))] if secs else None,
        "seconds_mean": mean,
        "prompt_tokens_mean": statistics.fmean(prompt) if prompt else None,
        "completion_tokens_mean": statistics.fmean(completion) if completion else None,
        "tokens_per_second_overall": (total_prompt + total_completion) / total_secs
        if total_secs
        else None,
        "facts_mean": statistics.fmean(r["facts"] for r in rows) if rows else None,
        "empty_rate": sum(1 for r in rows if r["facts"] == 0) / len(rows) if rows else None,
        "truncated_outputs": sum(1 for r in rows if r["truncated"]),
        "truncated_inputs": sum(1 for r in rows if r["truncated_input"]),
        "projected_corpus_hours": mean * CORPUS_ROUNDS / 3600,
        "pins": pins,
    }


def format_summary(s: dict, skip_rate: float = 0.0) -> str:
    hours = s["projected_corpus_hours"] * (1.0 - skip_rate)
    lines = [
        f"rounds {s['rounds']}: seconds/round p50 {s['seconds_p50']:.2f}  "
        f"p90 {s['seconds_p90']:.2f}  mean {s['seconds_mean']:.2f}",
        f"prompt tokens mean {s['prompt_tokens_mean']}, completion tokens mean "
        f"{s['completion_tokens_mean']}, overall {s['tokens_per_second_overall']:.0f} tok/s",
        f"facts/round mean {s['facts_mean']:.2f}, [] rate {s['empty_rate']:.2%}, "
        f"truncated outputs {s['truncated_outputs']}, truncated inputs {s['truncated_inputs']}",
        f"projected corpus pass: {hours:.1f} h for {CORPUS_ROUNDS:,} rounds"
        + (f" (after a {skip_rate:.1%} pre-filter skip rate)" if skip_rate else "")
        + f"  -> gate 2.1 (<= 24 h): {'PASS' if hours <= 24 else 'FAIL'}",
    ]
    if s.get("pins"):
        lines.append("pins: " + json.dumps(s["pins"], sort_keys=True))
    return "\n".join(lines)


def compare_emitted(a: Path, b: Path) -> tuple[int, int, list[str]]:
    """``(identical, total, differing ids)`` between two emitted JSONL files."""
    def read(path: Path) -> dict:
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row["raw_output"]
        return rows

    ra, rb = read(a), read(b)
    ids = sorted(set(ra) | set(rb))
    differing = [i for i in ids if ra.get(i) != rb.get(i)]
    return len(ids) - len(differing), len(ids), differing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.probes.extractor_bench")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (ids are stable)")
    parser.add_argument("--limit", type=int, default=100,
                        help="the slice the corpus is drawn from")
    parser.add_argument("--order", choices=["in", "shuffled"], default="in")
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument("--emit", default=None, help="write per-round outputs to this JSONL")
    parser.add_argument("--stability", action="store_true",
                        help="run the sample twice in fresh processes (in order, then "
                             "shuffled) and diff the raw outputs")
    parser.add_argument("--dev-set", type=int, default=None,
                        help="sample N rounds OUTSIDE the slice for prompt development")
    parser.add_argument("--out", default="runs/extract_dev.jsonl")
    parser.add_argument("--skip-rate", type=float, default=0.0,
                        help="pre-filter round-skip rate to discount the projection by")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.stability:
        a = Path("runs/extract_stability_in.jsonl")
        b = Path("runs/extract_stability_shuffled.jsonl")
        base = [sys.executable, "-m", "evals.probes.extractor_bench", "--rounds", str(args.rounds),
                "--seed", str(args.seed), "--limit", str(args.limit)]
        subprocess.run([*base, "--order", "in", "--emit", str(a)], check=True)
        subprocess.run([*base, "--order", "shuffled", "--shuffle-seed", str(args.shuffle_seed),
                        "--emit", str(b)], check=True)
        identical, total, differing = compare_emitted(a, b)
        print(f"byte-stability: {identical}/{total} rounds identical across a shuffled restart"
              f"  -> gate 2.2: {'PASS' if identical == total else 'FAIL'}")
        for i in differing:
            print(f"  differs: {i}")
        return 0 if identical == total else 1

    if args.dev_set:
        pool = outside_rounds(args.limit)
        rounds = sample_by_length(pool, args.dev_set, args.seed)
        print(f"dev set: {len(rounds)} rounds from {len(pool)} outside the slice", file=sys.stderr)
        summary = run_rounds(rounds, Path(args.out), verbose=args.verbose)
        print(format_summary(summary))
        print(f"wrote {args.out} — read every row by hand before touching the prompt")
        return 0

    rounds = sample_by_length(slice_rounds(args.limit), args.rounds, args.seed)
    if args.order == "shuffled":
        random.Random(args.shuffle_seed).shuffle(rounds)
    print("sample ids: " + ", ".join(r["id"] for r in rounds), file=sys.stderr)
    summary = run_rounds(rounds, Path(args.emit) if args.emit else None, verbose=args.verbose)
    print(format_summary(summary, skip_rate=args.skip_rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
