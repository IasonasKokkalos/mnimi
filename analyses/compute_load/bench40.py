"""The 40 dev rounds of evals.probes.extractor_bench --dev-set 40 (seed 0, limit 100),
under a chosen profile. Reuses the bench's own sampling, loop and summary.

python bench40.py --profile cpu|gpu --out FILE.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.probes import extractor_bench as bench

from mnimi.extract.llama import DECODE, QwenLlamaExtractor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    decode = DECODE if args.profile == "gpu" else {**DECODE, "n_gpu_layers": 0, "max_tokens": 2048}

    bench._build_extractor = lambda verbose=False: QwenLlamaExtractor(
        decode=decode, verbose=verbose
    )
    rounds = bench.sample_by_length(bench.outside_rounds(100), args.n, 0)
    print(
        f"profile={args.profile} rounds={len(rounds)} ids[:3]={[r['id'] for r in rounds[:3]]}",
        file=sys.stderr,
    )
    summary = bench.run_rounds(rounds, Path(args.out))
    print(bench.format_summary(summary))
    lines = Path(args.out).read_text(encoding="utf-8").splitlines()
    secs = sorted(float(json.loads(line)["seconds"]) for line in lines if line.strip())
    print(f"min {secs[0]:.2f}  max {secs[-1]:.2f}  n {len(secs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
