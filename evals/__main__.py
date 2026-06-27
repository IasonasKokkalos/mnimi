"""``python -m evals`` — run a baseline against LongMemEval and print the table."""

from __future__ import annotations

import argparse
import os
import sys

# Default reader/judge. Override the reader with --model, the judge with
# --judge-model. (Opus 4.8 is the current default Anthropic model.)
DEFAULT_MODEL = "claude-opus-4-8"


def build_system(name: str):
    if name == "no_memory":
        from .systems.no_memory import NoMemorySystem

        return NoMemorySystem()
    if name == "full_history":
        from .systems.full_history import FullHistorySystem

        return FullHistorySystem()
    raise SystemExit(f"unknown system: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run a memory system against LongMemEval and print a score table.",
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=["no_memory", "full_history"],
        help="which baseline to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="run only the first N questions (default 10; a cheap smoke run)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="reader model id (Anthropic)"
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_MODEL, help="judge model id (Anthropic)"
    )
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="override the LongMemEval file (HF name or local path); "
        "e.g. longmemeval_oracle.json for a smaller download",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. The reader and judge call the "
            "Anthropic API; export your key and re-run.",
            file=sys.stderr,
        )
        return 2

    from .report import print_report
    from .runner import run

    system = build_system(args.system)

    def progress(done: int, total: int, q, correct: bool) -> None:
        mark = "PASS" if correct else "FAIL"
        print(f"[{done}/{total}] {mark}  {q.question_id} ({q.question_type})", file=sys.stderr)

    results = run(
        system,
        reader_model=args.model,
        judge_model=args.judge_model,
        limit=args.limit,
        dataset_file=args.dataset_file,
        progress=progress,
    )
    print_report(args.system, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
