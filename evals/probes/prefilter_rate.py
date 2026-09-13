"""Pre-filter drop rate on the slice, and the false-drop check on evidence rounds (2.3).

CPU, minutes, no model: walks every round of the stratified slice through
``mnimi.extract.prefilter`` exactly as ``Memory.add`` will, and prints the
drop rate by role and by rule, the share of rounds that would never reach the
model, and — the gate — every round holding a ``has_answer`` turn that the
pre-filter would skip. The gate is 0: a pre-filter that drops evidence is
wrong, whatever it saves.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from mnimi.extract import prefilter

from ..dataset import DEFAULT_SAMPLE_SEED, SAMPLE_STRATIFIED, load
from .retrieval import _tag_rounds


def measure(limit: int) -> dict:
    by_role_rule: Counter = Counter()
    turns_by_role: Counter = Counter()
    rounds = rounds_skipped = evidence_rounds = 0
    false_drops: list[tuple[str, str, int, list[str]]] = []
    evidence_turns_dropped: list[tuple[str, str, int, str]] = []
    for q in load(limit=limit, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED):
        for session in q.sessions:
            for i, (round_, is_evidence) in enumerate(_tag_rounds(session)):
                rounds += 1
                evidence_rounds += is_evidence
                for turn in round_.turns:
                    role = turn.get("role", "")
                    turns_by_role[role] += 1
                    rule = prefilter.decide(turn)
                    if rule is not None:
                        by_role_rule[(role, rule)] += 1
                        if is_evidence and turn.get("has_answer"):
                            evidence_turns_dropped.append(
                                (q.question_id, session.session_id, i, rule)
                            )
                keep, dropped = prefilter.keep_round(round_.turns)
                if not keep:
                    rounds_skipped += 1
                    if is_evidence:
                        false_drops.append((q.question_id, session.session_id, i, dropped))
    return {
        "rounds": rounds,
        "rounds_skipped": rounds_skipped,
        "evidence_rounds": evidence_rounds,
        "turns_by_role": dict(turns_by_role),
        "drops_by_role_rule": {f"{r}/{rule}": n for (r, rule), n in sorted(by_role_rule.items())},
        "false_drops": false_drops,
        "evidence_turns_dropped": evidence_turns_dropped,
        "prefilter_lexicon_hash": prefilter.prefilter_lexicon_hash(),
    }


def format_report(m: dict) -> str:
    lines = [
        f"rounds {m['rounds']:,}; rounds the model would never see {m['rounds_skipped']:,} "
        f"({m['rounds_skipped'] / m['rounds']:.2%}); evidence rounds {m['evidence_rounds']}",
        "turn drops by role/rule:",
    ]
    for key, n in m["drops_by_role_rule"].items():
        role = key.split("/", 1)[0]
        lines.append(f"  {key:40s} {n:6d}  ({n / m['turns_by_role'][role]:.2%} of {role} turns)")
    lines.append(
        f"false drops on evidence rounds: {len(m['false_drops'])}  -> gate 2.3 (= 0): "
        f"{'PASS' if not m['false_drops'] else 'FAIL'}"
    )
    for qid, sid, i, rules in m["false_drops"]:
        lines.append(f"  FALSE DROP {qid} {sid} #{i} rules={rules}")
    lines.append(
        f"evidence-carrying turns dropped inside kept rounds (diagnostic): "
        f"{len(m['evidence_turns_dropped'])}"
    )
    for qid, sid, i, rule in m["evidence_turns_dropped"]:
        lines.append(f"  evidence turn dropped: {qid} {sid} #{i} rule={rule}")
    lines.append(f"prefilter_lexicon_hash {m['prefilter_lexicon_hash'][:16]}…")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.probes.prefilter_rate")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--log-drops", action="store_true", help="print every filtered: line")
    args = parser.parse_args(argv)
    if args.log_drops:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    report = measure(args.limit)
    print(format_report(report))
    return 0 if not report["false_drops"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
