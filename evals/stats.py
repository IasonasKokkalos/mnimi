"""Statistics for arm comparisons. One code path for n=20, n=100 and n=500.

Nothing here is hand-computed anywhere else: every interval, p-value and power
number quoted in a doc or an artifact comes from this module, so a doc and a
run cannot disagree about what the data said.

Three deliberate choices:

* **Wilson score intervals, not normal-approximation.** At n=20 with 2/20
  correct the normal interval runs below zero, which is not an interval.
* **Exact McNemar, not the chi-square approximation.** The approximation needs
  b + c to be reasonably large; at these slice sizes it routinely is not — 3
  discordant pairs is a realistic count here, and chi-square on 3 pairs is
  fiction.
* **Paired testing on per-question correctness.** Every arm answers the same
  questions, so the pairing is real and removes question difficulty from the
  comparison. It is also why the question-id sets must match exactly, which is
  asserted rather than assumed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import exp, lgamma, log, log1p, sqrt
from pathlib import Path
from statistics import NormalDist


@dataclass(frozen=True)
class Interval:
    """A proportion with a confidence interval. ``point`` is the raw rate."""

    successes: int
    n: int
    point: float
    low: float
    high: float
    alpha: float

    def as_percent(self, places: int = 1) -> str:
        p, lo, hi = (100 * v for v in (self.point, self.low, self.high))
        return f"{p:.{places}f}% [{lo:.{places}f}, {hi:.{places}f}]"


@dataclass(frozen=True)
class McNemar:
    """Result of an exact paired comparison of two arms.

    ``b`` and ``c`` are the discordant counts and are always reported: a
    p-value without them hides whether it rests on 3 pairs or 300.
    """

    b: int  # correct in the first arm, wrong in the second
    c: int  # wrong in the first arm, correct in the second
    n_pairs: int  # questions compared, not discordant pairs
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b + self.c


# Pre-specified comparisons, fixed before any n=100 or n=500 data exists.
# Reporting whichever pair happens to look good afterwards is how a benchmark
# number stops meaning anything; writing them down here is the commitment.
PRIMARY = ("mnimi", "naive_rag")
SECONDARY = (("mnimi", "no_memory"), ("mnimi", "full_history"), ("mnimi", "oracle"))


def wilson(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("wilson() needs at least one observation")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The boundary cases are exact analytically; clamping keeps float residue
    # (1.4e-17 for a lower bound of zero) out of published artifacts.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return Interval(successes, n, p, low, high, alpha)


def _binom_pmf_row(m: int, p: float) -> list[float]:
    """All pmf values for Binomial(m, p), by recurrence — no repeated lgamma."""
    if m < 0:
        return []
    if p <= 0.0:
        return [1.0] + [0.0] * m
    if p >= 1.0:
        return [0.0] * m + [1.0]
    row = [0.0] * (m + 1)
    row[0] = exp(m * log1p(-p))
    ratio = p / (1 - p)
    for k in range(m):
        row[k + 1] = row[k] * (m - k) / (k + 1) * ratio
    return row


def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return exp(
        lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log1p(-p)
    )


def exact_binomial_two_sided(k: int, m: int) -> float:
    """Two-sided exact p-value for ``k`` successes in ``m`` fair trials.

    The null distribution is symmetric, so doubling the smaller tail is exact
    rather than an approximation of the sum-of-small-probabilities rule.
    """
    if m == 0:
        return 1.0
    row = _binom_pmf_row(m, 0.5)
    tail = sum(row[: min(k, m - k) + 1])
    return min(1.0, 2 * tail)


def mcnemar_exact(b: int, c: int, n_pairs: int) -> McNemar:
    """Exact McNemar test on the discordant pairs."""
    return McNemar(b=b, c=c, n_pairs=n_pairs, p_value=exact_binomial_two_sided(b, b + c))


def discordance(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int, int]:
    """``(b, c, n_pairs)`` over two arms' per-question correctness.

    Fails loudly on mismatched question sets: comparing arms that answered
    different questions is not a paired test, it is two unpaired numbers with a
    p-value stapled on.
    """
    if set(first) != set(second):
        only_first = sorted(set(first) - set(second))[:5]
        only_second = sorted(set(second) - set(first))[:5]
        raise ValueError(
            "arms did not answer the same questions, so they cannot be paired: "
            f"{len(set(first) - set(second))} only in the first {only_first}, "
            f"{len(set(second) - set(first))} only in the second {only_second}"
        )
    b = sum(1 for q in first if first[q] and not second[q])
    c = sum(1 for q in first if not first[q] and second[q])
    return b, c, len(first)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment. Keys are comparison labels."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (label, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[label] = running
    return adjusted


def mcnemar_power(
    n_pairs: int, discordance_rate: float, gap: float, alpha: float = 0.05
) -> float:
    """Power of the exact McNemar test.

    Averages the conditional power over the distribution of the discordant
    count, because the exact test conditions on it: the number of discordant
    pairs is itself random, and a design that ignores that overstates power.

    ``discordance_rate`` is p01 + p10 (how often the arms disagree at all) and
    ``gap`` is the true accuracy difference p10 - p01.
    """
    if discordance_rate <= 0:
        return 0.0
    if abs(gap) > discordance_rate:
        raise ValueError("gap cannot exceed the discordance rate")
    psi = (1 + gap / discordance_rate) / 2  # P(a discordant pair favours arm 2)

    outer = _binom_pmf_row(n_pairs, discordance_rate)
    total = 0.0
    for m, weight in enumerate(outer):
        if weight < 1e-12:
            continue
        crit = -1
        null = _binom_pmf_row(m, 0.5)
        cumulative = 0.0
        for k in range(m // 2 + 1):
            cumulative += null[k]
            if min(1.0, 2 * cumulative) <= alpha:
                crit = k
            else:
                break
        if crit < 0:  # no rejection region exists at this m
            continue
        alt = _binom_pmf_row(m, psi)
        total += weight * (sum(alt[: crit + 1]) + sum(alt[m - crit :]))
    return total


def minimum_detectable_gap(
    n_pairs: int,
    discordance_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    tolerance: float = 1e-4,
) -> float | None:
    """Smallest true accuracy gap reaching ``power``, or ``None`` if impossible.

    ``None`` means no gap up to the discordance rate itself reaches the target
    — i.e. even a system that wins every single disagreement is not reliably
    detectable at this sample size.
    """
    ceiling = discordance_rate
    if mcnemar_power(n_pairs, discordance_rate, ceiling, alpha) < power:
        return None
    low, high = 0.0, ceiling
    while high - low > tolerance:
        mid = (low + high) / 2
        if mcnemar_power(n_pairs, discordance_rate, mid, alpha) >= power:
            high = mid
        else:
            low = mid
    return high


def load_correctness(run_dir: str | Path) -> dict[str, bool]:
    """Per-question correctness from a run's ``results.json``."""
    rows = json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))["results"]
    return {row["question_id"]: bool(row["correct"]) for row in rows}


def analyse(arms: dict[str, dict[str, bool]], alpha: float = 0.05) -> dict:
    """Wilson intervals per arm plus the pre-specified comparisons only.

    The primary comparison is reported uncorrected and labelled as such; the
    secondary family is Holm-corrected together. No other pair is tested,
    because a test chosen after seeing the data is not a test.
    """
    report: dict = {"arms": {}, "primary": None, "secondary": {}}
    for name, verdicts in sorted(arms.items()):
        report["arms"][name] = wilson(sum(verdicts.values()), len(verdicts), alpha)

    if all(name in arms for name in PRIMARY):
        first, second = PRIMARY
        b, c, n = discordance(arms[first], arms[second])
        report["primary"] = {
            "comparison": f"{first} vs {second}",
            "result": mcnemar_exact(b, c, n),
            "correction": "none (primary)",
        }

    raw = {}
    for first, second in SECONDARY:
        if first in arms and second in arms:
            b, c, n = discordance(arms[first], arms[second])
            raw[f"{first} vs {second}"] = mcnemar_exact(b, c, n)
    adjusted = holm({label: r.p_value for label, r in raw.items()})
    for label, result in raw.items():
        report["secondary"][label] = {
            "result": result,
            "p_holm": adjusted[label],
            "correction": "Holm",
        }
    return report


def power_table(
    discordance_rate: float, sizes=(100, 500), alpha: float = 0.05, power: float = 0.8
) -> dict[int, float | None]:
    """Minimum detectable accuracy gap at each sample size."""
    return {n: minimum_detectable_gap(n, discordance_rate, alpha, power) for n in sizes}


def _format(report: dict) -> str:
    lines = ["accuracy (Wilson 95% CI)", "------------------------"]
    for name, ci in report["arms"].items():
        lines.append(f"  {name:<20} {ci.as_percent():<26} ({ci.successes}/{ci.n})")

    primary = report.get("primary")
    if primary:
        r = primary["result"]
        lines += [
            "",
            "primary comparison (pre-specified, uncorrected)",
            "----------------------------------------------",
            f"  {primary['comparison']:<28} b={r.b} c={r.c} discordant={r.discordant} "
            f"n={r.n_pairs}  p={r.p_value:.4f}",
        ]
    if report["secondary"]:
        lines += ["", "secondary comparisons (Holm-corrected)", "-" * 38]
        for label, entry in report["secondary"].items():
            r = entry["result"]
            lines.append(
                f"  {label:<28} b={r.b} c={r.c} discordant={r.discordant} "
                f"n={r.n_pairs}  p={r.p_value:.4f}  p_holm={entry['p_holm']:.4f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m evals.stats <run_dir> [...]`` — same path at every n."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m evals.stats <run_dir> [<run_dir> ...]", file=sys.stderr)
        return 2
    arms = {}
    for directory in argv:
        pins = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
        arms[pins["pins"]["system"]] = load_correctness(directory)
    print(_format(analyse(arms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
