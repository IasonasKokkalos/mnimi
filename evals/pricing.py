"""Prices, projection, ledger and the budget gate for the API reader family.

The rule (PLAN.md, 2026-09-11): every sitting prints its projected cost
before submission and refuses above the remaining cap. The cap is $50 for the
whole programme. This module is the whole mechanism:

- a **dated price table** (the official pricing page; a model that is not in
  it cannot be projected and therefore cannot run);
- a **projection** computed from the real request bodies before the first
  call — an upper bound: input at the harness's chars/4 trim-gate convention
  (gpt-4o tokenises this dataset closer to 4.7 chars/token, so the estimate
  is high), output at ``max_tokens`` per request (actual answers run ~1/3 of
  it), judge calls at a fixed ~600+10 tokens;
- a **ledger** of actual spend (from the API's own ``usage`` counts) appended
  after every run, kept in the gitignored ``.cache/`` because it is a fact
  about this machine's key, not about the code;
- the **gate**: projection + spend-to-date > budget → refuse before any call.

None of this enters ``pins_hash``. Money is diagnostic.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

#: Official OpenAI pricing page, read on this date. USD per million tokens.
PRICES_AS_OF = "2026-09-11"
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
}
#: The Batch API halves both input and output.
BATCH_DISCOUNT = 0.5
#: The programme cap (mnimi docs/PLAN.md §4.1). Changing it is a decision,
#: not a flag: `--api-budget-usd` exists for one deliberate override and is
#: echoed loudly when used.
API_BUDGET_USD = 50.0
#: Judge call estimate: question + gold + a ~270-token answer + template.
JUDGE_PROMPT_TOKENS_EST = 600
JUDGE_COMPLETION_TOKENS_EST = 10
#: Mirrors runner._CHARS_PER_TOKEN; the projection and the trim gate must
#: agree on the estimate or the projection would describe a different prompt.
CHARS_PER_TOKEN = 4

LEDGER_ENV = "MNIMI_API_LEDGER"
DEFAULT_LEDGER = Path(".cache") / "api_ledger.jsonl"


class UnpricedModelError(KeyError):
    """A model with no row in the price table cannot be projected."""


def price(model: str) -> dict[str, float]:
    try:
        return PRICES_USD_PER_MTOK[model]
    except KeyError:
        raise UnpricedModelError(
            f"no price for {model!r} in evals.pricing (table as of {PRICES_AS_OF}); "
            "a run that cannot be projected cannot spend"
        ) from None


def estimate_usd(
    model: str, prompt_tokens: int, completion_tokens: int, *, batch: bool = False
) -> float:
    p = price(model)
    usd = (prompt_tokens or 0) * p["input"] / 1e6 + (completion_tokens or 0) * p["output"] / 1e6
    return usd * BATCH_DISCOUNT if batch else usd


@dataclass
class Projection:
    model: str
    batch: bool
    n_requests: int
    reader_prompt_tokens_est: int
    reader_completion_tokens_max: int
    reader_usd: float
    judge_model: str | None
    judge_calls: int
    judge_usd: float
    total_usd: float
    basis: str

    def line(self) -> str:
        rate = "batch rate" if self.batch else "standard rate"
        return (
            f"projected: reader ${self.reader_usd:.4f} ({rate}, {self.n_requests} requests, "
            f"~{self.reader_prompt_tokens_est:,} in / ≤{self.reader_completion_tokens_max:,} out)"
            f" + judge ${self.judge_usd:.4f} ({self.judge_calls} calls)"
            f" = ${self.total_usd:.4f} upper bound"
        )


def project(
    *,
    model: str,
    items,
    batch: bool,
    judge_model: str | None = None,
    judge_calls: int = 0,
) -> Projection:
    """Upper-bound cost of sending ``items`` (BatchItem-shaped) plus judging."""
    prompt_est = 0
    completion_max = 0
    for item in items:
        for message in item.body.get("messages", []):
            prompt_est += len(message.get("content") or "") // CHARS_PER_TOKEN
        completion_max += int(item.body.get("max_tokens") or 0)
    reader_usd = estimate_usd(model, prompt_est, completion_max, batch=batch)
    judge_usd = 0.0
    if judge_model is not None and judge_calls:
        judge_usd = judge_calls * estimate_usd(
            judge_model, JUDGE_PROMPT_TOKENS_EST, JUDGE_COMPLETION_TOKENS_EST
        )
    return Projection(
        model=model,
        batch=batch,
        n_requests=len(items),
        reader_prompt_tokens_est=prompt_est,
        reader_completion_tokens_max=completion_max,
        reader_usd=reader_usd,
        judge_model=judge_model,
        judge_calls=judge_calls,
        judge_usd=judge_usd,
        total_usd=reader_usd + judge_usd,
        basis=(
            f"upper bound: input at {CHARS_PER_TOKEN} chars/token, output at max_tokens per "
            f"request, judge at ~{JUDGE_PROMPT_TOKENS_EST}+{JUDGE_COMPLETION_TOKENS_EST} "
            f"tokens/call at standard rate; prices as of {PRICES_AS_OF}"
        ),
    )


# ------------------------------------------------------------------ ledger
def ledger_path() -> Path:
    return Path(os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER)


def entries() -> list[dict]:
    path = ledger_path()
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append(entry: dict) -> Path:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def spent_usd() -> float:
    """Actual where known, projected where a batch is still out.

    A resume line names the submitted line it replaces (``supersedes_ts``), so
    a batch is counted once: at its projection until collected, then at its
    actual cost.
    """
    rows = entries()
    superseded = {row["supersedes_ts"] for row in rows if row.get("supersedes_ts")}
    total = 0.0
    for row in rows:
        if row.get("ts") in superseded:
            continue
        actual = row.get("actual_usd")
        total += actual if actual is not None else (row.get("projected_usd") or 0.0)
    return total


def gate(total_usd: float, budget_usd: float, basis: str = "") -> str | None:
    """``None`` when the run may spend; otherwise the refusal message."""
    spent = spent_usd()
    if total_usd + spent > budget_usd:
        return (
            f"refusing to spend: projected ${total_usd:.4f} + spent so far "
            f"${spent:.2f} exceeds the budget of ${budget_usd:.2f}"
            f"{' (' + basis + ')' if basis else ''}. Nothing was sent. The programme cap "
            f"is ${API_BUDGET_USD:.2f} (mnimi docs/PLAN.md); pass --api-budget-usd only "
            "for a deliberate, recorded override."
        )
    return None


def status_line(budget_usd: float) -> str:
    spent = spent_usd()
    return f"spent so far ${spent:.4f} of ${budget_usd:.2f} | remaining ${budget_usd - spent:.4f}"


def main() -> int:
    rows = entries()
    print(f"ledger: {ledger_path()}  (prices as of {PRICES_AS_OF})")
    for row in rows:
        actual = row.get("actual_usd")
        print(
            f"  {row.get('ts')}  {row.get('stage', '?'):8s} {str(row.get('system')):13s} "
            f"n={row.get('limit')}  projected ${row.get('projected_usd') or 0:.4f}  "
            f"actual {f'${actual:.4f}' if actual is not None else 'pending'}"
            f"{'  ' + row['note'] if row.get('note') else ''}"
        )
    spent = spent_usd()
    print(f"spent ${spent:.4f} of ${API_BUDGET_USD:.2f}; remaining ${API_BUDGET_USD - spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "API_BUDGET_USD",
    "BATCH_DISCOUNT",
    "PRICES_AS_OF",
    "PRICES_USD_PER_MTOK",
    "Projection",
    "UnpricedModelError",
    "append",
    "asdict",
    "entries",
    "estimate_usd",
    "gate",
    "ledger_path",
    "project",
    "spent_usd",
]
