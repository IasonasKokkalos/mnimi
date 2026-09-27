"""What every third-party memory system's adapter shares (PHASE8 D9, amended 2026-09-27).

An adapter runs the shipped system and changes nothing inside it. It converts the
harness's messages to the system's input, handing the session date over only
through the system's own fields and never as a pseudo-turn. It converts the
system's hits to the one renderer's input. It pins the system's version, embedder,
LLM, configuration and context budget.

- **The context budget (the maintainer's decision of 2026-09-27, taken before any
  competitor number existed).** A third-party system's units are its memories:
  sentence fragments or single facts, 20× smaller than the rounds mnimi and
  naive_rag render. Ten of them would hand the reader about 250 tokens against
  mnimi's 5,364. So a third-party system's context is **the longest prefix of its
  own recall order whose rendering fits the budget** (always at least the top hit).
  The budget is fixed for every question type and counted in the reader's own
  tokenizer. Its value (5,364 tokens) is the shipped mnimi arm's mean rendered
  context, measured from committed predictions and recorded in each configuration.
  The in-house arms keep their identical ``k``.
- **The one renderer.** Hits render as ``memory:`` turns through
  ``mnimi.memory.render_turns``, the same bytes every arm's context goes through,
  under a ``[Session date: …]`` header per session. The speaker label ``memory`` is
  the disclosure that these units are the system's memories, not rounds.
- **Order.** Hits are selected in the system's recall order, then shown oldest
  session first, as ``_time_ordered`` shows rounds. Within one session they keep
  recall order, never an id order. Ids can be random per run, and a context must
  not depend on them.
- **Ids.** ``retrieved_ids`` are the system's own ids of the hits shown, in recall
  order.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import NamedTuple

from mnimi.memory import RENDER_UNIT_TURNS, render_turns, render_unit_template_hash

from .. import artifacts
from ..base import MemorySystem

#: The competitor keys build_pins accepts (schema /12); omitted there when None.
COMPETITOR_PIN_KEYS = (
    "competitor_name",
    "competitor_version",
    "competitor_embedder",
    "competitor_llm",
    "competitor_config_hash",
    "competitor_context_budget_tokens",
)

_WEEKDAY = re.compile(r"\s*\([^)]*\)\s*")


class Hit(NamedTuple):
    """One memory a system returned: its own id, its text, its session's date.

    ``turns``, when set, are the verbatim turns of a stored round (a system that
    stores naive_rag's units, PHASE8 Task 15): the hit renders as those turns, the
    bytes naive_rag shows for the same round, instead of one ``memory:`` line.
    """

    id: str
    text: str
    ts: str | None
    turns: tuple | None = None


def render_hits(hits: list[Hit], fmt: str = "text") -> str:
    """Oldest session first, recall order within a session, through the one renderer."""
    ordered = sorted(hits, key=lambda h: h.ts or "")  # stable: recall order within a date
    turns: list[dict] = []
    for h in ordered:
        if h.turns:
            turns.extend({"role": t.get("role", ""), "content": t.get("content", ""), "ts": h.ts}
                         for t in h.turns)
        else:
            turns.append({"role": "memory", "content": h.text, "ts": h.ts})
    return render_turns(turns, fmt=fmt)


def make_counter(name: str) -> Callable[[str], int]:
    """A token counter by name: ``tiktoken:<encoding>`` (the reader's own) or ``chars/4``."""
    if name == "chars/4":
        return lambda text: (len(text) + 3) // 4
    if name.startswith("tiktoken:"):
        import tiktoken

        encoding = tiktoken.get_encoding(name.split(":", 1)[1])
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    raise ValueError(f"unknown token counter {name!r}; expected 'tiktoken:<encoding>' or 'chars/4'")


def fit_to_budget(
    hits: list[Hit], fmt: str, budget_tokens: int, count: Callable[[str], int]
) -> list[Hit]:
    """The longest recall-order prefix whose rendering fits ``budget_tokens``; at least one hit.

    Rendering more hits never shortens the text, so the fitting prefixes are an
    initial segment and a binary search finds the longest.
    """
    if not hits:
        return []
    lo, hi = 1, len(hits)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(render_hits(hits[:mid], fmt)) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return hits[:lo]


def epoch(ts: str | None) -> float | None:
    """A LongMemEval session date (``2023/05/20 (Sat) 09:00``) as UTC epoch seconds.

    The weekday is stripped before parsing so no locale enters; UTC so the value
    is the same on every machine. ``None`` for anything that does not parse.
    """
    if not ts:
        return None
    try:
        clean = _WEEKDAY.sub(" ", ts).strip()
        return datetime.strptime(clean, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def config_hash(block: dict) -> str:
    """The committed configuration block's identity, as it enters the pins."""
    return artifacts.fingerprint(artifacts.canonical(block))


class CompetitorSystem(MemorySystem):
    """The harness contract for a third-party system; a subclass drives the system."""

    name = "competitor"

    def __init__(self, competitor_config: dict, render_format: str = "text") -> None:
        self._config_block = dict(competitor_config)
        self._fmt = render_format
        context = competitor_config.get("context") or {}
        if "budget_tokens" not in context:
            raise ValueError("a third-party system's configuration needs context.budget_tokens "
                             "(the maintainer's decision of 2026-09-27)")
        self.budget_tokens = int(context["budget_tokens"])
        self.candidates = int(context.get("candidates", 500))
        self._count = make_counter(context.get("tokenizer", "tiktoken:o200k_base"))
        self._last_ids: list[str] = []

    def competitor_pins(self) -> dict:
        """The system's name, version, embedder and LLM, keyed as ``COMPETITOR_PIN_KEYS``."""
        raise NotImplementedError

    def retrieval_pins(self) -> dict:
        return {
            "render_unit": RENDER_UNIT_TURNS,
            "render_unit_template_hash": render_unit_template_hash(RENDER_UNIT_TURNS),
            **self.competitor_pins(),
            "competitor_config_hash": config_hash(self._config_block),
            "competitor_context_budget_tokens": self.budget_tokens,
        }

    def context_from(self, hits: list[Hit]) -> str:
        """Fit the recall-ordered hits to the budget, record their ids, render them."""
        fitted = fit_to_budget(hits, self._fmt, self.budget_tokens, self._count)
        self._last_ids = [h.id for h in fitted]
        return render_hits(fitted, self._fmt)

    def retrieved_ids(self) -> list[str] | None:
        return list(self._last_ids)
