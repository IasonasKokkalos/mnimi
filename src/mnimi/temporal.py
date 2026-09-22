"""The time-aware term: the query's relative-date window against each record's date (PHASE6 D2, D3).

A question such as "what kitchen appliance did I buy 10 days ago?" names a
window on the calendar. Dense similarity cannot see it — "10 days ago" and
"today" embed nowhere near a date — but every fact record carries a resolved
``valid_time`` and every round its session date. This module turns the
expression into a window, anchored on the **question date**, and the ranking
adds ``w_time * time_match`` for every record whose effective time falls in
it (:func:`mnimi.ranking.combined_score`).

The question date reaches the library as a documented query prefix (D2):
``"[Current date: <ts>]\\n<question>"``. :func:`split_query` strips it; the
bare question is what gets embedded, so with ``time_weight = 0`` nothing
moves — a test asserts the vector is byte-identical. Without the prefix, or
without a parseable expression, there is no window and the term is inert.

Everything here is deterministic date arithmetic on the standard library. The
grammar, the slack and the matching rule are frozen as text and hashed
(:func:`temporal_rules_hash`, a harness pin — read at query time, no stored
byte changes). An edit is a version bump and a dated DECISIONS entry, never
an in-place change under a run, and the values were fixed in the
pre-registration before any probe ran; none is tuned against the benchmark.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import NamedTuple

from .extract.protocol import canonical, sha256_text
from .extract.resolver import _MONTHS, _NUMBER_RE, _WEEKDAYS, _number, _shift, anchor_date

TEMPORAL_RULES_VERSION = "v1"

#: Frozen as text, hashed into the harness pins as ``temporal_rules_hash``.
TEMPORAL_RULES = (
    "query: an optional leading '[Current date: <ts>]' line is the anchor and is stripped "
    "before embedding; no anchor or no expression means no window and time_match = 0",
    "grammar (first match in this order): (past|last|previous) <N> (day|week|month|year)s; "
    "(last|past|previous) (week|month|year); <N> (day|week|month|year)s? ago|earlier|before|back; "
    "(in|during|of)( the month of)? <month>; yesterday; last <weekday>",
    "N: digits, or a/an/one..twelve, or 'a couple of'; a weekday before '<N> units ago' is dropped",
    "target = anchor - N units (day, week: days; month, year: calendar months); "
    "yesterday = anchor - 1 day; last <weekday> = the most recent such day strictly before the "
    "anchor",
    "window: day and week precision = target +/- 3 days; month and year precision = the target's "
    "calendar month; 'in <month>' = the most recent such month on or before the anchor; "
    "last/previous month|year = the previous calendar month|year; last week = the 7 days "
    "ending at the anchor; past <unit> = past 1 <unit>; past N units = [anchor - N units, "
    "anchor]; ranges carry no slack",
    "effective time of a record = valid_time when set, else its session date (created_at)",
    "time_match = 1 when a YYYY-MM-DD effective time lies inside the window or a YYYY-MM effective "
    "time's month overlaps it; a bare year never matches; else 0",
    "score = (w_sim * relevance + w_rec * recency + w_time * time_match) * salience",
)

SLACK_DAYS = 3
QUERY_DATE_PREFIX = "[Current date: "
_PREFIX_RE = re.compile(r"^\[Current date: ([^\]\n]+)\]\n")

_UNIT = r"(day|week|month|year)s?"
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_RANGE_N_RE = re.compile(rf"\b(?:past|last|previous) {_NUMBER_RE} {_UNIT}\b")
_RANGE_RE = re.compile(r"\b(last|past|previous) (week|month|year)\b")
_AGO_RE = re.compile(rf"\b{_NUMBER_RE} {_UNIT} (?:ago|earlier|before|back)\b")
_IN_MONTH_RE = re.compile(rf"\b(?:in|during|of)(?: the month of)? ({_MONTH_RE})\b")
_YESTERDAY_RE = re.compile(r"\byesterday\b")
_LAST_WEEKDAY_RE = re.compile(rf"\blast ({_WEEKDAY_RE})\b")


class Window(NamedTuple):
    """A closed date range and the precision of the expression that named it."""

    start: date
    end: date
    precision: str  # "day" | "week" | "month" | "year" | "range"
    expression: str


def temporal_rules_hash() -> str:
    """Digest of the frozen rules (a harness pin)."""
    return sha256_text(
        canonical({"version": TEMPORAL_RULES_VERSION, "rules": list(TEMPORAL_RULES),
                   "slack_days": SLACK_DAYS})
    )


def dated_query(question_date: str | None, question: str) -> str:
    """The documented prefix form; the bare question when there is no date."""
    if not question_date:
        return question
    return f"{QUERY_DATE_PREFIX}{question_date}]\n{question}"


def split_query(query: str) -> tuple[str | None, str]:
    """``(question_date, bare question)``; the date is ``None`` without the prefix."""
    m = _PREFIX_RE.match(query)
    if not m:
        return None, query
    return m.group(1), query[m.end():]


def _month_window(year: int, month: int, expression: str, precision: str = "month") -> Window:
    last = calendar.monthrange(year, month)[1]
    return Window(date(year, month, 1), date(year, month, last), precision, expression)


def _around(target: date, precision: str, expression: str) -> Window:
    if precision in ("month", "year"):
        return _month_window(target.year, target.month, expression, precision)
    return Window(target - timedelta(days=SLACK_DAYS), target + timedelta(days=SLACK_DAYS),
                  precision, expression)


def parse_window(question: str, anchor: date | None) -> Window | None:
    """The window a question's relative-date expression names, or ``None``."""
    if anchor is None:
        return None
    text = " ".join(question.lower().replace(",", " ").split())

    m = _RANGE_N_RE.search(text)
    if m:
        n = _number(m.group(1))
        if n:
            return Window(_shift(anchor, -n, m.group(2)), anchor, "range", m.group(0))
    m = _RANGE_RE.search(text)
    if m:
        unit = m.group(2)
        if m.group(1) == "past":  # "the past month" = the last such span ending now
            return Window(_shift(anchor, -1, unit), anchor, "range", m.group(0))
        if unit == "week":
            return Window(anchor - timedelta(days=7), anchor, "range", m.group(0))
        if unit == "month":
            prev = _shift(anchor, -1, "month")
            return _month_window(prev.year, prev.month, m.group(0), "range")
        return Window(date(anchor.year - 1, 1, 1), date(anchor.year - 1, 12, 31), "range",
                      m.group(0))
    m = _AGO_RE.search(text)
    if m:
        n = _number(m.group(1))
        if n:
            unit = m.group(2)
            return _around(_shift(anchor, -n, unit), unit, m.group(0))
    m = _IN_MONTH_RE.search(text)
    if m:
        month = _MONTHS[m.group(1)]
        year = anchor.year if month <= anchor.month else anchor.year - 1
        return _month_window(year, month, m.group(0))
    m = _YESTERDAY_RE.search(text)
    if m:
        return _around(anchor - timedelta(days=1), "day", m.group(0))
    m = _LAST_WEEKDAY_RE.search(text)
    if m:
        weekday = _WEEKDAYS[m.group(1)]
        delta = (anchor.weekday() - weekday) % 7 or 7
        return _around(anchor - timedelta(days=delta), "day", m.group(0))
    return None


def window_for(query: str) -> tuple[str, Window | None]:
    """``(bare question, window)`` for a possibly prefixed query."""
    question_date, question = split_query(query)
    return question, parse_window(question, anchor_date(question_date))


def effective_time(valid_time: str | None, created_at: str | None) -> str | None:
    """``valid_time`` when set (``YYYY-MM-DD`` / ``YYYY-MM`` / ``YYYY``), else the session date."""
    if valid_time:
        return valid_time
    d = anchor_date(created_at)
    return d.isoformat() if d else None


def time_match(effective: str | None, window: Window | None) -> float:
    """1.0 when the effective time falls inside the window, else 0.0 (rule 7)."""
    if window is None or not effective:
        return 0.0
    parts = effective.split("-")
    try:
        if len(parts) == 3:
            d = date(int(parts[0]), int(parts[1]), int(parts[2]))
            return 1.0 if window.start <= d <= window.end else 0.0
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            first = date(year, month, 1)
            last = date(year, month, calendar.monthrange(year, month)[1])
            return 1.0 if first <= window.end and last >= window.start else 0.0
    except ValueError:
        return 0.0
    return 0.0
