"""Relative-date resolution, anchored on the message ``ts`` and nothing else.

The extractor copies a temporal expression verbatim (``when``); this module
turns it into ``valid_time`` — an ISO date (``2023-05-06``), or a month or
year prefix (``2023-04``, ``2022``) when the mention is that coarse, or
``None`` when it cannot be resolved (the mention is kept as
``time_mention``). Deterministic, table-driven, and it never reads the
clock: "today" is the session date, always (SPEC §Logical time).

Conventions, stated so a reader can predict every result:

* Month/day mentions without a year take the year of ``ts``, and a date that
  would fall after ``ts`` moves to the previous year — facts on this benchmark
  are mostly past events, and the prompt asks for what happened. **v2 (PHASE6
  D4, 2026-09-22):** when the mention or the fact it dates carries a future
  marker (:data:`FUTURE_MARKERS` — "next", "upcoming", "tomorrow", "planning",
  "plan(s) to", "will", "coming", "going to") and the mention itself carries
  no past marker (:data:`PAST_MARKERS`), the same forms resolve *forward*: the
  first occurrence on or after ``ts``. v1 dated "planning a trip to Hawaii in
  October" on a May session to the October before it — 48 of 1,062 dated
  facts on the n=500 contexts were a year off that way.
* "last weekend" / "this past weekend" (v2) is the most recent Saturday
  strictly before ``ts``, at day precision.
* "last <weekday>" is the most recent such day strictly before ``ts``;
  "next <weekday>" the first strictly after; a bare weekday the most recent
  on or before.
* "last week" / "the week before last" are the same weekday one / two weeks
  earlier (a point estimate); "last month" / "last year" are the previous
  month / year at that precision.
* Vague mentions ("recently", "a while ago", "a few days ago") resolve to
  ``None`` rather than to a guess.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

RESOLVER_VERSION = "v2"

#: v2 (PHASE6 D4): the words that make an undated month/day, month or weekday
#: mention resolve forward. Frozen with the version: an edit is a version bump.
FUTURE_MARKERS = (
    "next", "upcoming", "tomorrow", "planning", "plan to", "plans to", "will",
    "coming", "coming up", "going to",
)
#: A past marker in the mention itself keeps the v1 (backward) rule even when
#: the fact around it carries a future marker ("last April", "two weeks ago").
PAST_MARKERS = ("last", "ago", "yesterday", "back in", "this past", "past", "earlier", "previous")

_MONTHS = {
    name.lower(): i
    for i, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_WEEKDAYS.update({name.lower(): i for i, name in enumerate(calendar.day_abbr)})
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a couple of": 2, "couple of": 2, "a couple": 2,
}
_NUMBER_RE = (
    r"(\d{1,3}|a couple of|couple of|a couple|an|a|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|eleven|twelve)"
)
_UNIT_RE = r"(day|week|month|year)s?"
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RE = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))

_ANCHOR_RE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_AGO_RE = re.compile(rf"\b{_NUMBER_RE} {_UNIT_RE} (?:ago|earlier|before|back)\b")
_IN_RE = re.compile(rf"\bin {_NUMBER_RE} {_UNIT_RE}\b")
_LAST_WEEKDAY_RE = re.compile(
    rf"\b(last|this past|past|next|this|on|this coming|coming) ({_WEEKDAY_RE})\b"
)
_BARE_WEEKDAY_RE = re.compile(rf"\b({_WEEKDAY_RE})\b")
_LAST_UNIT_RE = re.compile(r"\b(last|this|next) (week|month|year)\b")
_BEFORE_LAST_RE = re.compile(r"\bthe (week|month|year) before last\b")
_MONTH_DAY_RE = re.compile(rf"\b({_MONTH_RE})\.? (\d{{1,2}})(?:st|nd|rd|th)?(?:,? (\d{{4}}))?\b")
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?({_MONTH_RE})\b(?:,? (\d{{4}}))?"
)
_MONTH_ONLY_RE = re.compile(
    rf"\b(?:in |last |this |during |back in )?({_MONTH_RE})\b(?: (\d{{4}}))?"
)
_NUMERIC_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_YEAR_RE = re.compile(r"\b(?:in |back in |since )?((?:19|20)\d{2})\b")
_FUTURE_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(FUTURE_MARKERS, key=len, reverse=True)) + r")\b"
)
_PAST_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(PAST_MARKERS, key=len, reverse=True)) + r")\b"
)
_LAST_WEEKEND_RE = re.compile(r"\b(last|this past|past) weekend\b")


def _anchor(ts: str | None) -> date | None:
    if not ts:
        return None
    m = _ANCHOR_RE.search(ts)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


#: The public name of the ``ts`` → date parse (PHASE3 Task 3: the supersede
#: ordering reads a session date the same way the resolver anchors on it).
anchor_date = _anchor


def _number(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return _NUMBERS.get(token)


def _shift_months(d: date, months: int) -> date:
    month_index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(month_index, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _shift(d: date, n: int, unit: str) -> date:
    if unit == "day":
        return d + timedelta(days=n)
    if unit == "week":
        return d + timedelta(weeks=n)
    if unit == "month":
        return _shift_months(d, n)
    return _shift_months(d, 12 * n)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _past_year(month: int, day: int, anchor: date) -> date | None:
    """A month/day with no year: the most recent occurrence on or before ``ts``."""
    candidate = _safe_date(anchor.year, month, day)
    if candidate is None:
        return None
    if candidate > anchor:
        candidate = _safe_date(anchor.year - 1, month, day)
    return candidate


def _next_year(month: int, day: int, anchor: date) -> date | None:
    """A month/day with no year under a future marker: the first occurrence on or after ``ts``."""
    candidate = _safe_date(anchor.year, month, day)
    if candidate is None:
        return None
    if candidate < anchor:
        candidate = _safe_date(anchor.year + 1, month, day)
    return candidate


def _year_of(month: int, day: int, anchor: date, forward: bool) -> date | None:
    return _next_year(month, day, anchor) if forward else _past_year(month, day, anchor)


def is_forward(mention: str, context: str | None) -> bool:
    """v2: a future marker in the mention or its fact, and no past marker in the mention."""
    text = " ".join(mention.lower().replace(",", " ").split())
    if _PAST_RE.search(text):
        return False
    if _FUTURE_RE.search(text):
        return True
    return bool(context) and bool(_FUTURE_RE.search(" ".join(context.lower().split())))


def verbatim_mention(mention: str | None, text: str) -> str | None:
    """``mention`` if it occurs in ``text`` (case- and whitespace-insensitive), else ``None``.

    The extractor is asked to copy the time expression verbatim; a mention
    that does not occur in the round is an invention (measured 2026-09-13 on
    the development set: the model echoed an example's "next week" once in
    forty rounds) and must not become a ``valid_time``. The fact itself is
    kept; only its date is dropped.
    """
    if not mention:
        return None
    needle = " ".join(mention.lower().split())
    haystack = " ".join(text.lower().split())
    return mention if needle and needle in haystack else None


def resolve(mention: str | None, ts: str | None, context: str | None = None) -> str | None:
    """``valid_time`` for a verbatim time mention, anchored on ``ts``.

    ``context`` (v2) is the fact the mention dates — its future markers make an
    undated month/day, month or weekday resolve forward (:func:`is_forward`).
    """
    if not mention:
        return None
    anchor = _anchor(ts)
    text = " ".join(mention.lower().replace(",", " ").split())
    forward = is_forward(mention, context)

    m = _ISO_RE.search(text)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d.isoformat() if d else None
    if anchor is None:
        return None

    if re.search(r"\b(today|tonight|this morning|this afternoon|this evening|earlier today|"
                 r"just now|right now)\b", text):
        return anchor.isoformat()
    if "day before yesterday" in text:
        return (anchor - timedelta(days=2)).isoformat()
    if "day after tomorrow" in text:
        return (anchor + timedelta(days=2)).isoformat()
    if re.search(r"\byesterday\b", text):
        return (anchor - timedelta(days=1)).isoformat()
    if re.search(r"\btomorrow\b", text):
        return (anchor + timedelta(days=1)).isoformat()
    if _LAST_WEEKEND_RE.search(text):
        # v2: the most recent Saturday strictly before ts, at day precision.
        delta = (anchor.weekday() - 5) % 7 or 7
        return (anchor - timedelta(days=delta)).isoformat()

    m = _LAST_WEEKDAY_RE.search(text)
    if m:
        qualifier, weekday = m.group(1), _WEEKDAYS[m.group(2)]
        if qualifier in ("next", "this coming", "coming"):
            delta = (weekday - anchor.weekday()) % 7 or 7
            return (anchor + timedelta(days=delta)).isoformat()
        if qualifier in ("last", "this past", "past"):
            delta = (anchor.weekday() - weekday) % 7 or 7
            return (anchor - timedelta(days=delta)).isoformat()
        if forward:  # v2: "this"/"on" under a future marker: first on or after
            delta = (weekday - anchor.weekday()) % 7
            return (anchor + timedelta(days=delta)).isoformat()
        delta = (anchor.weekday() - weekday) % 7  # "this"/"on": most recent on or before
        return (anchor - timedelta(days=delta)).isoformat()

    m = _AGO_RE.search(text)
    if m:
        n = _number(m.group(1))
        return _shift(anchor, -n, m.group(2)).isoformat() if n else None
    m = _IN_RE.search(text)
    if m:
        n = _number(m.group(1))
        return _shift(anchor, n, m.group(2)).isoformat() if n else None

    m = _BEFORE_LAST_RE.search(text)
    steps, unit = None, None
    if m:
        steps, unit = -2, m.group(1)
    else:
        m = _LAST_UNIT_RE.search(text)
        if m:
            steps, unit = {"last": -1, "this": 0, "next": 1}[m.group(1)], m.group(2)
    if steps is not None:
        if unit == "week":
            return (anchor + timedelta(weeks=steps)).isoformat()
        if unit == "month":
            return _shift_months(anchor, steps).strftime("%Y-%m")
        return str(anchor.year + steps)

    m = _MONTH_DAY_RE.search(text)
    if m:
        month, day = _MONTHS[m.group(1)], int(m.group(2))
        if m.group(3):
            d = _safe_date(int(m.group(3)), month, day)
        else:
            d = _year_of(month, day, anchor, forward)
        return d.isoformat() if d else None
    m = _DAY_MONTH_RE.search(text)
    if m:
        day, month = int(m.group(1)), _MONTHS[m.group(2)]
        if m.group(3):
            d = _safe_date(int(m.group(3)), month, day)
        else:
            d = _year_of(month, day, anchor, forward)
        return d.isoformat() if d else None
    m = _NUMERIC_RE.search(text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if m.group(3):
            year = int(m.group(3))
            year = year + 2000 if year < 100 else year
            d = _safe_date(year, month, day)
        else:
            d = _year_of(month, day, anchor, forward) if 1 <= month <= 12 else None
        return d.isoformat() if d else None
    m = _MONTH_ONLY_RE.search(text)
    if m:
        month = _MONTHS[m.group(1)]
        if m.group(2):
            return f"{int(m.group(2)):04d}-{month:02d}"
        if forward:  # v2: the first such month on or after the session's
            year = anchor.year if month >= anchor.month else anchor.year + 1
        else:
            year = anchor.year if month <= anchor.month else anchor.year - 1
        return f"{year:04d}-{month:02d}"
    m = _BARE_WEEKDAY_RE.search(text)
    if m:
        delta = (anchor.weekday() - _WEEKDAYS[m.group(1)]) % 7
        return (anchor - timedelta(days=delta)).isoformat()
    m = _YEAR_RE.search(text)
    if m:
        return m.group(1)
    return None
