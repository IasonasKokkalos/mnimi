"""LongMemEval loader + downloader.

Schema verified against github.com/xiaowu0162/LongMemEval and the
``xiaowu0162/longmemeval-cleaned`` HuggingFace dataset. The dataset viewer
throws an Arrow type error on the nested ``haystack_sessions`` field, so we
fetch the raw JSON via ``hf_hub_download`` and parse with stdlib ``json`` rather
than ``datasets.load_dataset`` — leaner and no schema-casting surprises.

Per-question fields:
  question_id, question_type, question, answer, question_date,
  haystack_session_ids, haystack_dates, haystack_sessions, answer_session_ids
Each haystack session is a list of turns ``{"role", "content"[, "has_answer"]}``.
Abstention questions have a ``question_id`` ending in ``_abs``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "xiaowu0162/longmemeval-cleaned"
# The "small" variant (~500 questions, ~40 sessions each, ~115k tokens).
DEFAULT_FILE = "longmemeval_s_cleaned.json"
# Override the data file (e.g. the 15MB "longmemeval_oracle.json") for cheap runs.
FILE_ENV = "LONGMEMEVAL_FILE"
# Where the (gitignored) download is cached.
CACHE_ENV = "LONGMEMEVAL_CACHE"
DEFAULT_CACHE = ".data"

# The six answerable categories. Abstention variants share their base category.
CATEGORIES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


@dataclass
class Session:
    """One chat session from a question's haystack."""

    session_id: str
    date: str
    turns: list[dict]


@dataclass
class Question:
    """One LongMemEval evaluation instance."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[Session]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        """Abstention questions ask something the history cannot answer."""
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        """Base category, with any ``_abs`` suffix stripped for aggregation."""
        qt = self.question_type
        return qt[: -len("_abs")] if qt.endswith("_abs") else qt


def _cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE))


def download(filename: str | None = None) -> Path:
    """Return a local path to the dataset file, downloading it if needed."""
    filename = filename or os.environ.get(FILE_ENV, DEFAULT_FILE)
    cache_dir = _cache_dir()
    target = cache_dir / filename
    if target.exists():
        return target
    # Eval-only dependency, imported lazily so importing this module is cheap.
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    fetched = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(fetched)


def load(limit: int | None = None, filename: str | None = None) -> list[Question]:
    """Load questions. ``filename`` may be a HF filename or a local path."""
    name = filename or os.environ.get(FILE_ENV, DEFAULT_FILE)
    candidate = Path(name)
    path = candidate if candidate.exists() else download(name)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    questions = [_parse(item) for item in raw]
    if limit is not None:
        questions = questions[:limit]
    return questions


def _parse(item: dict) -> Question:
    session_ids = item.get("haystack_session_ids") or []
    dates = item.get("haystack_dates") or []
    contents = item.get("haystack_sessions") or []
    sessions: list[Session] = []
    for i, turns in enumerate(contents):
        sessions.append(
            Session(
                session_id=session_ids[i] if i < len(session_ids) else f"session-{i}",
                date=dates[i] if i < len(dates) else "",
                turns=turns or [],
            )
        )
    return Question(
        question_id=item["question_id"],
        question_type=item["question_type"],
        question=item["question"],
        answer=item.get("answer", ""),
        question_date=item.get("question_date", ""),
        sessions=sessions,
        answer_session_ids=item.get("answer_session_ids") or [],
    )
