"""Minimal on-disk verdict cache for the LLM judge.

Call-avoidance only: keyed on ``(judge_fingerprint, question_id,
sha256(predicted))`` so a re-run with an identical prediction, graded by the
same judge, skips the API call and returns the same verdict.

**The fingerprint is load-bearing, and its absence was a bug.** The key was
``(question_id, sha256(predicted))`` alone until 2026-07-29, which meant a
change to the judge model or the judge prompt did not invalidate a single
cached verdict: re-grading after a prompt change replayed the old verdicts and
reported "nothing changed", which is indistinguishable from a real null result.
Reader drift was always caught (it changes ``predicted``, so it changes the
key); judge drift never was.

This is NOT the Phase B staged-artifact system — no reproducibility header, no
predictions.jsonl. Just a flat JSON file the judge reads through on every call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_PATH = ".cache/judge_verdicts.json"


class JudgeCache:
    """Flat JSON verdict cache. Counts hits/misses for run reporting."""

    def __init__(self, path: str = DEFAULT_PATH, *, judge_fingerprint: str = "") -> None:
        self.path = Path(path)
        self.judge_fingerprint = judge_fingerprint
        self.hits = 0
        self.misses = 0
        self._data: dict[str, bool] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt cache is a performance problem, not a correctness one:
                # start empty and let the run repopulate it.
                self._data = {}

    def key(self, question_id: str, predicted: str) -> str:
        digest = hashlib.sha256(predicted.encode("utf-8")).hexdigest()
        return f"{self.judge_fingerprint}:{question_id}:{digest}"

    def get(self, question_id: str, predicted: str) -> bool | None:
        """Return the cached verdict, or ``None`` on miss. Records the outcome."""
        k = self.key(question_id, predicted)
        # ``in`` not truthiness: a cached ``False`` is a hit, not a miss.
        if k in self._data:
            self.hits += 1
            return self._data[k]
        self.misses += 1
        return None

    def set(self, question_id: str, predicted: str, verdict: bool) -> None:
        """Store a verdict and write through to disk immediately."""
        self._data[self.key(question_id, predicted)] = verdict
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=0, sort_keys=True), encoding="utf-8"
        )
