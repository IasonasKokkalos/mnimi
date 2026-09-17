"""Ranking over rounds: SPEC's score and an exact top-k (PHASE4 D4).

``score = (w_sim * relevance + w_rec * recency) * salience`` for every record;
a round's score is its best record's; the ``k`` best rounds come back in score
order. ``rank_rounds`` is exact: it reads records in distance order in growing
batches and stops only when no unread record could reach the k-th round's
score — salience and recency never exceed 1 and the weights are never
negative, so an unread record scores at most ``max(w_sim * c + w_rec, 0)``,
``c`` being the lowest cosine read so far. With the default weights and a
uniform salience of 1.0 the result is ``Store.search_rounds``' rounds in its
order, record for record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .decay import logical_days
from .models import MemoryRecord, ScoredRecord

RANKING_SIMILARITY = "similarity"
RANKING_SCORE = "score"
RANKINGS = (RANKING_SIMILARITY, RANKING_SCORE)
SALIENCE_WEIGHT_KEYS = ("similarity", "recency")


def check_weights(weights: Mapping[str, float]) -> None:
    """``salience_weights`` is exactly ``similarity`` and ``recency``, each finite and >= 0."""
    if set(weights) != set(SALIENCE_WEIGHT_KEYS):
        raise ValueError(
            f"salience_weights must have exactly the keys {SALIENCE_WEIGHT_KEYS}, "
            f"got {sorted(weights)}"
        )
    for key in SALIENCE_WEIGHT_KEYS:
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"salience_weights[{key!r}] must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"salience_weights[{key!r}] must be finite and >= 0")


def recency(days: int, half_life_days: float) -> float:
    """``0.5 ** (days / half_life_days)``: 1.0 for the latest session, halving per half-life."""
    return 0.5 ** (days / half_life_days)


def combined_score(
    relevance: float, recency_value: float, salience: float, weights: Mapping[str, float]
) -> float:
    """SPEC §Ranking: ``(w_sim * relevance + w_rec * recency) * salience``."""
    return (weights["similarity"] * relevance + weights["recency"] * recency_value) * salience


def round_identity(record: MemoryRecord):
    """The key every record of one round shares; a v1 round (no key) is its own."""
    return record.round_key if record.round_key is not None else ("id", record.id)


def score_hit(
    record: MemoryRecord,
    cosine: float,
    now: str | None,
    half_life_days: float,
    weights: Mapping[str, float] | None,
) -> ScoredRecord:
    """One record's components; ``weights=None`` scores by similarity alone."""
    rec = recency(logical_days(now, record.created_at), half_life_days)
    score = cosine if weights is None else combined_score(cosine, rec, record.salience, weights)
    return ScoredRecord(
        record=record, relevance=cosine, recency=rec, salience=record.salience, score=score
    )


def rank_rounds(
    store,
    query_embedding: list[float],
    user_id: str,
    k: int,
    *,
    weights: Mapping[str, float],
    half_life_days: float,
    now: str | None,
    active_only: bool = False,
) -> list[ScoredRecord]:
    """The ``k`` best rounds by score, exact over the user's records (PHASE4 D4)."""
    fetch = k
    while True:
        hits = store.search(query_embedding, user_id=user_id, k=fetch, active_only=active_only)
        best: dict = {}
        for record, cosine in hits:
            candidate = score_hit(record, cosine, now, half_life_days, weights)
            key = round_identity(record)
            if key not in best or candidate.score > best[key].score:
                best[key] = candidate
        ranked = sorted(best.values(), key=lambda hit: -hit.score)
        if len(hits) < fetch:
            return ranked[:k]
        bound = max(weights["similarity"] * hits[-1][1] + weights["recency"], 0.0)
        if len(ranked) >= k and bound < ranked[k - 1].score:
            return ranked[:k]
        fetch *= 4
