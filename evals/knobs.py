"""The Phase 4 read-path flags, shared by ``python -m evals`` and the retrieval probe.

Seven flags, one definition (PHASE4 D9): six ``MemoryConfig`` fields —
``active_only``, ``ranking``, ``salience_weights``, ``recall_min_relevance``,
``decay_half_life_days``, ``decay_floor`` — and the harness's ``consolidate``
wiring (D8). A flag left unset is absent from the knobs, so the library default
applies; every value that reaches a ``MemoryConfig`` is a pin through
``MnimiSystem.retrieval_pins``. Stdlib only: ``python -m evals --help`` stays light.
"""

from __future__ import annotations

import argparse

#: Whether the mnimi arm calls ``Memory.consolidate`` once per store before the
#: question (PHASE4 D8). False until gate 4-iii adopts decay.
MNIMI_DEFAULT_CONSOLIDATE = False

_ON_OFF = ("on", "off")


def parse_salience_weights(text: str) -> dict[str, float]:
    """``"similarity=1.0,recency=0.0"`` -> ``{"similarity": 1.0, "recency": 0.0}``."""
    weights: dict[str, float] = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value, got {part!r}")
        try:
            weights[key.strip()] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not a number: {value!r}") from exc
    if set(weights) != {"similarity", "recency"}:
        raise argparse.ArgumentTypeError("salience weights need exactly similarity= and recency=")
    return weights


def add_read_path_flags(parser: argparse.ArgumentParser) -> None:
    """The seven flags, every default ``None`` (= the library's / the harness's default)."""
    parser.add_argument(
        "--active-only", default=None, choices=_ON_OFF,
        help="mnimi only: the read path sees only active records, so a superseded fact neither "
        "ranks nor renders (MemoryConfig.active_only, PHASE4 D5). Pinned (schema /10).",
    )
    parser.add_argument(
        "--ranking", default=None, choices=["similarity", "score"],
        help="mnimi only: 'similarity' is the v1.10 read path; 'score' is SPEC's "
        "(w_sim*relevance + w_rec*recency)*salience over rounds (MemoryConfig.ranking, "
        "PHASE4 D4). Pinned.",
    )
    parser.add_argument(
        "--salience-weights", default=None, type=parse_salience_weights,
        help="mnimi only, read under --ranking score: 'similarity=W,recency=W' "
        "(MemoryConfig.salience_weights). Pinned. Default: SPEC's similarity=1.0,recency=0.0.",
    )
    parser.add_argument(
        "--recall-min-relevance", default=None, type=float,
        help="mnimi only: drop recalled rounds below this cosine; 0.0 is off "
        "(MemoryConfig.recall_min_relevance, PHASE4 D7). Pinned. Never set in a run.",
    )
    parser.add_argument(
        "--decay-half-life-days", default=None, type=float,
        help="mnimi only: consolidate()'s half-life in logical days, also the recency "
        "constant (MemoryConfig.decay_half_life_days). Pinned. Default: SPEC's 30.",
    )
    parser.add_argument(
        "--decay-floor", default=None, type=float,
        help="mnimi only: the salience decay never goes below (MemoryConfig.decay_floor). "
        "Pinned. Default: SPEC's 0.15.",
    )
    parser.add_argument(
        "--consolidate", default=None, choices=_ON_OFF,
        help="mnimi only: call Memory.consolidate once per store after the last session, "
        "before the question (PHASE4 D8) - the day decay reaches the ranking. Pinned. "
        f"Default: {'on' if MNIMI_DEFAULT_CONSOLIDATE else 'off'}.",
    )


def read_path_knobs(args: argparse.Namespace) -> dict:
    """The ``MemoryConfig`` keywords the flags set; an unset flag is absent."""
    knobs: dict = {}
    if args.active_only is not None:
        knobs["active_only"] = args.active_only == "on"
    if args.ranking is not None:
        knobs["ranking"] = args.ranking
    if args.salience_weights is not None:
        knobs["salience_weights"] = dict(args.salience_weights)
    if args.recall_min_relevance is not None:
        knobs["recall_min_relevance"] = args.recall_min_relevance
    if args.decay_half_life_days is not None:
        knobs["decay_half_life_days"] = args.decay_half_life_days
    if args.decay_floor is not None:
        knobs["decay_floor"] = args.decay_floor
    return knobs


def consolidate_flag(args: argparse.Namespace) -> bool | None:
    """``--consolidate`` as a bool, ``None`` when unset (the harness default applies)."""
    return None if args.consolidate is None else args.consolidate == "on"


def read_path_resume_extras(args: argparse.Namespace) -> list[str]:
    """The set read-path flags, spelled so a batch resume repeats them exactly."""
    extras = []
    if args.active_only is not None:
        extras.append(f"--active-only {args.active_only}")
    if args.ranking is not None:
        extras.append(f"--ranking {args.ranking}")
    if args.salience_weights is not None:
        w = args.salience_weights
        extras.append(f"--salience-weights similarity={w['similarity']},recency={w['recency']}")
    if args.recall_min_relevance is not None:
        extras.append(f"--recall-min-relevance {args.recall_min_relevance}")
    if args.decay_half_life_days is not None:
        extras.append(f"--decay-half-life-days {args.decay_half_life_days}")
    if args.decay_floor is not None:
        extras.append(f"--decay-floor {args.decay_floor}")
    if args.consolidate is not None:
        extras.append(f"--consolidate {args.consolidate}")
    return extras
