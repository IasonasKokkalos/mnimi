"""Retrieval probe: the library's own ingest and search, no reader, no judge.

Rebuilt from ``mnimi docs/RETRIEVAL.md`` §0 (2026-07-31), whose scripts lived
in a session scratchpad and were lost. Everything here is observation of the
code the eval runs: ``MnimiSystem`` / ``NaiveRagSystem`` are driven exactly as
``evals.runner.build_batch_items`` drives them (one ``add()`` per session, in
the haystack's file order — batch composition changes the tokenizer's padding
and therefore the vectors), the dedup screens are counted at the store call
site by wrapping ``search`` and ``insert`` on the live store (delegating,
never replacing), and the query is searched once at k=50 rounds through the
read path's own ``Store.search_rounds`` so every k <= 50 is read off one
result. A retrieved record maps back to ``(session_id, round_index)`` by
re-deriving the rounds with the same ``_messages_to_rounds``.

Definitions (RETRIEVAL §0): an *evidence round* is an ingested round holding a
``has_answer`` turn; ANY@k = at least one evidence round in the top-k; ALL@k =
every evidence round in the top-k; a round dropped by dedup is a miss even if
a near-duplicate survived (the strict rule). Abstention questions are excluded
from recall denominators.

The extraction era (PHASE2): a round is several records — its round record
and its fact records, one ``round_key`` — and each is screened against its
own kind. The walk mirrors ``mnimi.memory.round_pieces`` piece by piece (the
extractor's cache makes the mirror's call free), a round counts as lost only
when none of its records survived, and the stage's counters (rounds sent to
the model, ``[]`` outputs, truncations, pre-filter skips, facts stored) ride
in every row. With ``--extractor qwen3`` this probe IS the corpus pass: the
cache fills as it walks, and gate 4-i is read off its output.

The probe is a gate, not a score: a retrieval-side change is sent to the API
only after this shows what it did to recall (DECISIONS, "Phase 1
pre-registration", "Phase 2 pre-registration").
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mnimi import MemoryConfig
from mnimi.conflict.screens import ACTION_KEEP, KEEP_REASONS
from mnimi.memory import (
    _messages_to_rounds,
    _normalize,
    fact_verdict,
    new_extraction_stats,
    round_pieces,
)
from mnimi.models import KIND_FACT, KIND_ROUND

from ..dataset import DEFAULT_SAMPLE_SEED, SAMPLE_STRATIFIED, Question, load
from ..runner import _session_to_messages
from ..systems.mnimi import EVAL_USER_ID

SEARCH_K = 50


@dataclass
class DropEvent:
    session_id: str
    round_index: int
    screen: str  # "exact" | "cosine"
    cosine: float | None
    neighbour_session_id: str | None  # the k=1 hit's session, for cosine drops
    is_evidence: bool
    piece: int = 0  # which piece of the round (0 for a round embedded whole)
    kind: str = KIND_ROUND  # "round" | "fact" — the record kind that was dropped


@dataclass
class QuestionProbe:
    question_id: str
    category: str
    is_abstention: bool
    n_rounds: int
    n_evidence_rounds: int
    stored: int
    evidence_ranks: list[int]  # 1-based rank in the k=50 search; -1 = absent or dropped
    top_sessions: list[str]  # session_id of each top-50 hit, in rank order
    drops: list[DropEvent] = field(default_factory=list)
    # Evidence rounds none of whose pieces survived dedup: (session_id,
    # round_index, screen of the last piece dropped). With chunking off and no
    # extractor this is exactly the evidence-flagged drops.
    lost_evidence: list[list] = field(default_factory=list)
    # The extraction stage, per question (zero without an extractor).
    facts_stored: int = 0
    rounds_sent_to_model: int = 0
    empty_extractions: int = 0
    truncated_outputs: int = 0
    truncated_inputs: int = 0
    prefilter_skips: int = 0
    # The Phase 3 stage, per question (empty / zero without it, and absent
    # from Phase 2 probe files — the defaults keep those loadable). Keeps are
    # cosine-pass fact pairs the screens routed away from the merge, by
    # reason; conflicts and supersessions come with Task 3.
    screen_keeps: dict = field(default_factory=dict)
    conflicts: dict = field(default_factory=dict)
    superseded: int = 0


class _Observed:
    """Wraps a live Store's search/insert so dedup decisions are visible."""

    def __init__(self, store) -> None:
        self.store = store
        self.probes: list[list] = []  # k=1 search results, in call order
        self.inserted: list[int] = []  # record ids, in call order
        self._search, self._insert = store.search, store.insert
        store.search = self.search
        store.insert = self.insert

    def search(self, embedding, user_id, k, kind=None, active_only=False):
        hits = self._search(embedding, user_id=user_id, k=k, kind=kind, active_only=active_only)
        if k == 1:
            self.probes.append(hits)
        return hits

    def insert(self, record):
        stored = self._insert(record)
        self.inserted.append(stored.id)
        return stored

    def restore(self) -> None:
        self.store.search, self.store.insert = self._search, self._insert


def _memory_of(system):
    """``(store, embedder, config, query_embedding_fn, dedups, extractor)`` for either arm."""
    if hasattr(system, "_memory"):  # MnimiSystem
        mem = system._memory
        return mem.store, mem.embedder, mem.config, mem._query_embedding, True, mem.extractor
    store, emb, config = system._store, system._embedder, system._config  # NaiveRagSystem
    return (
        store, emb, config, (lambda q: emb.embed([config.query_instruction + q])[0]), False, None
    )


def _tag_rounds(session):
    """``[(round, is_evidence)]`` for one session, re-deriving round membership."""
    rounds = _messages_to_rounds(_session_to_messages(session))
    source = [t for t in session.turns if str(t.get("content", "")).strip()]
    tagged, cursor = [], 0
    for round_ in rounds:
        flags = []
        for turn in round_.turns:
            if source[cursor]["content"] != turn["content"]:
                raise AssertionError("round/turn partition drifted from the source turns")
            flags.append(bool(source[cursor].get("has_answer")))
            cursor += 1
        tagged.append((round_, any(flags)))
    if cursor != len(source):
        raise AssertionError("not every source turn landed in a round")
    return tagged


def _exact_key(piece, round_, config):
    """The exact screen's key, as ``Memory.add`` computes it per kind and scope."""
    normalized = _normalize(piece.content)
    if piece.kind == KIND_FACT and config.dedup_scope != "store":
        return (round_.ts, normalized)
    return normalized


def probe_question(system, q: Question) -> QuestionProbe:
    """Ingest one question's haystack through ``system`` and rank its evidence."""
    system.reset()
    store, embedder, config, query_embedding, dedups, extractor = _memory_of(system)
    observed = _Observed(store)
    # A retrieved record maps back to its round by record id: the walk below
    # mirrors the system's insert order exactly (asserted), so the n-th insert
    # is the n-th surviving piece. Content is not a key — naive_rag stores
    # identical rounds twice, and the strict rule needs each round ranked on
    # its own. ``content_to_round`` only names the session a dedup neighbour
    # came from.
    content_to_round: dict[str, tuple[str, int]] = {}
    id_to_round: dict[int, tuple[str, int]] = {}
    all_rounds: list[tuple[str, int, bool]] = []
    seen: dict[str, set] = {KIND_ROUND: set(), KIND_FACT: set()}
    drops: list[DropEvent] = []
    lost_evidence: list[list] = []
    stats = new_extraction_stats()
    n_rounds = n_evidence = facts_stored = 0
    # The screens' keeps, counted by the mirror and checked against the
    # library's own counters at the end (mnimi only).
    keeps = dict.fromkeys(KEEP_REASONS, 0)
    conflict_stats = getattr(getattr(system, "_memory", None), "conflict_stats", None)
    conflict_before = dict(conflict_stats) if conflict_stats is not None else None
    try:
        for session in q.sessions:  # file order — exactly what runner._sessions_for feeds
            tagged = _tag_rounds(session)
            probe_cursor = len(observed.probes)
            insert_cursor = len(observed.inserted)
            system.add(_session_to_messages(session))
            for i, (round_, is_evidence) in enumerate(tagged):
                n_rounds += 1
                n_evidence += is_evidence
                all_rounds.append((session.session_id, i, is_evidence))
                survived = 0
                last_screen = None
                # The mirror's extractor call is a cache hit: system.add() just made it.
                pieces = round_pieces(round_, embedder, config, extractor, stats)
                for p_idx, piece in enumerate(pieces):
                    content_to_round.setdefault(piece.content, (session.session_id, i))
                    if not dedups:  # naive_rag stores every piece
                        id_to_round[observed.inserted[insert_cursor]] = (session.session_id, i)
                        insert_cursor += 1
                        survived += 1
                        continue
                    key = _exact_key(piece, round_, config)
                    if key in seen[piece.kind]:
                        drops.append(DropEvent(
                            session.session_id, i, "exact", None, None, is_evidence, p_idx,
                            piece.kind,
                        ))
                        last_screen = "exact"
                        continue
                    hits = observed.probes[probe_cursor]
                    probe_cursor += 1
                    cosine = hits[0][1] if hits else None
                    if hits and cosine >= config.dedup_cosine_threshold and (
                        config.dedup_scope == "store" or hits[0][0].created_at == round_.ts
                    ):
                        # The same function the write path calls (PHASE3): a
                        # fact pair the screens keep apart is inserted, not dropped.
                        verdict = fact_verdict(piece, hits[0][0], config)
                        if verdict is None or verdict.action != ACTION_KEEP:
                            neighbour = content_to_round.get(hits[0][0].content)
                            drops.append(DropEvent(
                                session.session_id, i, "cosine", cosine,
                                neighbour[0] if neighbour else None, is_evidence, p_idx,
                                piece.kind,
                            ))
                            last_screen = "cosine"
                            continue
                        keeps[verdict.reason] += 1
                    seen[piece.kind].add(key)
                    id_to_round[observed.inserted[insert_cursor]] = (session.session_id, i)
                    insert_cursor += 1
                    survived += 1
                    facts_stored += piece.kind == KIND_FACT
                if is_evidence and survived == 0:
                    lost_evidence.append([session.session_id, i, last_screen])
            if probe_cursor != len(observed.probes) or insert_cursor != len(observed.inserted):
                raise AssertionError("the walk did not mirror the system's dedup/insert calls")
        conflicts: dict = {}
        superseded = 0
        if conflict_stats is not None:
            delta = {k: conflict_stats[k] - conflict_before[k] for k in conflict_stats}
            library_keeps = {r: delta[f"kept_{r.replace('-', '_')}"] for r in KEEP_REASONS}
            if library_keeps != keeps:
                raise AssertionError(
                    f"the walk's screen keeps {keeps} differ from the library's {library_keeps}"
                )
            conflicts = {r: delta[f"conflicts_{r}"] for r in ("negation", "functional", "numeric")}
            superseded = delta["superseded"]
        # The read path's own search: k counts rounds, every record of one
        # round collapses to its best-ranked one (Store.search_rounds).
        hits = store.search_rounds(query_embedding(q.question), user_id=EVAL_USER_ID, k=SEARCH_K)
    finally:
        observed.restore()
    ranked = [id_to_round[r.id] for r, _cos in hits]
    ranks: dict[tuple[str, int], int] = {}
    for rank, (sid, i) in enumerate(ranked, start=1):
        ranks.setdefault((sid, i), rank)
    evidence_ranks = [ranks.get((sid, i), -1) for (sid, i, ev) in all_rounds if ev]
    return QuestionProbe(
        question_id=q.question_id,
        category=q.category,
        is_abstention=q.is_abstention,
        n_rounds=n_rounds,
        n_evidence_rounds=n_evidence,
        stored=store.count(EVAL_USER_ID),
        evidence_ranks=evidence_ranks,
        top_sessions=[sid for sid, _i in ranked],
        drops=drops,
        lost_evidence=lost_evidence,
        facts_stored=facts_stored,
        rounds_sent_to_model=stats["rounds_sent"],
        empty_extractions=stats["empty_extractions"],
        truncated_outputs=stats["truncated_outputs"],
        truncated_inputs=stats["truncated_inputs"],
        prefilter_skips=stats["prefilter_skips"],
        screen_keeps=keeps,
        conflicts=conflicts,
        superseded=superseded,
    )


def write_rows(path: Path, rows: list[QuestionProbe]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")
    return path


def read_rows(path: Path) -> list[QuestionProbe]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QuestionProbe(**{**r, "drops": [DropEvent(**d) for d in r["drops"]]}) for r in raw]


def build_extractor(name: str | None):
    """The extractor a flag names: ``None``/``"none"`` for the v1 arm, ``"qwen3"``
    for the pinned model (needs the ``[extract]`` extra)."""
    if name in (None, "none"):
        return None
    if name == "qwen3":
        from mnimi.extract.llama import QwenLlamaExtractor

        return QwenLlamaExtractor()
    raise SystemExit(f"unknown extractor {name!r}; expected none or qwen3")


def build(system_name: str, config: MemoryConfig, extractor: str | None = None,
          extractor_cache: str | None = None):
    if system_name == "mnimi":
        from ..systems.mnimi import MnimiSystem

        extractor_obj = build_extractor(extractor)
        if extractor_obj is None:
            return MnimiSystem(config=config)
        return MnimiSystem(config=config, extractor=extractor_obj, extraction_cache=extractor_cache)
    if system_name == "naive_rag":
        from ..systems.naive_rag import NaiveRagSystem

        return NaiveRagSystem(config=config)
    raise SystemExit(f"the probe supports mnimi and naive_rag, not {system_name}")


def run(system_name: str, limit: int, out: Path, config: MemoryConfig,
        extractor: str | None = None, extractor_cache: str | None = None) -> list[QuestionProbe]:
    system = build(system_name, config, extractor, extractor_cache)
    questions = load(limit=limit, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED)
    rows: list[QuestionProbe] = []
    started = time.time()
    for i, q in enumerate(questions, start=1):
        rows.append(probe_question(system, q))
        write_rows(out, rows)  # inspectable while it runs
        row = rows[-1]
        extra = (
            f" facts={row.facts_stored} sent={row.rounds_sent_to_model} "
            f"empty={row.empty_extractions} trunc={row.truncated_outputs}"
            if row.rounds_sent_to_model
            else ""
        )
        if any(row.screen_keeps.values()) or row.superseded:
            extra += f" keeps={row.screen_keeps} superseded={row.superseded}"
        cache = getattr(system, "cache_stats", None)
        if cache:
            extra += f" cache={cache}"
        print(
            f"[{i}/{len(questions)}] {q.question_id} ranks={row.evidence_ranks} "
            f"drops={len(row.drops)}{extra}  {time.time() - started:.0f}s",
            file=sys.stderr,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.probes.retrieval")
    parser.add_argument("--system", choices=["mnimi", "naive_rag"], default="mnimi")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dedup-threshold", type=float, default=MemoryConfig().dedup_cosine_threshold
    )
    parser.add_argument("--dedup-scope", choices=["store", "session"], default=None)
    parser.add_argument("--query-instruction", default=None, help="bge | '' | a literal")
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--conflict-resolution", choices=["on", "off"], default=None,
        help="mnimi only: the Phase 3 stage (screens + supersession); 'off' is the v1.9 path. "
        "Default: the library's.",
    )
    parser.add_argument(
        "--dedup-entropy-gate", type=float, default=None,
        help="mnimi only: MemoryConfig.dedup_entropy_gate. Default: the library's 2.0.",
    )
    parser.add_argument(
        "--extractor", choices=["none", "qwen3"], default=None,
        help="mnimi only: the write-time extractor (PHASE2). Default: none (the v1 arm). "
        "With qwen3 this run is the corpus pass: the extraction cache fills as it walks.",
    )
    parser.add_argument(
        "--extractor-cache", default=None,
        help="path of the extraction cache SQLite file (default: "
        ".cache/extract/<extractor pins hash>.sqlite)",
    )
    args = parser.parse_args(argv)
    instruction = args.query_instruction
    if instruction == "bge":
        from mnimi.embeddings import BGE_QUERY_INSTRUCTION

        instruction = BGE_QUERY_INSTRUCTION
    knobs = dict(
        dedup_cosine_threshold=args.dedup_threshold,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )
    if args.dedup_scope is not None:
        knobs["dedup_scope"] = args.dedup_scope
    if instruction is not None:
        knobs["query_instruction"] = instruction
    if args.conflict_resolution is not None:
        knobs["conflict_resolution"] = args.conflict_resolution == "on"
    if args.dedup_entropy_gate is not None:
        knobs["dedup_entropy_gate"] = args.dedup_entropy_gate
    config = MemoryConfig(**knobs)
    run(args.system, args.limit, Path(args.out), config, args.extractor, args.extractor_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
