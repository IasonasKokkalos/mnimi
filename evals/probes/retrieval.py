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
result. A retrieved record maps back to ``(session_id,
round_index)`` by re-deriving the rounds with the same ``_messages_to_rounds``.

Definitions (RETRIEVAL §0): an *evidence round* is an ingested round holding a
``has_answer`` turn; ANY@k = at least one evidence round in the top-k; ALL@k =
every evidence round in the top-k; a round dropped by dedup is a miss even if
a near-duplicate survived (the strict rule). Abstention questions are excluded
from recall denominators.

The probe is a gate, not a score: a retrieval-side change is sent to the API
only after this shows what it did to recall (DECISIONS, "Phase 1
pre-registration").
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mnimi import MemoryConfig
from mnimi.memory import _messages_to_rounds, _normalize, round_pieces

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
    piece: int = 0  # which window of the round (0 for a round embedded whole)


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
    # round_index, screen of the last piece dropped). With chunking off this
    # is exactly the evidence-flagged drops.
    lost_evidence: list[list] = field(default_factory=list)


class _Observed:
    """Wraps a live Store's search/insert so dedup decisions are visible."""

    def __init__(self, store) -> None:
        self.store = store
        self.probes: list[list] = []  # k=1 search results, in call order
        self.inserted: list[int] = []  # record ids, in call order
        self._search, self._insert = store.search, store.insert
        store.search = self.search
        store.insert = self.insert

    def search(self, embedding, user_id, k):
        hits = self._search(embedding, user_id=user_id, k=k)
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
    """``(store, embedder, config, query_embedding_fn, dedups)`` for either arm."""
    if hasattr(system, "_memory"):  # MnimiSystem
        mem = system._memory
        return mem.store, mem.embedder, mem.config, mem._query_embedding, True
    store, emb, config = system._store, system._embedder, system._config  # NaiveRagSystem
    return store, emb, config, (lambda q: emb.embed([config.query_instruction + q])[0]), False


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


def probe_question(system, q: Question) -> QuestionProbe:
    """Ingest one question's haystack through ``system`` and rank its evidence."""
    system.reset()
    store, embedder, config, query_embedding, dedups = _memory_of(system)
    observed = _Observed(store)
    # A retrieved record maps back to its round by record id: the walk below
    # mirrors the system's insert order exactly (asserted), so the n-th insert
    # is the n-th surviving round. Content is not a key — naive_rag stores
    # identical rounds twice, and the strict rule needs each round ranked on
    # its own. ``content_to_round`` only names the session a dedup neighbour
    # came from.
    content_to_round: dict[str, tuple[str, int]] = {}
    id_to_round: dict[int, tuple[str, int]] = {}
    all_rounds: list[tuple[str, int, bool]] = []
    seen_normalized: set[str] = set()
    drops: list[DropEvent] = []
    lost_evidence: list[list] = []
    n_rounds = n_evidence = 0
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
                for p_idx, piece in enumerate(round_pieces(round_, embedder, config)):
                    content_to_round.setdefault(piece.content, (session.session_id, i))
                    if not dedups:  # naive_rag stores every piece
                        id_to_round[observed.inserted[insert_cursor]] = (session.session_id, i)
                        insert_cursor += 1
                        survived += 1
                        continue
                    normalized = _normalize(piece.content)
                    if normalized in seen_normalized:
                        drops.append(DropEvent(
                            session.session_id, i, "exact", None, None, is_evidence, p_idx,
                        ))
                        last_screen = "exact"
                        continue
                    hits = observed.probes[probe_cursor]
                    probe_cursor += 1
                    cosine = hits[0][1] if hits else None
                    if hits and cosine >= config.dedup_cosine_threshold and (
                        config.dedup_scope == "store" or hits[0][0].created_at == round_.ts
                    ):
                        neighbour = content_to_round.get(hits[0][0].content)
                        drops.append(DropEvent(
                            session.session_id, i, "cosine", cosine,
                            neighbour[0] if neighbour else None, is_evidence, p_idx,
                        ))
                        last_screen = "cosine"
                        continue
                    seen_normalized.add(normalized)
                    id_to_round[observed.inserted[insert_cursor]] = (session.session_id, i)
                    insert_cursor += 1
                    survived += 1
                if is_evidence and survived == 0:
                    lost_evidence.append([session.session_id, i, last_screen])
            if probe_cursor != len(observed.probes) or insert_cursor != len(observed.inserted):
                raise AssertionError("the walk did not mirror the system's dedup/insert calls")
        # The read path's own search: k counts rounds, windows of one round
        # collapse to their best-ranked window (Store.search_rounds).
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
    )


def write_rows(path: Path, rows: list[QuestionProbe]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")
    return path


def read_rows(path: Path) -> list[QuestionProbe]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QuestionProbe(**{**r, "drops": [DropEvent(**d) for d in r["drops"]]}) for r in raw]


def build(system_name: str, config: MemoryConfig):
    if system_name == "mnimi":
        from ..systems.mnimi import MnimiSystem

        return MnimiSystem(config=config)
    if system_name == "naive_rag":
        from ..systems.naive_rag import NaiveRagSystem

        return NaiveRagSystem(config=config)
    raise SystemExit(f"the probe supports mnimi and naive_rag, not {system_name}")


def run(system_name: str, limit: int, out: Path, config: MemoryConfig) -> list[QuestionProbe]:
    system = build(system_name, config)
    questions = load(limit=limit, strategy=SAMPLE_STRATIFIED, seed=DEFAULT_SAMPLE_SEED)
    rows: list[QuestionProbe] = []
    started = time.time()
    for i, q in enumerate(questions, start=1):
        rows.append(probe_question(system, q))
        write_rows(out, rows)  # inspectable while it runs
        print(
            f"[{i}/{len(questions)}] {q.question_id} ranks={rows[-1].evidence_ranks} "
            f"drops={len(rows[-1].drops)}  {time.time() - started:.0f}s",
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
    parser.add_argument("--dedup-scope", choices=["store", "session"], default="store")
    parser.add_argument("--query-instruction", default="", help="'' | bge | a literal")
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args(argv)
    instruction = args.query_instruction
    if instruction == "bge":
        from mnimi.embeddings import BGE_QUERY_INSTRUCTION

        instruction = BGE_QUERY_INSTRUCTION
    config = MemoryConfig(
        dedup_cosine_threshold=args.dedup_threshold,
        dedup_scope=args.dedup_scope,
        query_instruction=instruction,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )
    run(args.system, args.limit, Path(args.out), config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
