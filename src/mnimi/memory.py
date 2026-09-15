"""The ``Memory`` facade: four methods over an embedder, a store and, optionally, an extractor.

Today these are deliberately thin. The store is real (real schema, real vector
search); the write-side intelligence — extraction (PHASE2), salience, conflict
resolution, decay — lands behind ``add`` and ``consolidate``. The public
surface stays at four methods regardless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from string import Template
from typing import NamedTuple

from .config import MemoryConfig
from .conflict.lexicon import negation_lexicon_hash
from .conflict.normalize import conflict_rules_hash, normalize_triple
from .conflict.screens import ACTION_KEEP, Verdict, screen_pair
from .embeddings import Embedder
from .extract import prefilter
from .extract.resolver import RESOLVER_VERSION, resolve, verbatim_mention
from .models import KIND_FACT, KIND_ROUND, MemoryRecord
from .store import Store

log = logging.getLogger("mnimi.memory")

_PUNCT_RE = re.compile(r"[^\w\s]")

_DEFAULT_CONFIG = MemoryConfig()

DEDUP_SCOPE_STORE = "store"
DEDUP_SCOPE_SESSION = "session"
DEDUP_SCOPES = (DEDUP_SCOPE_STORE, DEDUP_SCOPE_SESSION)

# ---------------------------------------------------------------------------
# The embed/render split.
#
# EMBED_TEMPLATE builds the text that gets embedded and keys the dedup screens
# for a ROUND record. It is FROZEN: an edit here changes every vector, every
# retrieval and the dedup key, and invalidates any threshold selected under
# the previous form — the 0.95 threshold's selection evidence cannot be
# regenerated (further threshold selection against LongMemEval is
# prohibited). Its hash is pinned into ``memory_meta`` at DB creation and
# checked on every open.
#
# FACT_EMBED_TEMPLATE builds the embedded text of a FACT record (PHASE2 D2):
# SPEC's extraction-era ``f"{raw}\n{content}"`` — the verbatim role-prefixed
# span, then the fact text (LongMemEval's measured key expansion, CHANGELOG
# #1). Frozen and hashed into ``memory_meta`` under its own key, so the v1
# template's hash does not move when the extraction era lands.
#
# RENDER_TEMPLATE builds the text a reader sees. It may evolve — an edit here
# changes reader context and not a single vector — and its hash is pinned in
# the harness artifact header so a format change is loud there instead.
#
# Two render FORMATS share one code path (2026-09-12, the gpt-4o era's
# presentation pair — DECISIONS "Pre-registration for the gpt-4o era"): the
# same blocks (one per timestamp change, the turns inside it), framed either
# as the labelled text below or as a JSON array (LongMemEval §5.5). Only the
# framing differs; the information content is identical by construction. Each
# format has its own template string and therefore its own hash, and the
# harness pins the hash of the format it ran — so the two are never confused
# in an artifact, and the default format's hash is exactly what it was before
# the second format existed.
#
# Three render UNITS (PHASE2 D5) decide what a retrieved ROUND is shown as:
# its verbatim turns (``turns`` — v1, and the format the other arms render),
# its turns under a ``facts:`` header (``round+facts``), or its facts alone
# (``facts``). The unit is a system-level pin (``render_unit_template_hash``
# in mnimi's retrieval pins), not a harness parity field: the FORMAT stays
# identical across arms, the unit is what the memory layer hands the reader.
# ---------------------------------------------------------------------------

EMBED_TEMPLATE = "[Session date: ${date}] ${text}"
_EMBED_T = Template(EMBED_TEMPLATE)

FACT_EMBED_TEMPLATE = "${raw}\n${fact}"
_FACT_EMBED_T = Template(FACT_EMBED_TEMPLATE)

# One header per timestamp change, then one role-labelled line per turn — the
# canonical context format shared by every arm of the eval (the harness's
# full_history baseline calls :func:`render_turns` directly).
RENDER_TEMPLATE = "[Session date: ${ts}]\n${role}: ${content}"
_RENDER_HEADER_T, _RENDER_TURN_T = (Template(part) for part in RENDER_TEMPLATE.split("\n"))

# The JSON framing of the same blocks. The string is a descriptor of the shape
# (it is what gets hashed), not a template that is substituted into: the
# rendered output is ``json.dumps`` of ``[{"session_date", "turns": [{"role",
# "content"}]}]`` with ``ensure_ascii=False`` and a two-space indent.
RENDER_JSON_TEMPLATE = (
    '[{"session_date": "${ts}", "turns": [{"role": "${role}", "content": "${content}"}]}]'
)
_RENDER_JSON_INDENT = 2

RENDER_FORMAT_TEXT = "text"
RENDER_FORMAT_JSON = "json"
RENDER_FORMATS = (RENDER_FORMAT_TEXT, RENDER_FORMAT_JSON)
_RENDER_TEMPLATES = {
    RENDER_FORMAT_TEXT: RENDER_TEMPLATE,
    RENDER_FORMAT_JSON: RENDER_JSON_TEMPLATE,
}

RENDER_UNIT_TURNS = "turns"
RENDER_UNIT_ROUND_FACTS = "round+facts"
RENDER_UNIT_FACTS = "facts"
RENDER_UNITS = (RENDER_UNIT_TURNS, RENDER_UNIT_ROUND_FACTS, RENDER_UNIT_FACTS)
# The facts header a round renders under (``round+facts``), and the per-fact
# block a round renders as (``facts``); ``(${valid_time})`` is omitted for a
# standing fact. These strings are the hashed descriptors of each unit.
RENDER_FACTS_HEADER_TEMPLATE = "facts:\n- ${fact} (${valid_time})"
RENDER_FACT_ITEM_TEMPLATE = "fact: ${fact} (${valid_time})\nsource: ${raw}"
_RENDER_UNIT_TEMPLATES = {
    RENDER_UNIT_TURNS: "${turns}",
    RENDER_UNIT_ROUND_FACTS: RENDER_FACTS_HEADER_TEMPLATE + "\n${turns}",
    RENDER_UNIT_FACTS: RENDER_FACT_ITEM_TEMPLATE,
}
_FACT_LINE_T = Template("- ${fact}")
_FACT_ITEM_T = Template("fact: ${fact}")
_FACT_SOURCE_T = Template("source: ${raw}")


def _check_render_format(fmt: str) -> str:
    if fmt not in _RENDER_TEMPLATES:
        raise ValueError(f"unknown render format {fmt!r}; expected one of {RENDER_FORMATS}")
    return fmt


def _check_render_unit(unit: str) -> str:
    if unit not in _RENDER_UNIT_TEMPLATES:
        raise ValueError(f"unknown render unit {unit!r}; expected one of {RENDER_UNITS}")
    return unit


def embed_template_hash() -> str:
    """Digest of the round embed-text template, pinned in ``memory_meta``."""
    return hashlib.sha256(EMBED_TEMPLATE.encode("utf-8")).hexdigest()


def fact_embed_template_hash() -> str:
    """Digest of the fact embed-text template, pinned in ``memory_meta``."""
    return hashlib.sha256(FACT_EMBED_TEMPLATE.encode("utf-8")).hexdigest()


def render_template_hash(fmt: str = RENDER_FORMAT_TEXT) -> str:
    """Digest of the render template for ``fmt``, pinned in the harness header.

    The default (text) digest is unchanged by the JSON format's existence: it
    hashes the same ``RENDER_TEMPLATE`` string it always did.
    """
    return hashlib.sha256(_RENDER_TEMPLATES[_check_render_format(fmt)].encode("utf-8")).hexdigest()


def render_unit_template_hash(unit: str = RENDER_UNIT_TURNS) -> str:
    """Digest of the render unit's block descriptor — a system-level pin."""
    return hashlib.sha256(
        _RENDER_UNIT_TEMPLATES[_check_render_unit(unit)].encode("utf-8")
    ).hexdigest()


def guard_kwargs(extractor=None) -> dict:
    """The extraction-era ``memory_meta`` rows a ``Store`` is opened with.

    One definition for every constructor site (``Memory``, the harness's
    ``naive_rag`` arm, the tests), so the guard cannot drift between them.
    """
    return {
        "extractor_pins": extractor.pins if extractor is not None else None,
        "fact_embed_template_hash": fact_embed_template_hash(),
        "prefilter_lexicon_hash": prefilter.prefilter_lexicon_hash(),
        "resolver_version": RESOLVER_VERSION,
        # Phase 3 (D12): the frozen negation lexicon and the normalization /
        # conflict rules — both decide which facts stay active.
        "negation_lexicon_hash": negation_lexicon_hash(),
        "conflict_rules_hash": conflict_rules_hash(),
    }


def _render_blocks(turns: list[dict]) -> list[dict]:
    """Group turns into ``{"session_date", "turns"}`` blocks, one per ``ts`` change.

    This is the one grouping both formats render — the text format emits a
    header where a block starts, the JSON format emits an object — so the two
    framings can never disagree about where a session boundary falls.
    """
    blocks: list[dict] = []
    current_ts = None
    for turn in turns:
        ts = turn.get("ts")
        if (ts and ts != current_ts) or not blocks:
            blocks.append({"session_date": ts, "turns": []})
            if ts:
                current_ts = ts
        blocks[-1]["turns"].append(
            {"role": turn.get("role", ""), "content": turn.get("content", "")}
        )
    return blocks


def render_turns(turns: list[dict], fmt: str = RENDER_FORMAT_TEXT) -> str:
    """Render ``{"role", "content", "ts"}`` turns into reader context.

    A dated header is emitted whenever ``ts`` changes, so the session boundary
    and its full timestamp stay reader-visible (temporal questions are
    unanswerable without them) at one header per block rather than one
    timestamp per turn. Every turn carries its speaker label — the dataset's
    single-session-assistant questions ask about assistant turns, which are
    unattributable without one.

    ``fmt`` selects the framing: ``"text"`` (the default, byte-identical to
    the pre-split renderer) or ``"json"`` (the same blocks as a JSON array).
    """
    _check_render_format(fmt)
    if fmt == RENDER_FORMAT_JSON:
        return json.dumps(_render_blocks(turns), ensure_ascii=False, indent=_RENDER_JSON_INDENT)
    lines: list[str] = []
    current_ts = None
    for turn in turns:
        ts = turn.get("ts")
        if ts and ts != current_ts:
            lines.append(_RENDER_HEADER_T.substitute(ts=ts))
            current_ts = ts
        lines.append(
            _RENDER_TURN_T.substitute(
                role=turn.get("role", ""), content=turn.get("content", "")
            )
        )
    return "\n".join(lines)


def _fact_text(record: MemoryRecord) -> str:
    text = _FACT_LINE_T.substitute(fact=record.fact or "")[2:]  # the bare fact text
    return f"{text} ({record.valid_time})" if record.valid_time else text


def _render_items(records: list[MemoryRecord], facts_of) -> list[dict]:
    """One item per retrieved round: its ``ts``, its turns, its fact records."""
    items: list[dict] = []
    rendered_rounds: set[str] = set()
    for record in records:
        if record.round_key is not None:
            if record.round_key in rendered_rounds:
                continue
            rendered_rounds.add(record.round_key)
        if record.round_key is not None and facts_of is not None:
            facts = list(facts_of(record.round_key))
        elif record.kind == KIND_FACT:
            facts = [record]
        else:
            facts = []
        turns = record.turns or [{"role": record.source, "content": record.content}]
        items.append({"ts": record.created_at, "turns": turns, "facts": facts})
    return items


def render_records(
    records: list[MemoryRecord],
    fmt: str = RENDER_FORMAT_TEXT,
    unit: str = RENDER_UNIT_TURNS,
    facts_of=None,
) -> str:
    """Render stored records (already ordered) through the one renderer.

    ``unit="turns"`` flattens each record's verbatim turns, stamping the
    record's ``created_at`` onto every turn so the header logic sees the same
    shape ``full_history`` feeds it — byte for byte the v1 output. A legacy
    record without ``turns`` falls back to its ``content`` string — degraded
    (no speaker attribution) but never silently dropped. A round stored as
    several records (R4 windows, or a round record with its facts) renders
    once, at its first record.

    ``unit="round+facts"`` renders the same blocks with a ``facts:`` header
    per round listing every fact record of that round — the ones retrieved
    and, through ``facts_of(round_key)``, the ones that were not — each with
    its resolved ``valid_time`` when it has one. ``unit="facts"`` renders a
    round as its fact records only (``fact:`` / ``source:`` lines); a round
    with no facts falls back to its turns.
    """
    _check_render_format(fmt)
    _check_render_unit(unit)
    if unit == RENDER_UNIT_TURNS:
        turns: list[dict] = []
        rendered_rounds: set[str] = set()
        for record in records:
            if record.round_key is not None:
                if record.round_key in rendered_rounds:
                    continue
                rendered_rounds.add(record.round_key)
            source_turns = record.turns or [{"role": record.source, "content": record.content}]
            for turn in source_turns:
                turns.append(
                    {
                        "role": turn.get("role", ""),
                        "content": turn.get("content", ""),
                        "ts": record.created_at,
                    }
                )
        return render_turns(turns, fmt=fmt)

    items = _render_items(records, facts_of)
    if fmt == RENDER_FORMAT_JSON:
        blocks: list[dict] = []
        current_ts = None
        for item in items:
            ts = item["ts"]
            if (ts and ts != current_ts) or not blocks:
                blocks.append({"session_date": ts, "items": []})
                if ts:
                    current_ts = ts
            entry: dict = {
                "facts": [
                    {"fact": f.fact, "valid_time": f.valid_time, "source": f.raw}
                    for f in item["facts"]
                ]
            }
            if unit == RENDER_UNIT_ROUND_FACTS or not item["facts"]:
                entry["turns"] = [
                    {"role": t.get("role", ""), "content": t.get("content", "")}
                    for t in item["turns"]
                ]
            blocks[-1]["items"].append(entry)
        return json.dumps(blocks, ensure_ascii=False, indent=_RENDER_JSON_INDENT)

    lines: list[str] = []
    current_ts = None
    for item in items:
        ts = item["ts"]
        if ts and ts != current_ts:
            lines.append(_RENDER_HEADER_T.substitute(ts=ts))
            current_ts = ts
        if unit == RENDER_UNIT_ROUND_FACTS:
            if item["facts"]:
                lines.append("facts:")
                lines.extend(_FACT_LINE_T.substitute(fact=_fact_text(f)) for f in item["facts"])
            for turn in item["turns"]:
                lines.append(_RENDER_TURN_T.substitute(role=turn.get("role", ""),
                                                       content=turn.get("content", "")))
        else:  # RENDER_UNIT_FACTS
            if item["facts"]:
                for f in item["facts"]:
                    lines.append(_FACT_ITEM_T.substitute(fact=_fact_text(f)))
                    lines.append(_FACT_SOURCE_T.substitute(raw=f.raw or ""))
            else:
                for turn in item["turns"]:
                    lines.append(_RENDER_TURN_T.substitute(role=turn.get("role", ""),
                                                           content=turn.get("content", "")))
    return "\n".join(lines)


class Memory:
    """Embeddable agent memory backed by a single SQLite file."""

    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        config: MemoryConfig = _DEFAULT_CONFIG,
        *,
        extractor=None,
    ) -> None:
        """``extractor`` is the one LLM (PHASE2 D6): an ``mnimi.extract.Extractor``
        whose facts become fact records beside every round. ``None`` — the
        default, and all the core deps can offer — is the v1 write path: rounds
        only, no fact records, no ``[extract]`` extra."""
        self.embedder = embedder
        self.config = config
        self.extractor = extractor
        if config.dedup_scope not in DEDUP_SCOPES:
            raise ValueError(
                f"unknown dedup_scope {config.dedup_scope!r}; expected one of {DEDUP_SCOPES}"
            )
        if config.chunk_tokens < 0 or (
            config.chunk_tokens and not 0 <= config.chunk_overlap < config.chunk_tokens
        ):
            raise ValueError(
                "chunk_tokens must be >= 0 and chunk_overlap must be in [0, chunk_tokens)"
            )
        _check_render_format(config.render_format)
        _check_render_unit(config.render_unit)
        self.store = Store(
            db_path,
            dim=embedder.dim,
            embedder_name=embedder.name,
            embedder_revision=embedder.revision,
            embed_template_hash=embed_template_hash(),
            chunk_tokens=config.chunk_tokens,
            chunk_overlap=config.chunk_overlap,
            **guard_kwargs(extractor),
        )
        # Diagnostics of the extraction stage, accumulated across add() calls
        # (the corpus pass reports them). Never a pin, never persisted.
        self.extraction_stats = new_extraction_stats()
        # The same for the Phase 3 stage: pairs screened and kept, conflicts
        # found and facts superseded (the probe reports them by category).
        self.conflict_stats = new_conflict_stats()

    def add(self, messages, user_id: str) -> None:
        """Write path: one round record per user+assistant round, plus one fact
        record per extracted fact when an extractor is present, all deduped.

        ``messages`` is ``list[dict]`` with ``role`` / ``content`` / ``ts`` —
        nothing else. Granularity is per-round (a user turn and its assistant
        reply), which must match ``naive_rag`` exactly or the comparison is
        confounded. The session date is folded into the round's embed text so
        it reaches the reader; role stays metadata and never enters a round's
        vector (a fact's ``raw`` span is role-prefixed by SPEC).

        Dedup is exact-normalize collapse followed by ONE cosine-threshold
        probe, per record, against records of the SAME kind (PHASE2 D3): a
        fact is compared with earlier facts, a round with earlier rounds. For
        rounds the exact screen is store-wide (its key folds the session date
        in); for facts both screens follow ``dedup_scope``. A fact pair the
        cosine probe flags is then read through SPEC's screens — negation,
        value substitution, the entropy gate (PHASE3, ``fact_verdict``) — and
        is dropped only when all three call it a duplicate; with
        ``config.conflict_resolution`` off, or for a round, the flag alone
        drops it (the v1.9 path).
        """
        rounds = _messages_to_rounds(messages)
        if not rounds:
            return
        per_round = [
            (round_, round_pieces(round_, self.embedder, self.config, self.extractor,
                                  self.extraction_stats))
            for round_ in rounds
        ]
        # Two embed batches, never one: the round batch is byte-identical to
        # v1's (batch composition moves BGE's vectors), and the fact batch
        # rides behind it.
        positions = {KIND_ROUND: [], KIND_FACT: []}
        for ri, (_round, pieces) in enumerate(per_round):
            for pi, piece in enumerate(pieces):
                positions[piece.kind].append((ri, pi))
        vectors: dict[tuple[int, int], list[float]] = {}
        for spots in positions.values():
            if not spots:
                continue
            texts = [per_round[ri][1][pi].content for ri, pi in spots]
            for spot, vector in zip(spots, self.embedder.embed(texts), strict=True):
                vectors[spot] = vector

        seen_round = {_normalize(c) for c in self.store.contents(user_id, kind=KIND_ROUND)}
        seen_fact = {
            self._fact_exact_key(ts, content)
            for ts, content in self.store.contents_with_ts(user_id, kind=KIND_FACT)
        }
        threshold = self.config.dedup_cosine_threshold
        for ri, (round_, pieces) in enumerate(per_round):
            for pi, piece in enumerate(pieces):
                embedding = vectors[(ri, pi)]
                if piece.kind == KIND_ROUND:
                    key = _normalize(piece.content)
                    seen = seen_round
                else:
                    key = self._fact_exact_key(round_.ts, piece.content)
                    seen = seen_fact
                if key in seen:
                    continue
                hits = self.store.search(embedding, user_id=user_id, k=1, kind=piece.kind)
                if (
                    hits
                    and hits[0][1] >= threshold
                    and self._in_dedup_scope(hits[0][0], round_)
                ):
                    verdict = fact_verdict(piece, hits[0][0], self.config)
                    if verdict is None:
                        continue  # a round, or the stage is off: the v1.9 drop
                    self.conflict_stats["pairs_screened"] += 1
                    if verdict.action != ACTION_KEEP:
                        self.conflict_stats["merged"] += 1
                        continue
                    self.conflict_stats[f"kept_{verdict.reason.replace('-', '_')}"] += 1
                    log.debug("kept: %s", verdict.reason)
                self.store.insert(
                    MemoryRecord(
                        user_id=user_id,
                        content=piece.content,
                        embedding=embedding,
                        created_at=round_.ts,
                        salience=piece.salience,
                        source=round_.roles,
                        turns=round_.turns,
                        round_key=piece.round_key,
                        kind=piece.kind,
                        fact=piece.fact,
                        raw=piece.raw,
                        subject=piece.subject,
                        predicate=piece.predicate,
                        object=piece.object,
                        valid_time=piece.valid_time,
                        time_mention=piece.time_mention,
                        pair_key=piece.pair_key,
                    )
                )
                seen.add(key)

    def _fact_exact_key(self, ts: str | None, content: str):
        """The exact screen's key for a fact record: store-wide or per session.

        A fact's embed text has no date fold (``FACT_EMBED_TEMPLATE``), so the
        session scope has to be applied here explicitly; a round's key already
        carries the date.
        """
        normalized = _normalize(content)
        if self.config.dedup_scope == DEDUP_SCOPE_STORE:
            return normalized
        return (ts, normalized)

    def _in_dedup_scope(self, neighbour: MemoryRecord, round_: Round) -> bool:
        """Whether a neighbour at or above the threshold counts as a duplicate.

        ``store`` scope: always. ``session`` scope: only when the neighbour
        carries the round's own ``ts`` — the library has no session id, and
        the session timestamp is the one clock every record carries.
        """
        if self.config.dedup_scope == DEDUP_SCOPE_STORE:
            return True
        return neighbour.created_at == round_.ts

    def _query_embedding(self, query: str) -> list[float]:
        """The vector a query is searched with.

        One place, so the retrieval probe (``evals/probes``) and the read path
        cannot drift apart; a query-side change (the BGE instruction prefix,
        R5) edits only this method.
        """
        (embedding,) = self.embedder.embed([self.config.query_instruction + query])
        return embedding

    def recall(self, query: str, user_id: str) -> list[MemoryRecord]:
        """Raw retrieval: the nearest stored rounds, no assembly.

        ``k`` counts rounds: a round's records (R4 windows; the round record
        and its facts) collapse to the best-ranked one (``Store.search_rounds``).
        """
        query_embedding = self._query_embedding(query)
        hits = self.store.search_rounds(query_embedding, user_id=user_id, k=self.config.top_k)
        return [record for record, _cosine in hits]

    def get_context(self, query: str, user_id: str) -> str:
        """Assemble recalled memories into a single context string for a reader.

        Ordered oldest-first, not by relevance. Retrieval order is a similarity
        ranking; handing a reader dated blocks in ranking order asks it to
        reconstruct a chronology from a shuffle, and temporal questions are 27%
        of the benchmark. The paper's own pipeline sorts retrieved items by
        timestamp before reading (§5.1).

        Rendered from each record's verbatim ``turns`` (and, under the
        extraction era's units, its round's fact records) — full timestamp
        header, ``user:``/``assistant:`` speaker labels — never from
        ``content``, which is the embed text and stays frozen when this format
        evolves.
        """
        records = _time_ordered(self.recall(query, user_id))
        return render_records(
            records,
            fmt=self.config.render_format,
            unit=self.config.render_unit,
            facts_of=lambda round_key: self.store.facts_of(user_id, round_key),
        )

    def consolidate(self, user_id: str) -> None:
        """Merge duplicates, resolve conflicts, decay stale memories.

        Stub today — not exercised by the Day 1 baselines. This is where the
        write-side policy will live.
        """
        return None


def new_extraction_stats() -> dict:
    """Counters ``round_pieces`` accumulates when an extractor is in play."""
    return {
        "rounds": 0,
        "prefilter_skips": 0,
        "rounds_sent": 0,
        "empty_extractions": 0,
        "truncated_outputs": 0,
        "truncated_inputs": 0,
        "facts": 0,
    }


def new_conflict_stats() -> dict:
    """Counters of the Phase 3 stage: the screens (Task 2) and supersession (Task 3)."""
    return {
        "pairs_screened": 0,
        "merged": 0,
        "kept_negation": 0,
        "kept_value": 0,
        "kept_low_entropy": 0,
        "candidates": 0,
        "conflicts_negation": 0,
        "conflicts_functional": 0,
        "conflicts_numeric": 0,
        "superseded": 0,
    }


def fact_verdict(piece: Piece, neighbour: MemoryRecord, config: MemoryConfig) -> Verdict | None:
    """What the screens say about a cosine-gate-pass pair, or ``None`` when they do not apply.

    One function for the write path and the retrieval probe's mirror walk, so
    the two cannot disagree. ``None`` — a round record, or
    ``config.conflict_resolution`` off — is the v1.9 outcome: the probe's flag
    alone drops the incoming piece.
    """
    if piece.kind != KIND_FACT or not config.conflict_resolution:
        return None
    return screen_pair(
        piece.fact or "",
        (piece.subject, piece.predicate, piece.object),
        neighbour.fact or "",
        (neighbour.subject, neighbour.predicate, neighbour.object),
        config.dedup_entropy_gate,
    )


def _time_ordered(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Oldest first, ties broken by insertion order.

    Sorts on the ``created_at`` string directly: session timestamps are
    zero-padded date-first (``2023-05-20``, ``2023/05/20``), so lexicographic
    order is chronological order without parsing — and parsing would mean
    guessing a format the caller owns. The ``id`` tiebreak keeps two records
    from one session in a deterministic order.
    """
    return sorted(records, key=lambda r: (r.created_at or "", r.id or 0))


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the exact-dup key."""
    return " ".join(_PUNCT_RE.sub("", text.lower()).split())


class Round(NamedTuple):
    """One ingestion unit: the embed text plus everything rendering needs."""

    content: str  # the EMBED text — frozen form, keys dedup, becomes the vector
    roles: str  # provenance summary, e.g. "user+assistant"
    ts: str | None  # the round's session timestamp, verbatim
    turns: list[dict]  # verbatim {"role", "content"} turns, for rendering only


class Piece(NamedTuple):
    """One embedded string of a round: the round whole, one R4 window of it, or one fact."""

    content: str  # the embed text of this record
    round_key: str | None  # shared by every piece of one round; None for a bare v1 round
    kind: str = KIND_ROUND
    fact: str | None = None
    raw: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    valid_time: str | None = None
    time_mention: str | None = None
    salience: float = 1.0
    pair_key: str | None = None  # the normalized subject|predicate of a fact's triple


def _round_text(round_: Round) -> str:
    """The un-templated turn text a round's embed content was built from."""
    return "\n".join(str(t.get("content", "")).strip() for t in round_.turns)


def round_key(ts: str | None, text: str) -> str:
    """Identity of a round across its records: its timestamp and its text."""
    return f"{ts}|{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def round_pieces(
    round_: Round,
    embedder: Embedder,
    config: MemoryConfig,
    extractor=None,
    stats: dict | None = None,
) -> list[Piece]:
    """What gets embedded for one round under ``config`` (R4) and ``extractor`` (PHASE2).

    ``chunk_tokens == 0``: the round whole, ``round_key`` None — v1's shape.
    Otherwise the round's text is split by the embedder's own tokenizer into
    overlapping windows and each window is templated like a whole round (the
    date fold is on every window, so ``EMBED_TEMPLATE`` is unchanged); a
    round that fits in one window is still whole and unkeyed. Both retrieval
    arms of the eval call this, so their embedded units stay identical.

    With an ``extractor`` (mnimi only): every piece of the round carries its
    ``round_key``; if the stage-1 pre-filter keeps at least one turn, the
    round is handed to the extractor and each fact becomes a ``fact`` piece —
    ``FACT_EMBED_TEMPLATE`` text, the fact's fields, and ``valid_time``
    resolved from the verbatim mention against the round's ``ts``. ``stats``,
    when given, accumulates the stage's counters (the corpus pass reports
    them).
    """
    if config.chunk_tokens <= 0:
        pieces = [Piece(round_.content, None)]
    else:
        text = _round_text(round_)
        windows = embedder.split(text, config.chunk_tokens, config.chunk_overlap)
        if len(windows) <= 1:
            pieces = [Piece(round_.content, None)]
        else:
            key = round_key(round_.ts, text)
            pieces = [
                Piece(_EMBED_T.substitute(date=_date_of(round_.ts), text=w) if round_.ts else w,
                      key)
                for w in windows
            ]
    if extractor is None:
        return pieces
    key = round_key(round_.ts, _round_text(round_))
    pieces = [piece._replace(round_key=key) for piece in pieces]
    if stats is not None:
        stats["rounds"] += 1
    keep, _dropped = prefilter.keep_round(round_.turns)
    if not keep:
        if stats is not None:
            stats["prefilter_skips"] += 1
        return pieces
    result = extractor.extract(round_.turns)
    facts = [f for f in result.facts if f.raw and f.content]
    if stats is not None:
        stats["rounds_sent"] += 1
        stats["empty_extractions"] += not facts
        stats["truncated_outputs"] += bool(result.truncated)
        stats["truncated_inputs"] += bool(getattr(result, "truncated_input", False))
        stats["facts"] += len(facts)
    round_text = _round_text(round_)
    for fact in facts:
        # A mention the round does not contain is an invention, not a date.
        mention = verbatim_mention(fact.when, round_text)
        triple = normalize_triple(fact.subject, fact.predicate, fact.object)
        pieces.append(
            Piece(
                content=_FACT_EMBED_T.substitute(raw=fact.raw, fact=fact.content),
                round_key=key,
                kind=KIND_FACT,
                fact=fact.content,
                raw=fact.raw,
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                valid_time=resolve(mention, round_.ts),
                time_mention=mention,
                salience=float(fact.salience),
                pair_key=triple.pair_key if triple is not None else None,
            )
        )
    return pieces


def _messages_to_rounds(messages) -> list[Round]:
    """Group ``{"role", "content", "ts"}`` dicts into per-round records.

    Each :class:`Round` pairs a user turn with the assistant reply that follows
    it, or stands a solo turn alone. ``content`` is the embed text: it carries
    the folded session DATE (not the full timestamp) and NO role labels — role
    is metadata, the embedded string is bare content (SPEC CHANGELOG #15) —
    and its byte form is frozen (see ``EMBED_TEMPLATE``). ``turns`` carries the
    verbatim turn texts for the render path. ``ts`` comes from the round's
    first turn and is the record's only clock.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"add() takes list[dict] messages, got {type(messages).__name__}"
        )
    turns: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(
                "add() takes list[dict] messages with 'role'/'content'/'ts', "
                f"got a list containing {type(message).__name__}"
            )
        if str(message.get("content", "")).strip():
            turns.append(message)

    rounds: list[Round] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        follower = turns[index + 1] if index + 1 < len(turns) else None
        if (
            turn.get("role") == "user"
            and follower is not None
            and follower.get("role") == "assistant"
        ):
            batch = [turn, follower]
        else:
            batch = [turn]
        index += len(batch)

        ts = batch[0].get("ts")
        text = "\n".join(str(t["content"]).strip() for t in batch)
        # Template.substitute never rescans substituted values, so `$` inside
        # message content is safe; the template itself is the hashed constant.
        content = _EMBED_T.substitute(date=_date_of(ts), text=text) if ts else text
        roles = "+".join(str(t.get("role")) for t in batch)
        round_turns = [
            {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
            for t in batch
        ]
        rounds.append(Round(content, roles, ts, round_turns))
    return rounds


def _date_of(ts: str) -> str:
    """The date part of an ISO timestamp; the fold is a date, not a time."""
    return ts.split("T")[0].split(" ")[0]
