"""The ``Memory`` facade: four methods over an embedder and a store.

Today these are deliberately thin. The store is real (real schema, real vector
search); the write-side intelligence — extraction, salience, conflict
resolution, decay — lands behind ``add`` and ``consolidate`` in later weeks. The
public surface stays at four methods regardless.
"""

from __future__ import annotations

import hashlib
import re
from string import Template
from typing import NamedTuple

from .config import MemoryConfig
from .embeddings import Embedder
from .models import MemoryRecord
from .store import Store

_PUNCT_RE = re.compile(r"[^\w\s]")

_DEFAULT_CONFIG = MemoryConfig()

# ---------------------------------------------------------------------------
# The embed/render split.
#
# EMBED_TEMPLATE builds the text that gets embedded and keys the dedup screens.
# It is FROZEN: an edit here changes every vector, every retrieval and the
# dedup key, and invalidates any threshold selected under the previous form —
# the 0.95 threshold's selection evidence cannot be regenerated (further
# threshold selection against LongMemEval is prohibited). Its hash is pinned
# into ``memory_meta`` at DB creation and checked on every open.
#
# RENDER_TEMPLATE builds the text a reader sees. It may evolve — an edit here
# changes reader context and not a single vector — and its hash is pinned in
# the harness artifact header so a format change is loud there instead.
# ---------------------------------------------------------------------------

EMBED_TEMPLATE = "[Session date: ${date}] ${text}"
_EMBED_T = Template(EMBED_TEMPLATE)

# One header per timestamp change, then one role-labelled line per turn — the
# canonical context format shared by every arm of the eval (the harness's
# full_history baseline calls :func:`render_turns` directly).
RENDER_TEMPLATE = "[Session date: ${ts}]\n${role}: ${content}"
_RENDER_HEADER_T, _RENDER_TURN_T = (Template(part) for part in RENDER_TEMPLATE.split("\n"))


def embed_template_hash() -> str:
    """Digest of the embed-text template, pinned in ``memory_meta``."""
    return hashlib.sha256(EMBED_TEMPLATE.encode("utf-8")).hexdigest()


def render_template_hash() -> str:
    """Digest of the render template, pinned in the harness artifact header."""
    return hashlib.sha256(RENDER_TEMPLATE.encode("utf-8")).hexdigest()


def render_turns(turns: list[dict]) -> str:
    """Render ``{"role", "content", "ts"}`` turns into reader context.

    A dated header is emitted whenever ``ts`` changes, so the session boundary
    and its full timestamp stay reader-visible (temporal questions are
    unanswerable without them) at one header per block rather than one
    timestamp per turn. Every turn carries its speaker label — the dataset's
    single-session-assistant questions ask about assistant turns, which are
    unattributable without one.
    """
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


def render_records(records: list[MemoryRecord]) -> str:
    """Render stored records (already ordered) through :func:`render_turns`.

    Flattens each record's verbatim turns, stamping the record's ``created_at``
    onto every turn so the header logic sees the same shape ``full_history``
    feeds it. A legacy record without ``turns`` falls back to its ``content``
    string — degraded (no speaker attribution) but never silently dropped.
    """
    turns: list[dict] = []
    for record in records:
        source_turns = record.turns or [{"role": record.source, "content": record.content}]
        for turn in source_turns:
            turns.append(
                {
                    "role": turn.get("role", ""),
                    "content": turn.get("content", ""),
                    "ts": record.created_at,
                }
            )
    return render_turns(turns)


class Memory:
    """Embeddable agent memory backed by a single SQLite file."""

    def __init__(
        self, db_path: str, embedder: Embedder, config: MemoryConfig = _DEFAULT_CONFIG
    ) -> None:
        self.embedder = embedder
        self.config = config
        self.store = Store(
            db_path,
            dim=embedder.dim,
            embedder_name=embedder.name,
            embedder_revision=embedder.revision,
            embed_template_hash=embed_template_hash(),
        )

    def add(self, messages, user_id: str) -> None:
        """Write path: one record per user+assistant round, deduped.

        ``messages`` is ``list[dict]`` with ``role`` / ``content`` / ``ts`` —
        nothing else. Granularity is per-round (a user turn and its assistant
        reply), which must match ``naive_rag`` exactly or the comparison is
        confounded. The session date is folded into content so it reaches the
        reader; role stays metadata and never enters the embedded string.

        v1 dedup is exact-normalize collapse followed by ONE cosine-threshold
        probe against the store. No negation screen, no entropy gate — those
        arrive with extraction, post-v1.
        """
        rounds = _messages_to_rounds(messages)
        if not rounds:
            return
        seen = {_normalize(content) for content in self.store.contents(user_id)}
        embeddings = self.embedder.embed([r.content for r in rounds])
        for round_, embedding in zip(rounds, embeddings, strict=True):
            normalized = _normalize(round_.content)
            if normalized in seen:
                continue
            hits = self.store.search(embedding, user_id=user_id, k=1)
            if hits and hits[0][1] >= self.config.dedup_cosine_threshold:
                continue
            self.store.insert(
                MemoryRecord(
                    user_id=user_id,
                    content=round_.content,
                    embedding=embedding,
                    created_at=round_.ts,
                    source=round_.roles,
                    turns=round_.turns,
                )
            )
            seen.add(normalized)

    def recall(self, query: str, user_id: str) -> list[MemoryRecord]:
        """Raw retrieval: the nearest stored memories, no assembly."""
        (query_embedding,) = self.embedder.embed([query])
        hits = self.store.search(query_embedding, user_id=user_id, k=self.config.top_k)
        return [record for record, _cosine in hits]

    def get_context(self, query: str, user_id: str) -> str:
        """Assemble recalled memories into a single context string for a reader.

        Ordered oldest-first, not by relevance. Retrieval order is a similarity
        ranking; handing a reader dated blocks in ranking order asks it to
        reconstruct a chronology from a shuffle, and temporal questions are 27%
        of the benchmark. The paper's own pipeline sorts retrieved items by
        timestamp before reading (§5.1).

        Rendered from each record's verbatim ``turns`` — full timestamp header,
        ``user:``/``assistant:`` speaker labels — never from ``content``, which
        is the embed text and stays frozen when this format evolves.
        """
        records = _time_ordered(self.recall(query, user_id))
        return render_records(records)

    def consolidate(self, user_id: str) -> None:
        """Merge duplicates, resolve conflicts, decay stale memories.

        Stub today — not exercised by the Day 1 baselines. This is where the
        write-side policy will live.
        """
        return None


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
