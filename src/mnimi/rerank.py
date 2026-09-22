"""Cross-encoder rerank over the top-``pool`` rounds (PHASE6 D5, L2).

37 of the 44 evidence rounds the n=500 probe left outside the top-10 sat inside
the top-50: an ordering problem inside the candidate pool, which is the
trigger ``docs/FUTURE.md`` set for this reranker. Admitted by the maintainer's
ruling of 2026-09-22: a 22M-parameter sentence-pair *classifier* with no
generation is not the LLM ``CLAUDE.md`` keeps off the read path.

Under ``ranking="rerank"`` the read path first takes the ``rerank_pool`` best
rounds exactly as ``ranking="score"`` would (so the adopted Phase 4 path is the
candidate generator and this is one variable against it), then scores every
record of every candidate round against the bare question with the
cross-encoder — a round record by its verbatim turns, a fact record by its
fact — and hands back the ``k`` rounds with the highest best-record score.
Everything the reader sees is unchanged in kind: the same rounds' turns and
facts, through the one renderer.

The model is injected like the embedder and the extractor (``Memory(...,
reranker=)``): the core stays two dependencies, the pinned model rides the
``[embed]`` extra (``onnxruntime``, ``tokenizers``, ``huggingface_hub`` — no
new dependency), and CI uses :class:`OverlapReranker`, a deterministic
token-overlap stand-in. The model, its HF revision and the pool size are
harness pins; nothing stored changes, so there is no guard row.
"""

from __future__ import annotations

from typing import Protocol

from .models import MemoryRecord, ScoredRecord
from .ranking import rank_rounds, round_identity

RERANK_POOL_DEFAULT = 50


class Reranker(Protocol):
    """A pair scorer: higher means the passage answers the question better."""

    name: str
    revision: str

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def record_text(record: MemoryRecord) -> str:
    """What the cross-encoder reads for one record (D5): a round's turns, a fact's fact."""
    if record.kind == "fact" and record.fact:
        return record.fact
    if record.turns:
        return "\n".join(
            f"{turn.get('role', '')}: {turn.get('content', '')}" for turn in record.turns
        )
    return record.content


class OverlapReranker:
    """CI's model-free stand-in: the share of the question's tokens present in the passage.

    Deterministic and dependency-free; ties broken by the caller's stable
    sort. Never a benchmark configuration.
    """

    name = "overlap"
    revision = "v1"

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for question, passage in pairs:
            q = {t for t in question.lower().split() if len(t) > 2}
            p = set(passage.lower().split())
            out.append(len(q & p) / len(q) if q else 0.0)
        return out


class MiniLMCrossEncoder:
    """``cross-encoder/ms-marco-MiniLM-L-6-v2`` via ONNX Runtime, revision-pinned (D5).

    The repo's own ``onnx/model.onnx`` (fp32, 91 MB) and ``tokenizer.json`` at
    the pinned commit; CPU provider, so the scores are the same bytes on any
    machine with the same ONNX Runtime — the stability gate (two fresh
    processes, byte-identical rank lists) is what admits it to a probe. Pairs
    are truncated to the model's 512 tokens, the question first.
    """

    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    revision = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    _MAX_TOKENS = 512
    _BATCH = 32

    def __init__(self) -> None:
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "MiniLMCrossEncoder needs the [embed] extra: pip install mnimi[embed]"
            ) from exc
        import numpy as np

        self._np = np
        model_path = hf_hub_download(self.name, "onnx/model.onnx", revision=self.revision)
        tokenizer_path = hf_hub_download(self.name, "tokenizer.json", revision=self.revision)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {node.name for node in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=self._MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        np = self._np
        scores: list[float] = []
        for start in range(0, len(pairs), self._BATCH):
            batch = pairs[start:start + self._BATCH]
            encodings = self._tokenizer.encode_batch([(q, p) for q, p in batch])
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            feed = {name: value for name, value in feed.items() if name in self._input_names}
            (logits,) = self._session.run([self._output_name], feed)
            scores.extend(float(x) for x in logits[:, 0])
        return scores


def rerank_rounds(
    store,
    query_embedding: list[float],
    question: str,
    user_id: str,
    k: int,
    *,
    pool: int,
    reranker: Reranker,
    weights,
    half_life_days: float,
    now: str | None,
    active_only: bool = False,
    window=None,
    time_weight: float = 0.0,
) -> list[ScoredRecord]:
    """The ``k`` best of the ``pool`` score-ranked rounds by cross-encoder score (D5).

    Candidates come from :func:`mnimi.ranking.rank_rounds` — the adopted read
    path, with its time-aware term when one is set — so a reranked run differs
    from a scored run by the reranker alone. Each candidate round is scored by
    its best record (its round record and its active facts); ties keep the
    candidate order, so equal scores fall back to the score ranking.
    ``ScoredRecord.score`` carries the cross-encoder logit; ``relevance`` the
    cosine of the record that ranked the round.
    """
    candidates = rank_rounds(
        store, query_embedding, user_id, pool, weights=weights,
        half_life_days=half_life_days, now=now, active_only=active_only,
        window=window, time_weight=time_weight,
    )
    if not candidates:
        return []
    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for index, hit in enumerate(candidates):
        record = hit.record
        members = [record]
        if record.round_key is not None:
            members.extend(
                f for f in store.facts_of(user_id, record.round_key, active_only=active_only)
                if f.id != record.id
            )
            if record.kind == "fact":
                # The round record itself, when a fact ranked the round.
                members.extend(
                    r for r in store.round_records(user_id, record.round_key)
                    if r.id != record.id
                )
        for member in members:
            pairs.append((question, record_text(member)))
            owners.append(index)
    scores = reranker.score(pairs)
    best: dict[int, float] = {}
    for owner, score in zip(owners, scores, strict=True):
        if owner not in best or score > best[owner]:
            best[owner] = score
    order = sorted(range(len(candidates)), key=lambda i: (-best.get(i, float("-inf")), i))
    out = []
    for i in order[:k]:
        hit = candidates[i]
        out.append(
            ScoredRecord(
                record=hit.record, relevance=hit.relevance, recency=hit.recency,
                salience=hit.salience, score=best.get(i, float("-inf")),
                time_match=hit.time_match,
            )
        )
    return out


def round_key_of(record: MemoryRecord):
    return round_identity(record)
