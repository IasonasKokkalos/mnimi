"""OMEGA's retrieval over per-round verbatim storage (PHASE8 Task 15, ruling R3).

OMEGA (``omega-memory==1.5.17``) has no raw-conversation ingestion. Its write path
takes records an agent curates (SPEC § Extraction). So this arm stores **exactly
naive_rag's units**: one record per round, whose content is the round's frozen embed
text (``mnimi.memory._messages_to_rounds``, the function naive_rag runs). The metadata
carries the session date twice. ``referenced_date`` is OMEGA's own field for an
explicit event time, in ISO form. ``session_date`` is the date verbatim, and ``round``
is the round's id.

The ingest follows the type-independent settings of OMEGA's own LongMemEval script
(``scripts/longmemeval_official.py``), when the config's ``ingest`` block asks for them:
- ``store(..., skip_inference=True)``: no write-time contradiction check (it runs the
  cross-encoder against the ten nearest records on every store, about 80 % of the
  ingest time measured here);
- ``created_at`` rewritten to the session's date, shifted so the question's date is the
  present, once the harness hands the question date in. OMEGA decays a record from
  ``created_at`` against the wall clock, so without it every record is minutes old.

The retrieval is OMEGA's own, at the library's defaults:
- vector search over its ONNX embedder, FTS5 full-text search and rank fusion;
- rule-based query decomposition;
- its cross-encoder reranker over the fused top 10, which reads each record's date;
- its decay, abstention thresholds and adaptive retry.
The script's query-side code is not the library's and is not used: the question-type
hints and boosts, the temporal-range inference and the query variants.

The rounds a query returns render as their verbatim turns through the one renderer,
the bytes naive_rag shows for the same round, fitted to the context budget. So the arm
reads as "naive_rag's units under OMEGA's retrieval stack".

Pinned by the adapter before OMEGA is imported, since OMEGA reads its environment then:
- ``OMEGA_HOME`` points at a scratch directory, so no side file reaches ``~/.omega``;
- ``OMEGA_ONNX_MODEL_DIR`` points at the four files OMEGA's own setup installs for
  ``bge-small-en-v1.5``, taken from mnimi's pinned revision of the repo OMEGA downloads
  from. The arm refuses to run if OMEGA reports another model or its hash fallback,
  because OMEGA falls back silently otherwise;
- the reranker is the one OMEGA downloads on a fresh install,
  ``ms-marco-MiniLM-L-6-v2``. It is named explicitly, so a larger reranker left on disk
  cannot switch in. Its files in OMEGA's cache are verified by sha256 against the
  pinned snapshot, and auto-download is off;
- ``OMEGA_QUERY_EXPANSION=0``. OMEGA's optional query expansion calls an LLM (by
  default Anthropic's, whose SDK this environment lacks, so it returns nothing here).
  Turning it off makes that explicit, and keeps ``competitor_llm`` truthfully "none".

Disclosed, not patched:
- OMEGA ships LongMemEval question-type retrieval profiles, selected by ``query_hint``.
  The arm passes no hint, because a question-type label is benchmark metadata a
  deployed system never sees. OMEGA's ``_default`` profile applies to every question;
- OMEGA stamps ``created_at`` and access times from the wall clock.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .competitor import CompetitorSystem, Hit, epoch

# sqlite-vec's vec0 refuses a KNN ``k`` above 4096; OMEGA asks for ``limit * 5``
# neighbours and swallows the error, which would silently drop its vector channel.
VEC0_K_MAX = 4096
OMEGA_VEC_MULTIPLIER = 5


def _snapshot(repo: str, revision: str, patterns: list[str]) -> Path:
    """A local path to ``repo`` at ``revision``, holding at least ``patterns``."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lay_out(spec: dict, target: Path) -> Path:
    """Copy the spec's files from the revision-pinned snapshot into ``target``, verified."""
    snap = None
    for rel, name, digest in spec["files"]:
        dest = target / name
        if dest.exists() and _sha256(dest) == digest:
            continue
        if snap is None:
            snap = _snapshot(spec["repo"], spec["revision"], [f[0] for f in spec["files"]])
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snap / rel, dest)
        if _sha256(dest) != digest:
            raise RuntimeError(f"{dest} does not match {spec['repo']}@{spec['revision']}: "
                               f"expected sha256 {digest}")
    return target


def _model_dir(spec: dict, root: Path) -> str:
    """OMEGA's ONNX embedder directory, laid out from the revision-pinned snapshot."""
    return str(_lay_out(spec, root / "embedder"))


def _iso(ts: str | None) -> str | None:
    """The session date as ISO 8601 (UTC), the form OMEGA compares and truncates."""
    seconds = epoch(ts)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


class OmegaSystem(CompetitorSystem):
    name = "omega"

    def __init__(self, competitor_config: dict, render_format: str = "text") -> None:
        super().__init__(competitor_config, render_format)
        if self.candidates * OMEGA_VEC_MULTIPLIER > VEC0_K_MAX:
            raise ValueError(
                f"candidates={self.candidates}: OMEGA asks sqlite-vec for "
                f"{OMEGA_VEC_MULTIPLIER} x limit neighbours, over vec0's k ceiling of "
                f"{VEC0_K_MAX}, and swallows the error; keep candidates <= "
                f"{VEC0_K_MAX // OMEGA_VEC_MULTIPLIER}")
        self._scratch = tempfile.TemporaryDirectory(prefix="omega-eval-",
                                                    ignore_cleanup_errors=True)
        root = Path(self._scratch.name)
        os.environ["OMEGA_HOME"] = str(root / "home")
        os.environ["OMEGA_QUERY_EXPANSION"] = "0"
        emb = competitor_config.get("embedder")
        if emb:
            os.environ["OMEGA_ONNX_MODEL_DIR"] = _model_dir(emb, root)
        rer = competitor_config.get("reranker")
        if rer:
            _lay_out(rer, Path(os.path.expanduser(rer["omega_dir"])))
            os.environ["OMEGA_RERANKER_MODEL"] = rer["omega_name"]
            os.environ["OMEGA_RERANKER_AUTODOWNLOAD"] = "0"
        import omega
        from omega import embedding as omega_embedding
        from omega.sqlite_store import SQLiteStore

        self._omega = omega
        self._store_cls = SQLiteStore
        if emb:
            generate = getattr(omega_embedding, "generate_embedding", None)
            if generate is not None:
                generate("mnimi adapter check: load the configured embedder")
            info = omega_embedding.get_embedding_model_info() or {}
            if (omega_embedding.is_embedding_degraded()
                    or info.get("model_name") != emb.get("omega_name")
                    or info.get("backend") != "onnx"):
                raise RuntimeError(f"OMEGA's embedder is not the pinned one: {info}; "
                                   "the arm refuses a fallback or degraded embedder")
        self._store_kwargs = dict(competitor_config.get("store") or {})
        ingest = competitor_config.get("ingest") or {}
        self._skip_inference = bool(ingest.get("skip_inference", False))
        self._backdate = bool(ingest.get("backdate_created_at", False))
        self._stored: list[tuple[str, str | None]] = []
        self._store = None
        self._dir: Path | None = None
        self._n = 0
        self._rounds: dict = {}
        self._sessions = 0

    def competitor_pins(self) -> dict:
        emb = self._config_block.get("embedder") or {}
        return {
            "competitor_name": self.name,
            "competitor_version": self._config_block.get("package")
            or getattr(self._omega, "__version__", None),
            "competitor_embedder": f"{emb['repo']}@{emb['revision']}" if emb else "library default",
            "competitor_llm": "none",  # query expansion off; no LLM at write or read time
        }

    def reset(self) -> None:
        if self._store is not None:
            self._store.close()
            if self._dir is not None:
                shutil.rmtree(self._dir, ignore_errors=True)
        self._n += 1
        self._dir = Path(self._scratch.name) / f"q{self._n}"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store = self._store_cls(db_path=str(self._dir / "omega.db"), **self._store_kwargs)
        self._rounds = {}
        self._sessions = 0
        self._stored = []
        self._last_ids = []

    def add(self, messages: list[dict]) -> None:
        if not messages:
            return
        from mnimi.memory import _messages_to_rounds

        session_id = f"s{self._sessions:03d}"
        self._sessions += 1
        for rnd in _messages_to_rounds(messages):
            round_id = f"r{len(self._rounds)}"
            self._rounds[round_id] = rnd
            meta = {"session_date": rnd.ts, "round": round_id}
            iso = _iso(rnd.ts)
            if iso is not None:
                meta["referenced_date"] = iso
            node_id = self._store.store(content=rnd.content, session_id=session_id, metadata=meta,
                                        skip_inference=self._skip_inference)
            self._stored.append((node_id, rnd.ts))

    def set_question_date(self, question_date: str | None) -> None:
        """Backdate every record so the question's date reads as now (the authors' anchoring).

        OMEGA decays a record from its ``created_at`` against the wall clock. Its authors'
        LongMemEval script therefore rewrites ``created_at`` to the session's date, shifted
        so the question's date is the present. Without it every record is minutes old and
        the decay is flat. A later session's date wins for a record OMEGA deduplicated,
        as in the script, which stamps after each store.
        """
        if not self._backdate or self._store is None:
            return
        anchor = epoch(question_date)
        if anchor is None:
            return
        shift = datetime.now(timezone.utc).timestamp() - anchor
        conn = self._store._conn
        for node_id, ts in self._stored:
            seconds = epoch(ts)
            if node_id is None or seconds is None:
                continue
            stamp = datetime.fromtimestamp(seconds + shift, tz=timezone.utc).isoformat()
            conn.execute("UPDATE memories SET created_at = ? WHERE node_id = ?", (stamp, node_id))
        conn.commit()
        self._store._invalidate_query_cache()

    def get_context(self, query: str) -> str:
        results = self._store.query(query, limit=self.candidates)
        hits = []
        for r in results:
            rnd = self._rounds.get((r.metadata or {}).get("round"))
            if rnd is None:
                continue
            hits.append(Hit(str(r.id), rnd.content, rnd.ts, tuple(rnd.turns)))
        return self.context_from(hits)
