"""The paper freeze, as the harness enforces it (Phase 7 D1; PHASE8 D4).

``LIBRARY_FREEZE_SHA256`` is a digest over every tracked file under ``src/mnimi``,
taken over LF-normalised bytes so a Windows checkout with ``core.autocrlf`` and
CI's Linux checkout agree. ``tests/test_paper_freeze.py`` pins it; the runner
refuses a run of record (``--limit >= 100`` on the predict stage) when the digest
differs or the tree is dirty, unless ``--allow-unfrozen`` marks the run
provisional. Lifting the freeze is a DECISIONS entry and a new constant, never an
edit that makes the test pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: sha256 over src/mnimi at v2.14.0 (e015931), set in PHASE7 Task 1 Step 4.
LIBRARY_FREEZE_SHA256 = "223d39c60e9279a06ba6bf5b8993869c79f4f6789c829c5bb1dd864914257dfc"
#: The tag whose src/mnimi the digest pins (PHASE8 Task 17); v2.14.0's bytes, unchanged.
#: The era this digest names. "paper-v1" is the paper's library (tag paper-v1 = 21239e2);
#: "v3.0.0-dev" is release/v3 after TESTFIXESPT1 (DECISIONS 2026-10-14) — every commit that
#: edits src/mnimi recomputes the constant above in the same commit.
FREEZE_NAME = "v3.0.0-dev"


def library_digest(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    lib = root / "src" / "mnimi"
    files = sorted(p for p in lib.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        rel = path.relative_to(root).as_posix()
        h.update(f"{rel}\0{hashlib.sha256(data).hexdigest()}\n".encode())
    return h.hexdigest()


def check(root: Path = ROOT) -> list[str]:
    """Why a run of record must not start: an unfrozen library, a dirty tree (PHASE8 D4)."""
    from . import manifest as manifest_mod

    reasons = []
    digest = library_digest(root)
    if digest != LIBRARY_FREEZE_SHA256:
        reasons.append(f"src/mnimi differs from the paper freeze ({FREEZE_NAME}): "
                       f"{digest[:12]}... != {LIBRARY_FREEZE_SHA256[:12]}...")
    porcelain = manifest_mod.git_status_porcelain()
    if porcelain:
        reasons.append("harness tree is dirty:\n" + porcelain)
    return reasons
