"""The paper freeze (Phase 7 D1): the library the paper describes is v2.14.0's, byte for byte.

A digest over every file under ``src/mnimi``, taken over LF-normalised bytes so a
Windows checkout with ``core.autocrlf`` and CI's Linux checkout agree, pinned here.
Lifting the freeze is a DECISIONS entry and a new constant, never an edit that
makes this test pass.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: sha256 over src/mnimi at v2.14.0 (e015931), set in PHASE7 Task 1 Step 4.
LIBRARY_FREEZE_SHA256 = "172457a3fd7da3bd2fdaa0cce880d30933fa84b830e6f97a205939299797c259"


def library_digest(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    lib = root / "src" / "mnimi"
    files = sorted(p for p in lib.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        data = path.read_bytes().replace(b"\r\n", b"\n")
        rel = path.relative_to(root).as_posix()
        h.update(f"{rel}\0{hashlib.sha256(data).hexdigest()}\n".encode())
    return h.hexdigest()


def test_the_library_is_frozen_for_the_paper():
    assert library_digest() == LIBRARY_FREEZE_SHA256, (
        "src/mnimi differs from the paper freeze (v2.14.0). The paper describes that library; "
        "lift the freeze with a DECISIONS entry and a new digest, never by editing the "
        "constant alone."
    )


def test_the_digest_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for root, text in ((lf, b"a = 1\nb = 2\n"), (crlf, b"a = 1\r\nb = 2\r\n")):
        (root / "src" / "mnimi").mkdir(parents=True)
        (root / "src" / "mnimi" / "x.py").write_bytes(text)
    assert library_digest(lf) == library_digest(crlf)
