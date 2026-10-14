"""The library freeze (Phase 7 D1; the guard PHASE8 D4; lifted for the developer release in
DECISIONS 2026-10-14 "TESTFIXESPT1"): a run of record uses the committed library, whose digest is
pinned in ``evals.freeze`` under the era ``FREEZE_NAME``. The paper's library is tag ``paper-v1``.

A digest over every file under ``src/mnimi``, taken over LF-normalised bytes so a
Windows checkout with ``core.autocrlf`` and CI's Linux checkout agree, pinned in
``evals.freeze``. Lifting the freeze is a DECISIONS entry and a new constant,
never an edit that makes this test pass.
"""

from __future__ import annotations

from evals.freeze import LIBRARY_FREEZE_SHA256, library_digest


def test_the_library_is_frozen_for_the_paper():
    assert library_digest() == LIBRARY_FREEZE_SHA256, (
        "src/mnimi differs from the committed freeze (FREEZE_NAME). Recompute "
        "LIBRARY_FREEZE_SHA256 in the commit that edits the library, with a DECISIONS entry "
        "for the era."
    )


def test_the_digest_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for root, text in ((lf, b"a = 1\nb = 2\n"), (crlf, b"a = 1\r\nb = 2\r\n")):
        (root / "src" / "mnimi").mkdir(parents=True)
        (root / "src" / "mnimi" / "x.py").write_bytes(text)
    assert library_digest(lf) == library_digest(crlf)


def test_check_names_a_changed_library(tmp_path, monkeypatch):
    from evals import freeze
    (tmp_path / "src" / "mnimi").mkdir(parents=True)
    (tmp_path / "src" / "mnimi" / "x.py").write_bytes(b"x = 1\n")
    monkeypatch.setattr("evals.manifest.git_status_porcelain", lambda: "")
    reasons = freeze.check(tmp_path)
    assert len(reasons) == 1 and "differs from the paper freeze (v3.0.0-dev)" in reasons[0]


def test_check_names_a_dirty_tree(monkeypatch):
    from evals import freeze
    monkeypatch.setattr("evals.manifest.git_status_porcelain", lambda: " M evals/x.py")
    reasons = freeze.check()
    assert any(r.startswith("harness tree is dirty") for r in reasons)
