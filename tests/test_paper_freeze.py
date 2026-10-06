"""The paper freeze (Phase 7 D1; the guard PHASE8 D4): the library the paper describes is
``paper-v1``'s, which is v2.14.0's, byte for byte.

A digest over every file under ``src/mnimi``, taken over LF-normalised bytes so a
Windows checkout with ``core.autocrlf`` and CI's Linux checkout agree, pinned in
``evals.freeze``. Lifting the freeze is a DECISIONS entry and a new constant,
never an edit that makes this test pass.
"""

from __future__ import annotations

from evals.freeze import LIBRARY_FREEZE_SHA256, library_digest


def test_the_library_is_frozen_for_the_paper():
    assert library_digest() == LIBRARY_FREEZE_SHA256, (
        "src/mnimi differs from the paper freeze (paper-v1). The paper describes that library; "
        "lift the freeze with a DECISIONS entry and a new digest, never by editing the "
        "constant alone."
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
    assert len(reasons) == 1 and "differs from the paper freeze (paper-v1)" in reasons[0]


def test_check_names_a_dirty_tree(monkeypatch):
    from evals import freeze
    monkeypatch.setattr("evals.manifest.git_status_porcelain", lambda: " M evals/x.py")
    reasons = freeze.check()
    assert any(r.startswith("harness tree is dirty") for r in reasons)
