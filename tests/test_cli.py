"""``mnimi export`` — the CLI over ``Memory.export()`` (MERGED-PLAN T2, LAUNCH §5).

The CLI is the sibling package ``mnimi_cli``: it opens a store with NO model (an embedder
and an extractor stub built from the store's own ``memory_meta`` rows), and writes either
``text`` — ``Memory.export()``'s bytes unchanged — or ``md`` — a transform of the records:
one heading per session, the round's turns as a quoted block of the renderer's bytes, the
facts as bullets, a superseded fact struck through. ``evals/`` never imports it.
"""

from __future__ import annotations

import pathlib

import pytest

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder
from mnimi_cli.main import main


def _message(content: str, role: str = "user", ts: str = "2023-05-20") -> dict:
    return {"role": role, "content": content, "ts": ts}


def _fact(content, raw, subject=None, predicate=None, obj=None, when=None, salience=1.0):
    from mnimi.extract.protocol import ExtractedFact

    return ExtractedFact(content=content, raw=raw, when=when, subject=subject,
                         predicate=predicate, object=obj, salience=salience)


_SCRIPT = {
    "Quick note: I live in Boston now.": [
        _fact("The user lives in Boston.", "user: I live in Boston now.", "user", "lives in",
              "Boston")],
    "Update from me: I live in Seattle now.": [
        _fact("The user lives in Seattle.", "user: I live in Seattle now.", "user", "lives in",
              "Seattle")],
}

_TS_A = "2023/01/10 (Tue) 09:00"
_TS_B = "2023/06/10 (Sat) 09:00"


def _scripted_store(db_path, script=None):
    """A store with two sessions and one supersession (the Boston fact loses to Seattle)."""
    from mnimi.extract.fake import ScriptedExtractor

    script = script if script is not None else _SCRIPT
    m = Memory(str(db_path), HashingEmbedder(), MemoryConfig(dedup_cosine_threshold=0.5),
               extractor=ScriptedExtractor(script))
    text_a, text_b = list(script)
    m.add([_message(text_a, ts=_TS_A), _message("Noted.", role="assistant", ts=_TS_A)],
          user_id="u")
    m.add([_message(text_b, ts=_TS_B)], user_id="u")
    return m


def test_text_format_is_the_library_export_bytes(tmp_path, capsys):
    m = _scripted_store(tmp_path / "t.db")
    expected = m.export("u")

    assert main(["export", str(tmp_path / "t.db"), "u", "--format", "text"]) == 0

    assert capsys.readouterr().out == expected


def test_export_opens_the_store_without_embedding_anything(tmp_path):
    """The CLI's stub embedder never embeds: a round-only store (no extractor) opens too."""
    m = Memory(str(tmp_path / "r.db"), HashingEmbedder())
    m.add([_message("hello", ts=_TS_A)], user_id="u")

    assert main(["export", str(tmp_path / "r.db"), "u", "-o", str(tmp_path / "r.md")]) == 0

    assert "hello" in (tmp_path / "r.md").read_text(encoding="utf-8")


def test_md_shows_both_facts_with_the_loser_struck_through(tmp_path):
    m = _scripted_store(tmp_path / "s.db")
    loser = [r for r in m.store.all_records("u") if r.kind == "fact" and r.salience == 0]
    winner = [r for r in m.store.all_records("u") if r.kind == "fact" and r.salience > 0]
    assert len(loser) == 1 and len(winner) == 1

    assert main(["export", str(tmp_path / "s.db"), "u", "-o", str(tmp_path / "s.md")]) == 0

    md = (tmp_path / "s.md").read_text(encoding="utf-8")
    assert "~~The user lives in Boston.~~" in md, "the loser is shown, struck through"
    assert f"(superseded by #{winner[0].id})" in md
    assert "The user lives in Seattle." in md
    assert "~~The user lives in Seattle.~~" not in md, "the winner is not struck through"
    assert f"→ supersedes #{loser[0].id}" in md, "export.py's direction: the winner supersedes"


def test_md_has_one_heading_per_session_oldest_first(tmp_path):
    _scripted_store(tmp_path / "h.db")

    assert main(["export", str(tmp_path / "h.db"), "u", "-o", str(tmp_path / "h.md")]) == 0

    md = (tmp_path / "h.md").read_text(encoding="utf-8")
    assert md.startswith("# memory export\n")
    assert "user_id: u" in md and f"now_logical: {_TS_B}" in md
    assert "records: 4 (rounds 2, facts 2)" in md
    assert md.count("\n## ") == 2, "one heading per session, none per round"
    assert md.index(f"## {_TS_A}") < md.index(f"## {_TS_B}")


def test_md_quotes_the_renderer_bytes_for_each_round(tmp_path):
    from mnimi.memory import render_records

    m = _scripted_store(tmp_path / "q.db")
    rounds = [r for r in m.store.all_records("u") if r.kind == "round"]

    assert main(["export", str(tmp_path / "q.db"), "u", "-o", str(tmp_path / "q.md")]) == 0

    md = (tmp_path / "q.md").read_text(encoding="utf-8")
    for record in rounds:
        quoted = "\n".join("> " + line for line in render_records([record]).split("\n"))
        assert quoted in md, "a round's quoted block is the renderer's bytes, line for line"


def test_md_bolds_a_fact_valid_time(tmp_path):
    script = {
        "I moved to Boston in March.": [
            _fact("The user moved to Boston.", "user: I moved to Boston in March.",
                  "user", "moved to", "Boston", when="March")],
        "Nothing to extract here.": [],
    }
    m = _scripted_store(tmp_path / "v.db", script)
    facts = [r for r in m.store.all_records("u") if r.kind == "fact"]
    assert len(facts) == 1 and facts[0].valid_time

    assert main(["export", str(tmp_path / "v.db"), "u", "-o", str(tmp_path / "v.md")]) == 0

    md = (tmp_path / "v.md").read_text(encoding="utf-8")
    assert f"- The user moved to Boston. (**{facts[0].valid_time}**)" in md


def test_missing_db_exits_non_zero_with_a_message(tmp_path, capsys):
    code = main(["export", str(tmp_path / "absent.db"), "u"])

    assert code != 0
    assert "absent.db" in capsys.readouterr().err
    assert not (tmp_path / "absent.db").exists(), "the CLI never creates a store"


def test_stdout_survives_a_cp1252_console(tmp_path, monkeypatch):
    """`mnimi export db u` to a Windows pipe (cp1252) must not crash on the arrow."""
    import io
    import sys

    _scripted_store(tmp_path / "c.db")
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))

    assert main(["export", str(tmp_path / "c.db"), "u"]) == 0

    sys.stdout.flush()
    assert "→ supersedes #" in raw.getvalue().decode("utf-8")


def test_o_writes_the_file_and_prints_nothing(tmp_path, capsys):
    _scripted_store(tmp_path / "w.db")

    assert main(["export", str(tmp_path / "w.db"), "u", "-o", str(tmp_path / "out.md")]) == 0

    assert (tmp_path / "out.md").read_text(encoding="utf-8").startswith("# memory export\n")
    assert capsys.readouterr().out == ""


def test_export_help_names_the_two_formats(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["export", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "md" in out and "text" in out


def test_evals_never_imports_the_cli():
    for path in pathlib.Path("evals").rglob("*.py"):
        assert "mnimi_cli" not in path.read_text(encoding="utf-8"), path


def test_cli_package_imports_no_model_runtime():
    source = pathlib.Path("src/mnimi_cli/main.py").read_text(encoding="utf-8")
    for forbidden in ("onnxruntime", "llama_cpp", "huggingface_hub", "BgeSmallEmbedder"):
        assert forbidden not in source, f"{forbidden} in the CLI - export loads no model"
