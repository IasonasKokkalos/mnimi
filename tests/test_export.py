"""``Memory.export()`` — the human-readable store dump (PHASE5 Task 9, D11).

SPEC §Human-readable memory: "``export()`` dumps the store as readable text/markdown",
with the original fact recoverable without a reverse lookup. The rules this file pins:
rounds render through the ONE shared renderer (never a second format), a superseded fact
is marked rather than hidden, the output is deterministic, it reads no clock, and no eval
arm can reach it — which is what makes it unable to move a benchmark number.
"""

from __future__ import annotations

import pathlib

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder


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


def _scripted(db_path, script=None, **config):
    from mnimi.extract.fake import ScriptedExtractor

    return Memory(str(db_path), HashingEmbedder(),
                  MemoryConfig(dedup_cosine_threshold=0.5, **config),
                  extractor=ScriptedExtractor(script if script is not None else _SCRIPT))


def _two_sessions(m, script=_SCRIPT):
    text_a, text_b = list(script)
    m.add([_message(text_a, ts=_TS_A)], user_id="u")
    m.add([_message(text_b, ts=_TS_B)], user_id="u")


def test_export_header_carries_the_user_now_logical_and_counts_by_kind(tmp_path):
    m = _scripted(tmp_path / "h.db")
    _two_sessions(m)

    text = m.export("u")

    assert text.startswith("# memory export\n")
    assert "user_id: u\n" in text
    assert f"now_logical: {_TS_B}\n" in text
    assert "records: 4 (rounds 2, facts 2)\n" in text


def test_export_groups_records_by_session_oldest_first(tmp_path):
    m = _scripted(tmp_path / "o.db")
    _two_sessions(m)

    # The header names now_logical (the LATEST session), so ordering is asserted on
    # the body alone — otherwise the assertion reads the header, not the records.
    body = m.export("u").split("\n\n", 1)[1]

    assert body.index(_TS_A) < body.index(_TS_B), "sessions run oldest first"
    assert body.index("Boston") < body.index("Seattle")


def test_export_renders_rounds_through_the_one_shared_renderer(tmp_path):
    from mnimi.memory import render_records

    m = _scripted(tmp_path / "r.db")
    _two_sessions(m)
    rounds = [r for r in m.store.all_records("u") if r.kind == "round"]

    text = m.export("u")

    for record in rounds:
        block = render_records([record])
        assert block in text, "a round's block is the renderer's bytes, not a second format"


def test_export_fact_line_carries_salience_pair_key_and_valid_time(tmp_path):
    script = {
        "I moved to Boston in March.": [
            _fact("The user moved to Boston.", "user: I moved to Boston in March.",
                  "user", "moved to", "Boston", when="March")],
    }
    m = _scripted(tmp_path / "f.db", script)
    m.add([_message("I moved to Boston in March.", ts=_TS_A)], user_id="u")

    text = m.export("u")

    assert "fact: The user moved to Boston." in text
    assert "salience 1.0000" in text
    # "moved to" normalizes into the frozen "lives in" functional group
    # (mnimi.conflict.normalize) — the pair_key is the normalized predicate.
    assert "pair user|lives in" in text
    assert "valid_time " in text, "a resolved date is recoverable without a reverse lookup"


def test_export_marks_a_superseded_fact_rather_than_hiding_it(tmp_path):
    m = _scripted(tmp_path / "s.db")
    _two_sessions(m)
    superseded = [r for r in m.store.all_records("u")
                  if r.kind == "fact" and r.salience == 0]
    assert len(superseded) == 1, "the Boston fact loses to the Seattle one"

    text = m.export("u")

    assert "The user lives in Boston." in text, "a superseded fact is kept, not hidden"
    assert "superseded" in text
    assert "salience 0.0000" in text


def test_export_of_an_empty_store_returns_a_header_and_no_records(tmp_path):
    m = Memory(str(tmp_path / "e.db"), HashingEmbedder())

    text = m.export("nobody")

    assert text.startswith("# memory export\n")
    assert "records: 0 (rounds 0, facts 0)\n" in text
    assert "now_logical: -\n" in text


def test_export_is_deterministic_across_calls(tmp_path):
    m = _scripted(tmp_path / "d.db")
    _two_sessions(m)

    assert m.export("u") == m.export("u")


def test_export_reads_no_wall_clock():
    source = pathlib.Path("src/mnimi/export.py").read_text(encoding="utf-8")
    for forbidden in ("datetime.now", "date.today", "time.time", "utcnow"):
        assert forbidden not in source, f"{forbidden} in export.py - all time is logical"


def test_no_eval_system_imports_the_export_module():
    for path in pathlib.Path("evals/systems").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "mnimi.export" not in source, path
        assert ".export(" not in source, f"{path} calls export() - it cannot reach a number"
