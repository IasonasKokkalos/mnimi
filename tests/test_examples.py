"""``examples/chat.py`` — the chat loop with memory (MERGED-PLAN T3, LAUNCH §6.1).

A smoke test with no network and no model: the example's ``build_memory`` under
``--embedder hashing --extractor none``, one scripted turn through a fake backend
callable, and the ``/export`` command writing ``memory.md`` beside the DB through
``mnimi_cli``. The clock lives in the example (one ``ts`` per launch), never in the
library, so the test hands it a fixed date.
"""

from __future__ import annotations

import datetime
import importlib.util
import pathlib

import pytest

_CHAT = pathlib.Path("examples/chat.py")


@pytest.fixture(scope="module")
def chat():
    spec = importlib.util.spec_from_file_location("mnimi_example_chat", _CHAT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_ts_is_the_calendar_day_at_midnight(chat):
    assert chat.session_ts(datetime.date(2026, 10, 9)) == "2026-10-09T00:00:00"


def test_build_memory_hashing_none_needs_no_extra(chat, tmp_path):
    memory = chat.build_memory(str(tmp_path / "a.db"), embedder="hashing", extractor="none")

    assert memory.embedder.name == "hashing"
    assert memory.extractor is None


def test_one_turn_feeds_context_then_stores_both_turns(chat, tmp_path):
    memory = chat.build_memory(str(tmp_path / "t.db"), embedder="hashing", extractor="none")
    seen: list[tuple[str, str]] = []

    def backend(system: str, user: str) -> str:
        seen.append((system, user))
        return "Boston, noted."

    ts = chat.session_ts(datetime.date(2026, 10, 9))
    reply = chat.turn(memory, "me", backend, "I live in Boston.", ts)
    reply2 = chat.turn(memory, "me", backend, "Where do I live?", ts)

    assert reply == "Boston, noted." and reply2 == "Boston, noted."
    assert seen[0][0].startswith("You are an assistant with memory.")
    assert seen[0][1] == "I live in Boston."
    assert "I live in Boston." in seen[1][0], "the second turn's prompt carries the first round"
    hits = memory.recall("Boston", "me")
    assert len(hits) == 2 and all(h.record.turns for h in hits)
    assert hits[0].record.turns[0]["content"] in ("I live in Boston.", "Where do I live?")
    assert memory.store.all_records("me")[0].created_at == ts


def test_export_command_writes_memory_md_beside_the_db(chat, tmp_path, capsys):
    db = tmp_path / "e.db"
    memory = chat.build_memory(str(db), embedder="hashing", extractor="none")
    ts = chat.session_ts(datetime.date(2026, 10, 9))
    chat.turn(memory, "me", lambda s, u: "ok", "I like tea.", ts)

    assert chat.handle_command("/export", memory, "me", str(db), "I like tea.", ts) is True

    out = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert out.startswith("# memory export\n") and "I like tea." in out
    assert "memory.md" in capsys.readouterr().out


def test_recall_and_facts_commands_print_without_an_extractor(chat, tmp_path, capsys):
    db = tmp_path / "r.db"
    memory = chat.build_memory(str(db), embedder="hashing", extractor="none")
    ts = chat.session_ts(datetime.date(2026, 10, 9))
    chat.turn(memory, "me", lambda s, u: "ok", "My cat is called Pixel.", ts)

    assert chat.handle_command("/recall cat", memory, "me", str(db), "x", ts) is True
    assert chat.handle_command("/facts", memory, "me", str(db), "My cat is called Pixel.",
                               ts) is True
    assert chat.handle_command("/quit", memory, "me", str(db), "x", ts) is False
    assert chat.handle_command("hello", memory, "me", str(db), "x", ts) is None

    out = capsys.readouterr().out
    assert "Pixel" in out and "score" in out
    assert "no facts" in out, "without an extractor the round stored no fact records"


def test_example_imports_no_backend_at_module_level():
    source = _CHAT.read_text(encoding="utf-8")
    head = source.split("def ", 1)[0]
    for forbidden in ("import ollama", "import openai", "from openai", "from ollama"):
        assert forbidden not in head, f"{forbidden} at module level - backends load lazily"
