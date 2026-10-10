"""``mnimi-mcp`` — the stdio MCP server over ``Memory`` (MERGED-PLAN T8, LAUNCH §6.2).

Driven through the SDK's in-memory client (``Client(server)``: no subprocess, no
transport) under ``MNIMI_EMBEDDER=hashing MNIMI_EXTRACTOR=none`` over a ``tmp_path``
DB. Skipped when the ``mcp`` package is absent (it is not in the root ``[dev]`` extra,
as ``bge`` and ``extract`` are skipped without theirs).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys

import pytest

pytestmark = pytest.mark.mcp
try:
    from mcp import Client

    from mnimi_mcp import server
except ImportError:  # pragma: no cover - CI's path: no mcp, or an mcp 1.x without Client
    pytest.skip(
        "needs the mcp 2.x package (pip install -e integrations/mcp "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu)",
        allow_module_level=True,
    )

DAY = datetime.date(2026, 10, 10)
TOOLS = {"remember", "recall", "context", "export", "consolidate"}


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MNIMI_DB", str(tmp_path / "agent.db"))
    monkeypatch.setenv("MNIMI_EMBEDDER", "hashing")
    monkeypatch.setenv("MNIMI_EXTRACTOR", "none")
    monkeypatch.delenv("MNIMI_USER_ID", raising=False)
    monkeypatch.setattr(server, "today", lambda: DAY)
    server.reset()
    yield tmp_path
    server.reset()


async def _call(name: str, **arguments):
    async with Client(server.mcp) as client:
        result = await client.call_tool(name, arguments)
    assert not result.is_error, result
    return result


def _structured(result):
    return result.structured_content


def test_the_five_tools_and_the_export_resource_are_listed_before_any_memory_is_built(env):
    async def go():
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
        return tools, resources

    tools, resources = run(go())
    assert {t.name for t in tools.tools} == TOOLS
    assert [str(r.uri) for r in resources.resources] == ["memory://export"]
    assert server._memory is None, "listing builds no Memory (lazy on first call)"


def test_remember_stamps_the_servers_day_and_recall_round_trips(env):
    stored = _structured(run(_call("remember", messages=[
        {"role": "user", "content": "I live in Boston."},
        {"role": "assistant", "content": "Boston, noted."},
    ])))
    assert stored == {"round": 1, "fact": 0, "superseded": 0, "ts": "2026-10-10T00:00:00"}

    hits = _structured(run(_call("recall", query="where do I live", k=5)))["result"]
    assert len(hits) == 1
    hit = hits[0]
    assert set(hit) == {"score", "salience", "kind", "text", "session_date"}
    assert hit["kind"] == "round" and hit["session_date"] == "2026-10-10T00:00:00"
    assert "I live in Boston." in hit["text"] and "Boston, noted." in hit["text"]
    assert hit["salience"] == 1.0 and 0.0 <= hit["score"] <= 1.0 + 1e-9

    context = _structured(run(_call("context", query="where do I live")))["result"]
    assert context.startswith("[Session date: 2026-10-10T00:00:00]")
    assert "user: I live in Boston." in context


def test_a_model_supplied_ts_is_ignored_the_servers_day_wins(env):
    run(_call("remember", messages=[
        {"role": "user", "content": "My dentist is on Thursday.", "ts": "1999-01-01T00:00:00"},
    ]))
    memory = server.memory()
    records = memory.store.all_records("me")
    assert [r.created_at for r in records] == ["2026-10-10T00:00:00"]


def test_remember_refuses_a_bad_role_and_an_empty_list(env):
    async def bad(messages):
        async with Client(server.mcp) as client:
            return await client.call_tool("remember", {"messages": messages})

    role = run(bad([{"role": "system", "content": "x"}]))
    assert role.is_error and "role" in role.content[0].text
    empty = run(bad([]))
    assert empty.is_error and "at least 1" in empty.content[0].text, empty.content[0].text


def test_export_carries_the_remembered_text_and_the_resource_equals_the_tool(env):
    run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
    exported = _structured(run(_call("export")))["result"]
    from mnimi.export import HEADER

    assert exported.startswith(HEADER)
    assert "## 2026-10-10T00:00:00" in exported and "I live in Boston." in exported

    async def read():
        async with Client(server.mcp) as client:
            return await client.read_resource("memory://export")

    contents = run(read()).contents
    assert len(contents) == 1 and contents[0].text == exported
    assert contents[0].mime_type == "text/markdown"


def test_consolidate_returns_the_passes_own_deltas_and_is_idempotent(env, monkeypatch):
    monkeypatch.setattr(server, "today", lambda: datetime.date(2026, 9, 1))
    run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
    monkeypatch.setattr(server, "today", lambda: DAY)  # 39 days later: one half-life and a bit
    run(_call("remember", messages=[{"role": "user", "content": "The weather is fine."}]))
    first = _structured(run(_call("consolidate")))
    second = _structured(run(_call("consolidate")))
    assert set(first) == {"conflict", "decay", "now_logical"}
    assert first["now_logical"] == "2026-10-10T00:00:00"
    # The counts are THIS call's, not the process's lifetime: one pass, the old round decayed.
    assert first["decay"] == {"passes": 1, "decayed": 1, "at_floor": 0}
    assert second["decay"] == {"passes": 1, "decayed": 0, "at_floor": 0}, "idempotent"
    assert first["conflict"]["superseded"] == 0 and first["conflict"]["pairs_screened"] == 0


def test_user_id_comes_from_the_environment(env, monkeypatch):
    monkeypatch.setenv("MNIMI_USER_ID", "jason")
    server.reset()
    run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
    memory = server.memory()
    assert memory.store.count("jason") == 1 and memory.store.count("me") == 0


def test_nothing_is_written_to_stdout_and_the_library_logs_to_stderr(env, capsys):
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, logging.getLogger("mnimi.memory").level,
             logging.getLogger("mnimi_mcp").level)
    try:
        server.configure_logging()
        assert [h.stream for h in root.handlers] == [sys.stderr]
        run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
        run(_call("recall", query="Boston"))
        out, err = capsys.readouterr()
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        logging.getLogger("mnimi.memory").setLevel(saved[2])
        logging.getLogger("mnimi_mcp").setLevel(saved[3])
    assert out == "", "stdio transport owns stdout"
    assert "mnimi_mcp: opening" in err, "the server's own INFO line went to stderr"


def test_the_settings_are_read_from_env_and_db_is_required(monkeypatch, tmp_path):
    monkeypatch.delenv("MNIMI_DB", raising=False)
    monkeypatch.delenv("MNIMI_USER_ID", raising=False)
    with pytest.raises(ValueError, match="MNIMI_DB is required"):
        server.settings()  # a ValueError, never SystemExit: a tool call must not kill the server
    monkeypatch.setenv("MNIMI_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("MNIMI_EXTRACTOR", "turbo")
    with pytest.raises(ValueError, match="MNIMI_EXTRACTOR"):
        server.settings()
    monkeypatch.setenv("MNIMI_EXTRACTOR", "none")
    monkeypatch.setenv("MNIMI_EMBEDDER", "hashing")
    s = server.settings()
    assert (s.db, s.user_id, s.extractor, s.embedder) == (
        str(tmp_path / "x.db"), "me", "none", "hashing",
    )


def test_mnimi_db_expands_a_tilde_and_env_vars_and_is_made_absolute(monkeypatch, tmp_path):
    home = tmp_path / "home"
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.setenv("MNIMI_DB", "~/mnimi/agent.db")  # the shape a shell passes through literally
    monkeypatch.setenv("MNIMI_EXTRACTOR", "none")
    monkeypatch.setenv("MNIMI_EMBEDDER", "hashing")
    assert server.settings().db == os.path.join(str(home), "mnimi", "agent.db")
    monkeypatch.setenv("MNIMI_VAR_FOR_TEST", str(tmp_path))
    monkeypatch.setenv("MNIMI_DB", "${MNIMI_VAR_FOR_TEST}/a.db")
    assert server.settings().db == os.path.join(str(tmp_path), "a.db")
    monkeypatch.setenv("MNIMI_DB", "relative.db")
    assert os.path.isabs(server.settings().db)


def test_main_exits_with_the_message_when_the_db_is_missing(monkeypatch):
    monkeypatch.delenv("MNIMI_DB", raising=False)
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert "MNIMI_DB is required" in str(exc.value)


def test_a_failed_open_reaches_the_client_as_the_tools_error_text(env):
    # The store was built under other pins: the client must read the MemoryMetaError, not a
    # bare "Error executing tool", and the resource must say the same.
    from mnimi import Memory
    from mnimi.embeddings import HashingEmbedder

    Memory(os.environ["MNIMI_DB"], HashingEmbedder(dim=128)).store.close()

    async def go():
        async with Client(server.mcp) as client:
            tool = await client.call_tool("context", {"query": "x"})
            try:
                await client.read_resource("memory://export")
            except Exception as exc:  # noqa: BLE001 - the SDK's client-side error type
                return tool, str(exc)
            return tool, ""

    tool, resource_error = run(go())
    assert tool.is_error
    assert "memory_meta mismatch" in tool.content[0].text and "embedder_dim" in tool.content[0].text
    assert "memory_meta mismatch" in resource_error
    assert server._memory is None, "a failed build leaves nothing half-open"


def test_recall_and_context_hand_the_library_the_servers_date_as_the_query_prefix(env, monkeypatch):
    run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
    mem = server.memory()
    seen: list[str] = []
    original_recall = mem.recall

    def spy(query, user_id):
        seen.append(query)
        return original_recall(query, user_id)

    monkeypatch.setattr(mem, "recall", spy)
    run(_call("recall", query="where did I live 39 days ago"))
    run(_call("context", query="where do I live"))
    # The shipped configuration's time-aware term fires only on the documented prefix; the bare
    # question is what gets embedded (mnimi.temporal.split_query strips it).
    assert seen == [
        "[Current date: 2026-10-10T00:00:00]\nwhere did I live 39 days ago",
        "[Current date: 2026-10-10T00:00:00]\nwhere do I live",
    ]


def test_the_version_is_one_value():
    import mnimi_mcp

    assert server.mcp.version == mnimi_mcp.__version__ == "0.1.0"


def test_tool_results_are_json_serialisable(env):
    run(_call("remember", messages=[{"role": "user", "content": "I live in Boston."}]))
    for name, args in (("recall", {"query": "Boston"}), ("export", {}), ("consolidate", {})):
        json.dumps(_structured(run(_call(name, **args))))
