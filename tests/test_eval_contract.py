"""The baselines satisfy the MemorySystem contract.

Imports only ``evals.base`` and ``evals.systems`` — no ``anthropic`` or
``huggingface_hub`` — so this runs under the ``[dev]`` extra alone.
"""

from __future__ import annotations

from evals.base import MemorySystem
from evals.systems.full_history import FullHistorySystem
from evals.systems.no_memory import NoMemorySystem


def test_both_systems_are_memorysystem_subclasses():
    assert issubclass(NoMemorySystem, MemorySystem)
    assert issubclass(FullHistorySystem, MemorySystem)


def test_systems_expose_the_contract_methods():
    for cls in (NoMemorySystem, FullHistorySystem):
        system = cls()
        for method in ("reset", "add", "get_context"):
            assert callable(getattr(system, method))


def test_no_memory_returns_empty_context():
    system = NoMemorySystem()
    system.reset()
    system.add([{"role": "user", "content": "I live in Athens"}])
    assert system.get_context("where do I live?") == ""


def test_sessions_carry_structured_ts_not_a_pseudo_turn():
    """Phase B contract: the timestamp is a field, not an injected system turn."""
    from evals.dataset import Session
    from evals.runner import _session_to_messages

    session = Session(
        session_id="s1",
        date="2023-05-20",
        turns=[
            {"role": "user", "content": "I live in Athens"},
            {"role": "assistant", "content": "Noted"},
        ],
    )
    messages = _session_to_messages(session)

    assert len(messages) == 2, "no extra harness-invented turn"
    assert {m["role"] for m in messages} == {"user", "assistant"}
    assert all(m["ts"] == "2023-05-20" for m in messages)


def test_full_history_keeps_session_dates_reader_visible():
    """Temporal questions are unanswerable if the dates never reach the reader."""
    system = FullHistorySystem()
    system.reset()
    system.add([{"role": "user", "content": "I moved", "ts": "2023-05-20"}])
    system.add([{"role": "user", "content": "I moved again", "ts": "2023-07-01"}])

    context = system.get_context("when did I move?")

    assert "2023-05-20" in context
    assert "2023-07-01" in context
    # One dated header per session, not one per turn.
    assert context.count("[Session date:") == 2


def test_full_history_concatenates_then_resets():
    system = FullHistorySystem()
    system.reset()
    system.add([{"role": "user", "content": "I love hiking"}])
    system.add([{"role": "assistant", "content": "Noted!"}])

    context = system.get_context("what do I love?")
    assert "hiking" in context
    assert "Noted!" in context

    system.reset()
    assert system.get_context("what do I love?") == ""
