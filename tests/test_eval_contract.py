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
