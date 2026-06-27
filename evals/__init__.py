"""LongMemEval harness for agent-memory systems.

Standalone from the ``agentmem`` library: ``base.MemorySystem`` is the contract
any system under test implements, so baselines and competitors run through the
same rig. Kept import-light on purpose — pulling in ``evals`` or ``evals.systems``
does not import ``anthropic`` or ``huggingface_hub`` (those load lazily in the
modules that actually call them).
"""
