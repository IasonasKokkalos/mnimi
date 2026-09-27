"""The picklable system factory ``--workers`` hands each worker process (PHASE8 Task 14).

A worker process builds its own system once, from the same arguments the parent
used, and runs the unchanged per-question ``runner.ingest_and_context``. Only the
factory's qualified name and its arguments cross the process boundary, never a
live system.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemFactory:
    name: str
    kwargs: dict = field(default_factory=dict)

    def __call__(self):
        from .__main__ import build_system

        return build_system(self.name, **self.kwargs)
