"""A fresh SQLite file per question, for the store-backed systems."""

from __future__ import annotations

import tempfile
from pathlib import Path


class ScratchDb:
    """Hands out a new database path per ``reset()`` and deletes the old one.

    A real file rather than ``:memory:`` on purpose: the on-disk path is the
    one a user runs, it is what exercises ``Store``'s open-time behaviour
    (schema creation and the ``memory_meta`` write), and an in-memory store
    would quietly skip both. Files are deleted as soon as they are replaced —
    a 500-question run would otherwise leave 500 stores behind.
    """

    def __init__(self, prefix: str) -> None:
        # ignore_cleanup_errors because the LAST store of a run is still open
        # when the interpreter tears the directory down — nothing calls reset()
        # after the final question. On Windows that unlink raises WinError 32.
        # The cost is one leftover file per process in %TEMP%; the alternative
        # is finalizer ordering machinery inside a benchmark harness.
        self._tmp = tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)
        self._count = 0
        self.path: Path | None = None

    def next(self) -> Path:
        """Delete the previous database and return a fresh path.

        The caller must close its connection first: on Windows an open handle
        makes the unlink fail, which would silently accumulate files.
        """
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self._count += 1
        self.path = Path(self._tmp.name) / f"q{self._count}.db"
        return self.path
