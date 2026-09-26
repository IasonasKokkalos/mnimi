"""``--config <file>``: a committed argument list, expanded in place (PHASE8 D5; PLAN A3).

``configs/<run_id>.json`` is ``{"config_schema": "mnimi-run-config/1", "args": [...]}``:
exactly the arguments that ran, spliced where ``--config`` stood, so a later
command-line flag overrides one from the file. The manifest records the file's
sha256 and whether ``git show HEAD:<file>`` holds the same bytes; a run of record
(``--limit >= 100``) needs a committed config or ``--allow-unfrozen``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

CONFIG_SCHEMA = "mnimi-run-config/1"


def load(path: str | Path) -> list[str]:
    """The argument list of a config file, checked for its schema and shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("config_schema") != CONFIG_SCHEMA:
        raise ValueError(f"{path}: config_schema must be {CONFIG_SCHEMA!r}")
    args = payload.get("args")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError(f"{path}: 'args' must be a list of strings")
    return list(args)


def expand(argv: list[str]) -> tuple[list[str], str | None]:
    """Replace the first ``--config <path>`` pair by the file's arguments; return the path."""
    argv = list(argv)
    if "--config" not in argv:
        return argv, None
    i = argv.index("--config")
    if i + 1 >= len(argv):
        raise ValueError("--config needs a file")
    path = argv[i + 1]
    return argv[:i] + load(path) + argv[i + 2:], path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_show(path: str | Path) -> bytes | None:
    """The committed bytes of ``path`` at HEAD, or ``None`` when git has none."""
    try:
        rel = Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel = Path(path).as_posix()
    try:
        proc = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, timeout=5,
                              check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def committed(path: str | Path) -> bool:
    """Whether the file's bytes are what HEAD holds at that path."""
    shown = _git_show(path)
    return shown is not None and hashlib.sha256(shown).hexdigest() == sha256(path)
