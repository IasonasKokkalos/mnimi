"""mnimi — embeddable, local-first agent memory."""

from .config import MemoryConfig
from .memory import Memory

__all__ = ["Memory", "MemoryConfig"]
