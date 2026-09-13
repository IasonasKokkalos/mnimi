"""The extractor contract: a round's turns in, atomic facts out, pins declared.

An ``Extractor`` is the one seam through which an LLM touches the library. It
never sees the session date (the resolver anchors `when` on ``ts`` outside the
model), it never decides merge / conflict / decay, and everything that
determines its output is declared in ``pins`` so the store can refuse to mix
two configurations (SPEC §Reproducibility guard).
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Protocol, runtime_checkable


class ExtractedFact(NamedTuple):
    """One atomic fact as the model emitted it, before resolution and storage."""

    content: str
    """Self-contained sentence stating the fact ("The user …" / "The assistant …")."""
    raw: str
    """Verbatim, role-prefixed excerpt of the turn the fact came from."""
    when: str | None
    """The temporal expression copied verbatim from the turn, or ``None``.
    Resolved to ``valid_time`` deterministically against the message ``ts``
    (``mnimi.extract.resolver``); the model never computes a date."""
    subject: str | None
    predicate: str | None
    object: str | None
    """The triple, nullable as a unit (SPEC §Stage 2)."""
    salience: float


class ExtractionResult(NamedTuple):
    facts: list[ExtractedFact]
    truncated: bool
    """The model hit its token budget or the output did not parse: ``facts``
    is then ``[]`` (SPEC: a truncated parse is ``[]`` for that turn, logged)."""
    raw_output: str
    """What the model produced, byte for byte — what the cache stores."""
    truncated_input: bool = False
    """The round text was cut to the input cap before the model saw it."""


#: The five guard keys an extractor declares (SPEC §Reproducibility guard).
PIN_KEYS = (
    "extractor_model",
    "extractor_quant",
    "extractor_runtime",
    "extractor_decode_hash",
    "extractor_prompt_hash",
)


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns one round into facts under declared pins."""

    @property
    def pins(self) -> dict:
        """The ``PIN_KEYS`` → value mapping written to ``memory_meta``."""
        ...

    def extract(self, turns: list[dict]) -> ExtractionResult:
        """Facts for one round's ``{"role", "content"}`` turns."""
        ...


def canonical(obj) -> str:
    """Deterministic JSON — the library's one hashing serialisation."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extractor_pins_hash(pins: dict) -> str:
    """One string per extractor configuration — the cache file's name."""
    missing = [key for key in PIN_KEYS if key not in pins]
    if missing:
        raise ValueError(f"extractor pins are missing {missing}")
    return sha256_text(canonical({key: pins[key] for key in PIN_KEYS}))
