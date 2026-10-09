"""A chat loop with memory: the smallest real use of mnimi (MERGED-PLAN T3, LAUNCH §6.1).

    python examples/chat.py --embedder hashing --extractor none
    python examples/chat.py --backend ollama --embedder bge --extractor cpu

Each turn: ``get_context(question)`` -> a system prompt -> the reply -> ``add()`` of both
turns. One ``ts`` per launch: a day is a session, the benchmark's shape. The clock lives
HERE, in the application, and never inside the library, which reads only the ``ts`` it is
handed. Commands: ``/recall <q>``, ``/facts``, ``/export``, ``/consolidate``, ``/quit``.

The backends (Ollama, OpenAI) are the example's dependencies, not the library's, and are
imported only when chosen. ``examples/README.md`` has the install lines and the week's
six tests.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from collections.abc import Callable

from mnimi import Memory, MemoryConfig

SYSTEM_PROMPT = "You are an assistant with memory. Relevant memories:\n{context}"

DEFAULT_MODEL = {"ollama": "qwen2.5:7b-instruct", "openai": "gpt-4o-mini"}

#: What a backend is to this loop: ``(system_prompt, user_text) -> reply``.
Backend = Callable[[str, str], str]


def session_ts(today: datetime.date | None = None) -> str:
    """One timestamp per launch: the calendar day at midnight (a day is a session)."""
    day = today or datetime.date.today()
    return day.isoformat() + "T00:00:00"


def build_embedder(name: str):
    if name == "hashing":
        from mnimi.embeddings import HashingEmbedder

        return HashingEmbedder()
    from mnimi.embeddings import BgeSmallEmbedder

    try:
        return BgeSmallEmbedder()
    except ImportError as exc:
        sys.exit(f'--embedder bge needs the [embed] extra: pip install "mnimi[embed]" ({exc})')


def build_extractor(name: str, db: str):
    """``none`` (the v1 write path: rounds only), ``cpu`` (the recipe), ``gpu`` (the pin)."""
    if name == "none":
        return None
    try:
        from mnimi.extract.llama import DECODE, QwenLlamaExtractor
    except ImportError as exc:
        sys.exit(f'--extractor {name} needs the [extract] extra: pip install "mnimi[extract]" '
                 f"({exc})")
    from mnimi.extract.cache import CachedExtractor

    # The CPU profile passes the public decode= keyword and nothing else: it yields its
    # own extractor_decode_hash, so a CPU store is its own configuration (LAUNCH M7).
    decode = dict(DECODE) if name == "gpu" else {**DECODE, "n_gpu_layers": 0}
    cache = os.path.join(os.path.dirname(os.path.abspath(db)), f"extract-cache-{name}.sqlite")
    try:
        return CachedExtractor(QwenLlamaExtractor(decode=decode), cache)
    except (ImportError, RuntimeError) as exc:
        sys.exit(f"--extractor {name}: {exc}")


def build_memory(db: str, embedder: str = "bge", extractor: str = "none") -> Memory:
    return Memory(db, build_embedder(embedder), MemoryConfig(),
                  extractor=build_extractor(extractor, db))


def build_backend(name: str, model: str | None = None) -> Backend:
    model = model or DEFAULT_MODEL[name]
    if name == "ollama":
        try:
            import ollama
        except ImportError:
            sys.exit("--backend ollama needs `pip install ollama` and a running Ollama")

        def reply(system: str, user: str) -> str:
            response = ollama.chat(model=model, messages=[
                {"role": "system", "content": system}, {"role": "user", "content": user}])
            return response["message"]["content"]

        return reply
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("--backend openai needs `pip install openai` and OPENAI_API_KEY")
    client = OpenAI()

    def reply_openai(system: str, user: str) -> str:
        response = client.chat.completions.create(model=model, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": user}])
        return response.choices[0].message.content or ""

    return reply_openai


def turn(memory: Memory, user_id: str, backend: Backend, text: str, ts: str) -> str:
    """One round: context in, reply out, both turns stored under the session's ``ts``."""
    context = memory.get_context(text, user_id)
    reply = backend(SYSTEM_PROMPT.format(context=context), text)
    memory.add([{"role": "user", "content": text, "ts": ts},
                {"role": "assistant", "content": reply, "ts": ts}], user_id)
    return reply


def _first_line(record) -> str:
    """What a hit is: a fact record's fact, else the round's first turn."""
    if record.kind == "fact" and record.fact:
        return record.fact
    turns = record.turns or [{"content": record.content}]
    return turns[0].get("content", "")


def _facts_of_round(context: str, user_text: str) -> list[str]:
    """The ``facts:`` bullets the renderer printed just before the round's user turn.

    Only lines inside a ``facts:`` block count: a reply's own markdown list continues
    an ``assistant:`` line without a prefix and must not read as facts.
    """
    facts: list[str] = []
    in_facts = False
    for line in context.split("\n"):
        if line == "facts:":
            facts, in_facts = [], True
        elif in_facts and line.startswith("- "):
            facts.append(line[2:])
        elif line == f"user: {user_text}":
            return facts
        else:
            if line.startswith(("user: ", "assistant: ")):
                facts = []
            in_facts = False
    return []


def handle_command(command: str, memory: Memory, user_id: str, db: str, last_text: str,
                   ts: str) -> bool | None:
    """Run a ``/command``. ``True`` = handled, ``False`` = quit, ``None`` = not a command."""
    if not command.startswith("/"):
        return None
    name, _, rest = command.partition(" ")
    if name == "/quit":
        return False
    if name == "/recall":
        for hit in memory.recall(rest.strip(), user_id):
            print(f"score {hit.score:.3f}  salience {hit.salience:.2f}  {hit.record.kind}  "
                  f"[{hit.record.created_at}] {_first_line(hit.record)}")
        return True
    if name == "/facts":
        facts = _facts_of_round(memory.get_context(last_text, user_id), last_text)
        if not facts:
            print("no facts stored for the last round (an extractor is needed: "
                  "--extractor cpu|gpu)")
        for fact in facts:
            print(f"- {fact}")
        return True
    if name == "/export":
        from mnimi_cli.main import main as mnimi_cli_main

        path = os.path.join(os.path.dirname(os.path.abspath(db)), "memory.md")
        if mnimi_cli_main(["export", db, user_id, "-o", path, "--format", "md"]) == 0:
            print(f"wrote {path}")
        return True
    if name == "/consolidate":
        memory.consolidate(user_id)
        print("consolidated")
        return True
    print(f"unknown command {name}; /recall <q>, /facts, /export, /consolidate, /quit")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--db", default="./agent.db")
    parser.add_argument("--user", default="me")
    parser.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", help="the backend's model (default per backend)")
    parser.add_argument("--embedder", choices=("bge", "hashing"), default="bge")
    parser.add_argument("--extractor", choices=("cpu", "gpu", "none"), default="none")
    args = parser.parse_args(argv)

    # The library's INFO lines are the point of watching it work: `superseded ...`,
    # `routed to conflict ...`, `decayed ...` on mnimi.memory.
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("mnimi.memory").setLevel(logging.INFO)

    memory = build_memory(args.db, args.embedder, args.extractor)
    backend = build_backend(args.backend, args.model)
    ts = session_ts()
    print(f"mnimi chat: db={args.db} user={args.user} session ts={ts}; "
          "/recall <q>, /facts, /export, /consolidate, /quit")
    last_text = ""
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        handled = handle_command(text, memory, args.user, args.db, last_text, ts)
        if handled is False:
            return 0
        if handled is True:
            continue
        last_text = text
        print(f"assistant> {turn(memory, args.user, backend, text, ts)}")


if __name__ == "__main__":
    sys.exit(main())
