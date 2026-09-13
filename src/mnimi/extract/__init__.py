"""The write-time extraction stage: the one LLM, its pre-filter, cache and resolver.

SPEC §Extraction (locked): a deterministic speech-act pre-filter, then a pinned,
local, grammar-constrained model that turns a round into atomic facts. This
package holds the protocol (`.protocol`), the output schema and grammar
(`.schema`), the pinned prompt (`.prompt`), the runtime binding (`.llama`), a
model-free stand-in for CI (`.fake`), the on-disk cache (`.cache`), the stage-1
pre-filter (`.prefilter`) and the relative-date resolver (`.resolver`).

Heavy dependencies (``llama_cpp``, ``huggingface_hub``) are imported only
inside `.llama`, behind the ``[extract]`` extra: ``import mnimi`` and
``import mnimi.extract`` work with the two core deps alone, and a test asserts
it. Nothing here reads wall-clock time; every date is resolved against the
message ``ts``.
"""
