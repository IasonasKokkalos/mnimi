"""Conflict handling: the deterministic dedup screens and supersession (PHASE3).

SPEC §Dedup strategy steps 3–5 and §Write path step 4, built without an LLM:
a frozen negation lexicon (`.lexicon`), triple normalization and the pair key
(`.normalize`), the negation / value-substitution screens and the entropy
gate over cosine-gate-pass pairs (`.screens`, Task 2), and the ordering that
decides which of two conflicting facts stays active (`.supersede`, Task 3).

Everything here is a pure function of stored fields; nothing reads the
clock, nothing calls a model. Two artifacts are pinned in ``memory_meta``:
``negation_lexicon_hash`` (the markers and antonym groups) and
``conflict_rules_hash`` (the normalization tables, the functional-predicate
groups, the value-token limit and the rule identifiers). Editing either is a
versioned migration plus re-ingest, never an in-place change under a run.
"""
