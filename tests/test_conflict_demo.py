"""Phase 3 Task 4: the conflict demo set and gate 3-ii (PHASE3 D10)."""

from __future__ import annotations

import sys

from evals.probes import conflict_demo
from evals.probes.conflict_demo import FAMILIES, SIZES, criterion, generate, run, summarize

from mnimi.conflict.normalize import normalize_triple


def test_demo_set_is_deterministic_and_sized():
    pairs = generate(0)
    assert pairs == generate(0), "the set is a pure function of the seed"
    assert pairs != generate(1)
    assert len(pairs) == 100 == sum(SIZES.values())
    assert FAMILIES == ("value-change", "negation", "dated-update", "paraphrase", "unrelated")
    by_family = {family: [p for p in pairs if p.family == family] for family in FAMILIES}
    assert {family: len(rows) for family, rows in by_family.items()} == SIZES
    ids = [p.id for p in pairs]
    assert len(set(ids)) == 100
    user_turns = [t["content"] for p in pairs for t in (p.turns_a[0], p.turns_b[0])]
    assert len(set(user_turns)) == 200, "every user turn keys its own script entry"
    assert all(t["role"] == "user" for p in pairs for t in (p.turns_a[0], p.turns_b[0]))
    conflicts = [p for p in pairs if p.family in ("value-change", "negation", "dated-update")]
    assert all(p.expect_supersede for p in conflicts) and len(conflicts) == 80
    assert not any(p.expect_supersede for p in pairs if p.family in ("paraphrase", "unrelated"))
    # The subfamily sizes D10 fixes.
    sub = {}
    for p in pairs:
        sub[(p.family, p.subfamily)] = sub.get((p.family, p.subfamily), 0) + 1
    assert sub == {
        ("value-change", "functional"): 15, ("value-change", "count"): 15,
        ("negation", "marker"): 15, ("negation", "antonym"): 15,
        ("dated-update", "in-order"): 10, ("dated-update", "reversed"): 10,
        ("paraphrase", "paraphrase"): 10, ("unrelated", "unrelated"): 10,
    }


def test_every_pair_normalizes_to_its_declared_pair_key():
    for pair in generate(0):
        (fact_a,), (fact_b,) = pair.facts_a, pair.facts_b
        ta = normalize_triple(fact_a.subject, fact_a.predicate, fact_a.object)
        tb = normalize_triple(fact_b.subject, fact_b.predicate, fact_b.object)
        assert ta is not None and tb is not None, pair.id
        assert ta.pair_key == tb.pair_key == pair.pair_key, pair.id
        # The raw spans are the role-prefixed verbatim sentences, and a dated
        # pair's mention sits verbatim in its own turn (the resolver's rule).
        assert fact_a.raw == "user: " + pair.turns_a[0]["content"], pair.id
        assert fact_b.raw == "user: " + pair.turns_b[0]["content"], pair.id
        for fact, turns in ((fact_a, pair.turns_a), (fact_b, pair.turns_b)):
            if fact.when is not None:
                assert fact.when.lower() in turns[0]["content"].lower(), pair.id
        if pair.family == "dated-update":
            assert fact_a.when is not None and fact_b.when is not None, pair.id
        if pair.family == "negation":
            assert pair.expected_polarity == -1 and ta.object == tb.object == pair.expected_object
            assert ta.polarity == +1 and tb.polarity == -1, pair.id
        elif pair.family == "unrelated":
            assert ta.object != tb.object, pair.id
        elif pair.family == "paraphrase":
            assert ta.object == tb.object == pair.expected_object, pair.id
        else:
            assert pair.expected_polarity == +1 and ta.object != tb.object, pair.id
            assert pair.expected_object in (ta.object, tb.object), pair.id


def test_gate_3ii_criterion_on_the_hashing_embedder(capsys):
    pairs = generate(0)
    on = run(pairs, conflict_resolution=True)
    off = run(pairs, conflict_resolution=False)
    summary_on, summary_off = summarize(on), summarize(off)
    table = conflict_demo.format_table(summary_on, summary_off)
    with capsys.disabled():
        sys.stdout.write("\n" + table + "\n")
    passed, why = criterion(summary_on, summary_off)
    assert passed, why
    # The pre-registered baseline expectation: an ambiguous store fails the
    # strict "exactly one active" reading on every conflict pair, and the
    # controls stand (D10).
    conflicts_off = sum(summary_off["families"][f]["correct_after_add"]
                        for f in ("value-change", "negation", "dated-update"))
    controls_off = sum(summary_off["families"][f]["correct_after_add"]
                       for f in ("paraphrase", "unrelated"))
    assert conflicts_off == 0 and controls_off == 20
    assert summary_off["conflicts"]["correct_after_consolidate"] == 0
    assert summary_off["controls"]["correct_after_consolidate"] == 20
    assert criterion(summary_off, summary_off)[0] is False


def test_demo_probe_imports_no_eval_only_dependency():
    import subprocess

    code = (
        "import sys, evals.probes.conflict_demo; "
        "assert not {'openai', 'ollama', 'dotenv', 'huggingface_hub', 'llama_cpp', "
        "'onnxruntime'} & set(sys.modules), sorted(set(sys.modules) & "
        "{'openai', 'ollama', 'dotenv', 'huggingface_hub', 'llama_cpp', 'onnxruntime'})"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
