"""A15 (PHASE8 D10): a store built from a full cache never loads the GGUF or needs a CUDA build.

The unit test never imports ``llama_cpp`` (the test-path rule): the two seams that
touch it — constructing the real extractor and computing its runtime pins — are
patched, and the wrapper's behaviour is what is asserted. The ``extract``-marked
test checks the real pins against the real class on a machine that has the extra.
"""

from __future__ import annotations

import pytest

from mnimi.extract.cache import CachedExtractor
from mnimi.extract.protocol import PIN_KEYS

FAKE_PINS = {key: f"fake-{key}" for key in PIN_KEYS}


def test_a_full_cache_never_constructs_the_real_extractor(tmp_path, monkeypatch):
    from evals import lazy_extractor

    def boom(**kwargs):
        raise AssertionError("the real extractor was constructed")

    monkeypatch.setattr(lazy_extractor, "_construct", boom)
    monkeypatch.setattr(lazy_extractor, "_pins", lambda **kwargs: dict(FAKE_PINS))
    lazy = lazy_extractor.LazyQwenExtractor()
    assert set(lazy.pins) == set(PIN_KEYS)
    cached = CachedExtractor(lazy, tmp_path / "c.sqlite")
    turns = [{"role": "user", "content": "I live in Boston."}]
    cached.db.execute(
        "INSERT INTO extractions (key, raw_output, truncated, truncated_input) VALUES (?, ?, 0, 0)",
        (CachedExtractor.key(turns), "[]"),
    )
    cached.db.commit()
    result = cached.extract(turns)
    assert result.facts == [] and cached.stats == {"hits": 1, "misses": 0}


def test_a_cache_miss_constructs_the_real_extractor_once_and_delegates(tmp_path, monkeypatch):
    from evals import lazy_extractor

    from mnimi.extract.protocol import ExtractionResult

    built = []

    class _Real:
        pins = dict(FAKE_PINS)

        def extract(self, turns):
            return ExtractionResult(facts=[], truncated=False, raw_output="[]")

    def construct(**kwargs):
        built.append(kwargs)
        return _Real()

    monkeypatch.setattr(lazy_extractor, "_construct", construct)
    monkeypatch.setattr(lazy_extractor, "_pins", lambda **kwargs: dict(FAKE_PINS))
    lazy = lazy_extractor.LazyQwenExtractor(verbose=True)
    lazy.extract([{"role": "user", "content": "a"}])
    lazy.extract([{"role": "user", "content": "b"}])
    assert built == [{"verbose": True}], "constructed once, with the wrapper's kwargs"


@pytest.mark.extract
def test_lazy_pins_equal_the_real_pins():
    pytest.importorskip("llama_cpp")  # the [extract] extra; skipped in CI like the bge tests
    from evals import lazy_extractor

    from mnimi.extract.llama import QwenLlamaExtractor

    assert lazy_extractor.LazyQwenExtractor().pins == QwenLlamaExtractor().pins
