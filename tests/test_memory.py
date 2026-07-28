"""Memory facade behavior: config threading, dedup, and the add() contract."""

from __future__ import annotations

from mnimi import Memory, MemoryConfig
from mnimi.embeddings import HashingEmbedder


def test_config_defaults_match_spec():
    config = MemoryConfig()
    assert config.dedup_cosine_threshold == 0.85
    assert config.top_k == 10


def test_config_is_threaded_into_memory(tmp_path):
    config = MemoryConfig(dedup_cosine_threshold=0.9, top_k=3)
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder(), config)
    assert memory.config is config


def test_memory_defaults_to_spec_config(tmp_path):
    memory = Memory(str(tmp_path / "mem.db"), HashingEmbedder())
    assert memory.config.dedup_cosine_threshold == 0.85
    assert memory.config.top_k == 10
