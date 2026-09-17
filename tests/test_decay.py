"""Logical time and decay as pure functions, and the frozen decay rules (PHASE4 D1-D3, D9)."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from mnimi import decay


def test_importing_mnimi_decay_loads_no_heavy_dependency():
    code = (
        "import sys, mnimi, mnimi.decay; "
        "assert not {'llama_cpp', 'huggingface_hub', 'onnxruntime'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_now_logical_is_the_latest_dated_session_timestamp():
    stamps = ["2023/05/20 (Sat) 09:00", "2023/06/01 (Thu) 08:00", "2023/05/31 (Wed) 23:00", None]
    assert decay.now_logical(stamps) == "2023/06/01 (Thu) 08:00"
    same_day = ["2023/06/01 (Thu) 08:00", "2023/06/01 (Thu) 14:05"]
    assert decay.now_logical(same_day) == "2023/06/01 (Thu) 14:05", "ties: the greater string"
    assert decay.now_logical(["2023-07-02", "2023/06/30 (Fri) 10:00"]) == "2023-07-02"
    assert decay.now_logical(["session one", None]) is None and decay.now_logical([]) is None


def test_logical_days_counts_whole_days_and_never_goes_negative():
    assert decay.logical_days("2023/06/30 (Fri) 10:00", "2023/05/31 (Wed) 23:59") == 30
    assert decay.logical_days("2023-05-20", "2023/05/20 (Sat) 09:00") == 0
    assert decay.logical_days("2023-05-19", "2023-05-20") == 0, "clamped at 0"
    assert decay.logical_days(None, "2023-05-20") == 0
    assert decay.logical_days("2023-05-20", "soon") == 0, "an undated side decays nothing"


def test_decayed_salience_halves_per_half_life_and_clamps_at_the_floor():
    table = [
        (1.0, 30, 0.5),
        (1.0, 60, 0.25),
        (0.5, 30, 0.25),
        (1.0, 90, 0.15),  # 0.125 clamps at the floor
        (0.25, 30, 0.15),  # 0.125 clamps at the floor
        (0.1, 30, 0.1),  # a salience below the floor is never raised
    ]
    for initial, days, expected in table:
        got = decay.decayed_salience(initial, days, 30.0, 0.15)
        assert got == pytest.approx(expected), (initial, days)


def test_decayed_salience_at_zero_days_is_the_initial_value_exactly():
    for initial in (1.0, 0.5, 0.25):
        assert decay.decayed_salience(initial, 0, 30.0, 0.15) == initial


def test_decay_rules_are_frozen_and_hashed(monkeypatch):
    assert decay.DECAY_RULES_VERSION == "v1"
    assert len(decay.DECAY_RULES) == 9 and all(rule.isascii() for rule in decay.DECAY_RULES)
    assert decay.decay_rules_hash() == (
        "d4a0bcf0733031ae8ff27116a26db3029ea5a3bf5ce1a5b1e28a0f13056617d5"
    )
    monkeypatch.setattr(decay, "DECAY_RULES", decay.DECAY_RULES[:-1])
    assert decay.decay_rules_hash() != (
        "d4a0bcf0733031ae8ff27116a26db3029ea5a3bf5ce1a5b1e28a0f13056617d5"
    )


def test_decay_reads_no_clock():
    source = inspect.getsource(decay)
    for forbidden in ("datetime.now", "date.today", "time.time", "utcnow"):
        assert forbidden not in source
