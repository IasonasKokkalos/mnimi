"""--config: a committed argument list, expanded in place (PHASE8 D5)."""

from __future__ import annotations

import json

import pytest
from evals import config_file


def _write(path, args, schema="mnimi-run-config/1"):
    path.write_text(json.dumps({"config_schema": schema, "args": args}), encoding="utf-8")


def test_expand_splices_the_file_where_the_flag_stood(tmp_path):
    cfg = tmp_path / "c.json"
    _write(cfg, ["--system", "mnimi", "--limit", "500"])
    argv, used = config_file.expand(["--config", str(cfg), "--limit", "20"])
    assert argv == ["--system", "mnimi", "--limit", "500", "--limit", "20"] and used == str(cfg)
    assert config_file.expand(["--system", "x"]) == (["--system", "x"], None)


def test_load_refuses_a_wrong_schema_or_shape(tmp_path):
    cfg = tmp_path / "c.json"
    _write(cfg, ["--system"], schema="other/1")
    with pytest.raises(ValueError, match="config_schema"):
        config_file.load(cfg)
    cfg.write_text(json.dumps({"config_schema": "mnimi-run-config/1", "args": "--system"}),
                   encoding="utf-8")
    with pytest.raises(ValueError, match="list of strings"):
        config_file.load(cfg)


def test_sha256_and_committed(tmp_path, monkeypatch):
    cfg = tmp_path / "c.json"
    _write(cfg, ["--system", "mnimi"])
    assert len(config_file.sha256(cfg)) == 64
    monkeypatch.setattr(config_file, "_git_show", lambda p: cfg.read_bytes())
    assert config_file.committed(cfg) is True
    monkeypatch.setattr(config_file, "_git_show", lambda p: None)
    assert config_file.committed(cfg) is False
