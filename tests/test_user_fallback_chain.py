"""The fallback chain belongs to the operator, not to the shipped defaults.

Two things they asked for: pick any smaller model on this machine, or any model
served by another TurboFit node on the tailnet — and have that choice survive a
restart without editing the repo. A remote level costs this machine no VRAM,
which is the whole point of shifting to a sibling box while the GPUs are busy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime import chain as chain_mod


def _doc(levels: list[dict]) -> dict:
    return {
        "schema": chain_mod.CHAIN_SCHEMA,
        "context_floor": 65536,
        "lanes": {"dedicated": {"label": "Dedicated", "levels": levels}},
    }


def _local(level_id: str = "big") -> dict:
    return {
        "id": level_id, "kind": "model", "model": "qwen3-8-27b-gsq-rco-iq3s",
        "engine": "llama.cpp", "context": 65536, "needs_mb_per_card": 20000,
        "evidence": "measured",
    }


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "fallback-chain.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_remote_level_is_accepted_and_costs_no_local_vram(tmp_path: Path) -> None:
    remote = {
        "id": "sibling", "kind": "model", "model": "glm-5-3-flash-maya-xs",
        "engine": "strata", "context": 131072, "needs_mb_per_card": 0,
        "evidence": "portable-fit",
        "endpoint": "http://100.68.42.68:8091/v1", "peer": "omarchy",
    }
    loaded = chain_mod.load_chain(_write(tmp_path, _doc([_local(), remote])))
    level = loaded.level("dedicated", "sibling")
    assert level is not None
    assert level.remote is True
    assert level.needs_mb_per_card == 0
    payload = level.to_dict()
    assert payload["remote"] is True
    assert payload["peer"] == "omarchy"
    assert payload["endpoint"].startswith("http://")


def test_local_level_is_not_remote(tmp_path: Path) -> None:
    loaded = chain_mod.load_chain(_write(tmp_path, _doc([_local()])))
    level = loaded.level("dedicated", "big")
    assert level is not None and level.remote is False
    assert "endpoint" not in level.to_dict()


def test_endpoint_must_be_an_http_url(tmp_path: Path) -> None:
    bad = _local("sibling")
    bad["endpoint"] = "100.68.42.68:8091"
    with pytest.raises(ValueError, match="http"):
        chain_mod.load_chain(_write(tmp_path, _doc([bad])))


def test_endpoint_is_only_valid_on_a_model_level(tmp_path: Path) -> None:
    bad = {
        "id": "drop", "kind": "yield", "model": None, "engine": None,
        "context": 0, "needs_mb_per_card": 0, "evidence": "measured",
        "endpoint": "http://100.68.42.68:8091/v1",
    }
    with pytest.raises(ValueError, match="model level"):
        chain_mod.load_chain(_write(tmp_path, _doc([bad])))


def test_user_chain_overrides_the_shipped_one(tmp_path: Path, monkeypatch) -> None:
    path = _write(tmp_path, _doc([_local("mine")]))
    env = {"TURBOFIT_CHAIN": str(path)}
    assert chain_mod.user_chain_path(env) == path
    monkeypatch.setenv("TURBOFIT_CHAIN", str(path))
    loaded = chain_mod.load_chain()  # no explicit path: env must win
    assert loaded.level("dedicated", "mine") is not None


def test_xdg_config_chain_is_found(tmp_path: Path) -> None:
    config = tmp_path / "turbofit"
    config.mkdir()
    (config / "fallback-chain.json").write_text(json.dumps(_doc([_local("xdg")])))
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    assert chain_mod.user_chain_path(env) == config / "fallback-chain.json"


def test_no_user_chain_falls_back_to_the_shipped_default(tmp_path: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    assert chain_mod.user_chain_path(env) is None
    shipped = chain_mod.load_chain()
    assert shipped.lane_ids()
