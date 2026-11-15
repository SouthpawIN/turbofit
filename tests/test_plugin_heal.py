"""Registration heals: the provider must reach profile homes, not the active one N times."""

from __future__ import annotations

import yaml

from plugin_tools import _save_configuration_to_all_homes, ensure_provider_registered


def _home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "turbosouth").mkdir(parents=True)
    (home / "profiles" / "turbosouth" / "config.yaml").write_text(
        "model: turbosouth-only\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_every_home_gets_a_turbofit_provider(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)

    touched = ensure_provider_registered(base_url="http://127.0.0.1:8091/v1")

    assert len(touched) == 2
    for config in (home / "config.yaml", home / "profiles/turbosouth/config.yaml"):
        payload = yaml.safe_load(config.read_text())
        assert payload["providers"]["turbofit"]["api"] == "http://127.0.0.1:8091/v1"


def test_the_heal_is_idempotent(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    ensure_provider_registered(base_url="http://127.0.0.1:8091/v1")
    assert ensure_provider_registered(base_url="http://127.0.0.1:8091/v1") == []


def test_the_heal_keeps_what_was_already_in_a_profile(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)

    ensure_provider_registered(base_url="http://127.0.0.1:8091/v1")

    payload = yaml.safe_load((home / "profiles/turbosouth/config.yaml").read_text())
    assert payload["model"] == "turbosouth-only"
    assert "turbofit" in payload["providers"]


def test_fanning_a_configuration_out_writes_distinct_homes(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)

    _save_configuration_to_all_homes({"providers": {"turbofit": {"api": "http://x/v1"}}})

    default = yaml.safe_load((home / "config.yaml").read_text())
    profile = yaml.safe_load((home / "profiles/turbosouth/config.yaml").read_text())
    assert default["providers"]["turbofit"]["api"] == "http://x/v1"
    assert profile["providers"]["turbofit"]["api"] == "http://x/v1"
    # The profile keeps its own identity -- the old stub rewrote one file N times.
    assert profile["model"] == "turbosouth-only"
