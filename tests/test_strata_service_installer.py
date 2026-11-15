"""Strata runtime installer: systemd unit generation.

The unit is what keeps Strata alive across SSH sessions and reboots, and
what stops a stale runtime service from squatting VRAM.
"""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/install-strata-runtime"
    loader = SourceFileLoader("install_strata_runtime", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform != "linux", reason="systemd units are Linux-only")
def test_systemd_unit_is_written_and_wrapper_is_executable(tmp_path: Path) -> None:
    module = _module()
    result = module.install_systemd_unit(
        strata_dir=tmp_path / "Strata",
        venv_dir=tmp_path / "strata-venv",
        data_dir=tmp_path / "Strata-data",
        model="IQ2_XS",
        context=32768,
        port=8082,
        gpus="0,1",
        home=tmp_path,
        run_cmd=False,
    )

    assert result["service"] == "installed"

    unit = tmp_path / ".config/systemd/user/turbofit-strata.service"
    wrapper = tmp_path / ".local/bin/turbofit-strata-serve"
    assert unit.is_file()
    assert wrapper.is_file()

    unit_text = unit.read_text()
    assert "ExecStart=%h/.local/bin/turbofit-strata-serve" in unit_text
    assert "Restart=on-failure" in unit_text
    # A long start budget matters: experts take minutes to page in.
    assert "TimeoutStartSec=900" in unit_text

    wrapper_text = wrapper.read_text()
    assert "--model" in wrapper_text and "IQ2_XS" in wrapper_text
    assert "--port" in wrapper_text and "8082" in wrapper_text
    assert "--gpu" in wrapper_text and "0,1" in wrapper_text
    assert "--data-dir" in wrapper_text
    # Wrapper must be runnable by the unit.
    assert wrapper.stat().st_mode & 0o111


@pytest.mark.skipif(sys.platform != "linux", reason="systemd units are Linux-only")
def test_wrapper_defaults_are_overridable_via_environment(tmp_path: Path) -> None:
    module = _module()
    module.install_systemd_unit(
        strata_dir=tmp_path / "Strata",
        venv_dir=tmp_path / "strata-venv",
        data_dir=tmp_path / "Strata-data",
        home=tmp_path,
        run_cmd=False,
    )
    wrapper_text = (tmp_path / ".local/bin/turbofit-strata-serve").read_text()
    # Every value the user might need to retune is an env override.
    for var in ("STRATA_MODEL", "STRATA_CONTEXT", "STRATA_PORT", "STRATA_GPUS",
                "STRATA_HOST", "STRATA_DATA_DIR", "STRATA_ALLOWED_HOSTS"):
        assert var in wrapper_text, var


def test_installer_accepts_service_flags() -> None:
    module = _module()
    parser_source = (ROOT / "scripts/install-strata-runtime").read_text()
    assert "--install-service" in parser_source
    assert callable(module.install_systemd_unit)
