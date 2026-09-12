from __future__ import annotations

import importlib.util
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/install-mlx-runtime"


def load_script():
    loader = SourceFileLoader("install_mlx_runtime", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installer_creates_isolated_pinned_mlx_runtime(tmp_path: Path) -> None:
    module = load_script()
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(list(command))
        if command[1] == "venv":
            python = tmp_path / "mlx-lm-0.31.3/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("python")
            python.chmod(0o755)

    record = module.install(
        runtime_root=tmp_path,
        uv="/opt/homebrew/bin/uv",
        platform_name="darwin",
        architecture="arm64",
        run=run,
        version_probe=lambda _python: {"mlx": "0.32.2", "mlx-lm": "0.31.3"},
    )

    python = tmp_path / "mlx-lm-0.31.3/.venv/bin/python"
    assert calls == [
        [
            "/opt/homebrew/bin/uv",
            "venv",
            "--python",
            "3.12",
            str(tmp_path / "mlx-lm-0.31.3/.venv"),
        ],
        [
            "/opt/homebrew/bin/uv",
            "pip",
            "install",
            "--python",
            str(python),
            "mlx==0.32.2",
            "mlx-lm==0.31.3",
        ],
    ]
    assert record["status"] == "verified"
    assert record["python"] == str(python)


def test_installer_fails_closed_off_apple_silicon(tmp_path: Path) -> None:
    module = load_script()

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        module.install(
            runtime_root=tmp_path,
            uv="uv",
            platform_name="linux",
            architecture="x86_64",
        )


def test_installer_help_bootstraps_from_legacy_system_python() -> None:
    system_python = Path("/usr/bin/python3")
    if not system_python.exists():
        return
    result = subprocess.run(
        [str(system_python), str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0
    assert "unsupported operand type" not in result.stderr
    assert "Install or verify" in result.stdout
