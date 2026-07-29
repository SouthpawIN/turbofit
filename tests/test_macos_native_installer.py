from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "install_macos_native_service",
    ROOT / "scripts" / "install-macos-native-service",
    loader=SourceFileLoader(
        "install_macos_native_service",
        str(ROOT / "scripts" / "install-macos-native-service"),
    ),
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pinned_runtime_is_preferred_over_path(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("")
    monkeypatch.setattr(MODULE, "LLAMA_RUNTIME_BINARY", binary)
    monkeypatch.setattr(
        MODULE.shutil,
        "which",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not inspect PATH")),
    )

    assert MODULE._llama_server() == str(binary)


def test_existing_pinned_runtime_is_version_checked(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("")
    monkeypatch.setattr(MODULE, "LLAMA_RUNTIME_BINARY", binary)
    monkeypatch.setattr(MODULE, "_llama_version", lambda _path: "version: 10173 (test)")

    assert MODULE._install_llama_runtime() == str(binary)


def test_native_launch_enables_bounded_prompt_cache() -> None:
    arguments = MODULE._model_args("/runtime/llama-server")

    assert arguments[0] == "/runtime/llama-server"
    assert arguments[arguments.index("--cache-reuse") + 1] == "256"
    assert arguments[arguments.index("--cache-ram") + 1] == "1024"
    assert arguments[arguments.index("--checkpoint-min-step") + 1] == "2048"
    assert arguments[arguments.index("--batch-size") + 1] == "256"
    assert arguments[arguments.index("--ubatch-size") + 1] == "128"


def test_native_route_requires_the_16gb_hardware_class() -> None:
    assert MODULE._route_state()["active"] == "hardware-16gb-macos-native"
