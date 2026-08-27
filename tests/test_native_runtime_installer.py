from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/install-native-runtimes"
    loader = SourceFileLoader("install_native_runtimes", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_native_installer_generates_backend_specific_cmake_configuration() -> None:
    module = _module()

    assert module.cmake_configuration("cuda") == ("build-cuda", "-DGGML_CUDA=ON")
    assert module.cmake_configuration("rocm") == ("build-rocm", "-DGGML_HIP=ON")
    assert module.cmake_configuration("metal") == ("build-metal", "-DGGML_METAL=ON")
    assert module.cmake_configuration("vulkan") == ("build-vulkan", "-DGGML_VULKAN=ON")
    assert module.cmake_configuration("cpu") == ("build-cpu", "-DGGML_NATIVE=ON")


def test_native_installer_resolves_visual_studio_release_executable(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module.os, "name", "nt")

    assert module.native_server_binary(Path("runtime"), "build-vulkan") == Path(
        "runtime/build-vulkan/bin/Release/llama-server.exe"
    )


def test_native_installer_excludes_separate_freetoken_candidate() -> None:
    module = _module()
    manifest = __import__("json").loads((ROOT / "references/native-runtimes.json").read_text())

    selected = module.select_native_runtimes(manifest["runtimes"], set())

    assert {item["id"] for item in selected} == {
        "mainline-llama.cpp", "prism-llama.cpp", "ik-llama.cpp"
    }
    dflash = module.select_native_runtimes(manifest["runtimes"], {"dflash2-llama.cpp"})
    assert [item["id"] for item in dflash] == ["dflash2-llama.cpp"]