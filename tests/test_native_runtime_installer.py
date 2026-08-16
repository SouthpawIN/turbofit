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