# Native Windows installation

Turbofit supports Windows 10/11 with a user-scoped pinned llama.cpp runtime and gateway. No administrator account or machine-wide service is required.

## Prerequisites

- Python 3.11+
- Git, CMake, and Ninja
- Visual Studio Build Tools with C++ support
- NVIDIA CUDA toolkit for `-Backend cuda`; otherwise use `-Backend cpu`

## Artifacts

From PowerShell in the Turbofit plugin directory:

```powershell
python scripts\download-artifacts --family bonsai-27b
```

Artifacts are revision- and SHA-pinned by `references/artifact-manifest.json`. Set `TURBOFIT_MODEL_ROOT` to override the default `%USERPROFILE%\Models\storage\gguf` destination.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows-native-service.ps1 -Backend cuda
```

The installer:

1. builds or verifies the pinned llama.cpp revision;
2. launches the Bonsai floor on loopback port 8092;
3. writes stable `auto`, `active:main`, and `active:aux` route state;
4. launches the Turbofit gateway on loopback port 8091;
5. registers both as limited user scheduled tasks; and
6. verifies model and gateway health before returning.

CPU-only installation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows-native-service.ps1 -Backend cpu
```

Advanced users may pass `-Binary`, `-Model`, `-Context`, `-Port`, or `-GatewayPort` explicitly.

## Uninstall services

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows-native-service.ps1 -Uninstall
```

This unregisters the two user tasks. It does not delete downloaded models or verified runtime sources.
