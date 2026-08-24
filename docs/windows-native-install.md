# Native Windows installation

Turbofit supports Windows 10/11 with a user-scoped pinned llama.cpp runtime and gateway. No administrator account or machine-wide service is required.

This is the headless path when Hermes Dashboard is blocked or Desktop has no `/turbofit` command. Plugin install alone does **not** bind `127.0.0.1:8091`. Restarting the Hermes messaging gateway only reloads messaging adapters.

## Prerequisites

- Python 3.11+
- Git, CMake, and Ninja
- Visual Studio Build Tools with C++ support
- NVIDIA CUDA toolkit for `-Backend cuda`; otherwise use `-Backend cpu`

## One-shot path

From PowerShell in the installed plugin tree (or a Turbofit Git checkout):

```powershell
cd $env:LOCALAPPDATA\hermes\plugins\turbofit
# Review first — this registers user scheduled tasks and needs a model artifact:
# Get-Content .\scripts\install-windows-native-service.ps1 | more
python scripts\download-artifacts --family bonsai-27b
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-windows-native-service.ps1 -Backend cuda
curl.exe -s http://127.0.0.1:8091/v1/models
```

CPU-only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-windows-native-service.ps1 -Backend cpu
```

Prefer Dashboard **Turbofit → Apply** when possible so selection matches current hardware evidence.

## What the installer does

1. builds or verifies the pinned llama.cpp revision;
2. requires a model file (default Bonsai path under `%USERPROFILE%\Models\storage\gguf` or `scripts/download-artifacts`);
3. launches the floor model on loopback port 8092;
4. writes stable `auto`, `active:main`, and `active:aux` route state;
5. launches the Turbofit gateway on loopback port **8091**;
6. registers **TurbofitRuntime** + **TurbofitGateway** as limited user scheduled tasks;
7. verifies model and gateway health before returning.

Advanced users may pass `-Binary`, `-Model`, `-Context`, `-Port`, or `-GatewayPort` explicitly. Set `TURBOFIT_MODEL_ROOT` to override the default `%USERPROFILE%\Models\storage\gguf` destination.

## Health check

```powershell
curl.exe -s http://127.0.0.1:8091/v1/models
# or
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8091/v1/models', timeout=5).read()[:500])"
```

Success = JSON listing `auto` / active routes. `WinError 10061` = nothing is listening on 8091. That is setup missing, not a firewall rule.

If another machine on the LAN should use this box, persist the gateway on all interfaces and point the **remote** Hermes at the Windows LAN URL. Keep `custom:turbofit` / `auto`. Do not set `model.provider custom` or a GGUF filename.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-windows-native-service.ps1 -Backend vulkan -Binary C:\path\to\llama-server.exe -GatewayHost 0.0.0.0
```

On the **other** Hermes machine (not the Windows GPU box):

```text
/turbofit serve
```

or `turbofit_configure` with `base_url=http://192.168.1.101:8091/v1`. The Windows Sirvir profile stays on `http://127.0.0.1:8091/v1`.

## Uninstall services

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-windows-native-service.ps1 -Uninstall
```

This unregisters the two user tasks. It does not delete downloaded models or verified runtime sources.
