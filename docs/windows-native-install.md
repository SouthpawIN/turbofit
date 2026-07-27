# Windows Native Installation Guide

Run TurboFit natively on Windows with an NVIDIA GPU — no Docker, no WSL2.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11 (64-bit) |
| **GPU** | NVIDIA with ≥8 GB VRAM (RTX 3060 or better recommended) |
| **Driver** | NVIDIA driver ≥ 535 (CUDA 12.x compatible) |
| **Python** | 3.10+ (from [python.org](https://python.org)) |
| **Disk** | ≥25 GB free (models + runtime) |

## Step 1 — Install llama.cpp (prebuilt)

Download the latest Windows CUDA build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases):

```powershell
# Example for b10107 (CUDA 12.4) — check releases for latest
mkdir C:\Users\%USERNAME%\turbofit\llama.cpp
# Extract llama-server.exe into that directory
```

Verify:

```powershell
C:\Users\%USERNAME%\turbofit\llama.cpp\llama-server.exe --version
```

## Step 2 — Create Python venv

```powershell
cd C:\Users\%USERNAME%\turbofit
python -m venv .venv
.venv\Scripts\activate
pip install pyyaml pynvml
```

## Step 3 — Clone TurboFit

```powershell
git clone https://github.com/SouthpawIN/turbofit.git repo
```

## Step 4 — Download models

```powershell
python scripts\download-models.py --base-dir C:\Users\%USERNAME%\.turbohaul\models
```

This downloads with resume support and verifies SHA-256 checksums. Models:

| Model | Size | VRAM needed |
|-------|------|-------------|
| Bonsai 27B Q1_0 | 3.8 GB | ~4 GB |
| GRM 2.6 Plus Q4_K_M | 16.8 GB | ~18 GB |

## Step 5 — Create manifests

Create `C:\Users\%USERNAME%\.turbohaul\manifests\` with one JSON file per model.

**`bonsai-27b-q1.json`:**

```json
{
  "gguf_path": "C:\\Users\\%USERNAME%\\.turbohaul\\models\\prism-ml--Bonsai-27B-gguf\\Bonsai-27B-Q1_0.gguf",
  "family": "llama",
  "llama_server_flags": {
    "ctx_size": 262144,
    "n_gpu_layers": 99,
    "parallel": 1,
    "flash_attn": "on",
    "cache_type_k": "q4_0",
    "cache_type_v": "q4_0",
    "jinja": true,
    "no_perf": true,
    "threads": 8
  }
}
```

**`grm-2.6-plus-q4.json`:**

```json
{
  "gguf_path": "C:\\Users\\%USERNAME%\\.turbohaul\\models\\DAXZEIT--GRM-2.6-Plus-0628-MTP-reasoning-i1-GGUF\\grm-2.6-plus-0628-Q4_K_M-reasoning-imat.gguf",
  "family": "llama",
  "llama_server_flags": {
    "ctx_size": 131072,
    "n_gpu_layers": 99,
    "parallel": 1,
    "flash_attn": "on",
    "cache_type_k": "q4_0",
    "cache_type_v": "q4_0",
    "jinja": true,
    "no_perf": true,
    "threads": 8
  }
}
```

> **Context sizing:** `--parallel N` splits `ctx_size` evenly across N slots.
> With `parallel: 1`, the full context is available to a single request.
> KV cache type `q4_0` halves cache memory vs `f16` with minimal quality loss.

## Step 6 — Start the shim

```powershell
.venv\Scripts\python.exe scripts\turbohaul-shim.py --port 11401
```

The shim:
- Discovers manifests in `~/.turbohaul/manifests/`
- Lazy-loads models on first request
- Exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models`
- Supports SSE streaming (converts non-stream upstream to SSE for clients)
- Retries failed inference up to 3 times

Verify:

```powershell
curl http://127.0.0.1:11401/health
# {"status": "ok", "backend": "native-llama-server"}

curl http://127.0.0.1:11401/v1/models
```

## Step 7 — Run as a scheduled task (production mode)

> **Foreground = debug.** Running the shim directly shows live logs and is useful for troubleshooting, but it dies when the console closes. **Task Scheduler is the production method** — it survives logoff, auto-restarts on failure, and runs without a visible window. A `--daemon` flag is available as a lighter alternative that self-backgrounds and logs to file, but it lacks Task Scheduler's restart-on-crash guarantees.

Create `C:\Users\%USERNAME%\turbofit\start-shim.bat`:

```bat
@echo off
cd /d C:\Users\%USERNAME%\turbofit
call .venv\Scripts\activate.bat
python scripts\turbohaul-shim.py --port 11401
```

Register with Task Scheduler:

```powershell
schtasks /Create /TN "TurboFitShim" /TR "C:\Users\%USERNAME%\turbofit\start-shim.bat" /SC ONLOGON /RL HIGHEST /F
schtasks /Run /TN "TurboFitShim"
```

## Step 8 — Connect Hermes Agent

### Option A: Direct (same machine)

```yaml
# ~/.hermes/config.yaml or profile config
custom_providers:
  - name: turbofit
    base_url: http://127.0.0.1:11401/v1
    api_key: turbofit-local

model:
  default: bonsai-27b-q1
  provider: custom:turbofit
  context_length: 262144
  max_tokens: 8192
```

### Option B: Remote (Hermes on Linux, GPU on Windows)

Windows firewall may block inbound connections. Use an SSH tunnel from the Linux host:

```bash
# On the Linux host
ssh -L 127.0.0.1:11401:127.0.0.1:11401 user@<windows-ip> -N -f
```

For persistence, create a systemd service:

```ini
# /etc/systemd/system/turbofit-tunnel.service
[Unit]
Description=TurboFit SSH Tunnel
After=network-online.target

[Service]
ExecStart=/usr/bin/ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:11401:127.0.0.1:11401 user@<windows-ip>
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then point Hermes at `http://127.0.0.1:11401/v1` as in Option A.

## Step 9 — Use as fallback

Add TurboFit as the last fallback in your Hermes config so cloud models are tried first:

```yaml
fallback_providers:
  - provider: custom:turbofit
    model: bonsai-27b-q1
    base_url: http://127.0.0.1:11401/v1
    api_key: turbofit-local
    context_length: 262144
    max_tokens: 8192
```

## VRAM Sizing Reference (RTX 3090 24 GB)

| Context | KV Cache (q4_0) | + Weights | Total | Headroom |
|---------|-----------------|-----------|-------|----------|
| 65K | ~2.4 GB | +3.8 GB | ~7 GB | 17 GB |
| 262K | ~9.4 GB | +3.8 GB | ~14 GB | 10 GB |
| 1M | ~38 GB | +3.8 GB | ~42 GB | ❌ won't fit |

## Idle Behavior

With `sleep_idle_seconds: 60` in the manifest, llama-server automatically:

1. **Stops compute** after 60 seconds of no requests (GPU utilization drops to ~0%)
2. **Releases VRAM** — model weights and KV cache are unloaded (observed: 13.3 GB → 3.7 GB on RTX 3090)
3. **Lazy-reloads** on the next request via the shim's health-probe readiness

This means TurboFit genuinely "fits around your computer" — your GPU is fully available for games, rendering, or other work when the model isn't actively serving requests.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `model failed to become ready` | `--flash-attn` needs a value in newer builds | Set `"flash_attn": "on"` in manifest |
| Empty response / `Expecting value` | Client sent `stream: true`, shim returned JSON | Update shim (now handles SSE conversion) |
| `exceeds context size` | `--parallel N` splits ctx per slot | Increase `ctx_size` or reduce `parallel` |
| GPU at 100% but no response | Context too large, zero compute headroom | Reduce `ctx_size` to leave ≥4 GB free |
| Process dies on SSH disconnect | Windows kills child processes | Use Task Scheduler (Step 7) |
| `os.sysconf` AttributeError | Running on Windows without the patch | Apply `hardware.py` Windows RAM fix |

## Performance (RTX 3090, Bonsai 27B Q1_0)

| Test | Prompt eval | Generation |
|------|-------------|------------|
| Short prompt | ~1,166 tok/s | ~57 tok/s |
| 89K tokens | 803 tok/s | 31 tok/s |
| 200K tokens | 509 tok/s | 16 tok/s |
