"""Pinned generic MLX lane for Apple Silicon."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .managed_engine import EngineLaunch


MLX_VERSION = "0.32.2"
MLX_LM_VERSION = "0.31.3"
MODEL_REPO = "orcarouter/Qwen3.8-27B-Uncensored-MLX"
MODEL_REVISION = "b4603df5fd2a51e7fed2560ee7090caa4e13e4b7"
MODEL_FAMILY = "qwen3-8-27b-uncensored-mlx-8bit"
MODEL_RELATIVE_PATH = Path("Qwen3.8-27B-Uncensored-MLX/8-bit")
MODEL_CONTEXT = 262_144
DEFAULT_RESPONSE_TOKENS = 32_768


def load_python_runtime(path: str | Path, runtime_id: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "turbofit.python-runtimes/v1":
        raise ValueError("unsupported Python runtime manifest")
    runtimes = payload.get("runtimes")
    if not isinstance(runtimes, dict) or runtime_id not in runtimes:
        raise KeyError(f"unknown Python runtime: {runtime_id}")
    runtime = runtimes[runtime_id]
    if not isinstance(runtime, dict):
        raise ValueError(f"invalid Python runtime: {runtime_id}")
    return runtime


def build_apple_mlx_launch(
    *,
    model_root: str | Path,
    runtime_root: str | Path,
    port: int = 18081,
) -> EngineLaunch:
    model_path = Path(model_root).expanduser().resolve() / MODEL_RELATIVE_PATH
    python = (
        Path(runtime_root).expanduser().resolve()
        / f"mlx-lm-{MLX_LM_VERSION}"
        / ".venv"
        / "bin"
        / "python"
    )
    command = (
        str(python),
        "-m",
        "mlx_lm",
        "server",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--max-tokens",
        str(DEFAULT_RESPONSE_TOKENS),
    )
    return EngineLaunch(
        engine_id="mlx",
        model_id=MODEL_FAMILY,
        model_path=model_path,
        port=port,
        context_length=MODEL_CONTEXT,
        bind_host="127.0.0.1",
        upstream_model_id=str(model_path),
        command=command,
    )
