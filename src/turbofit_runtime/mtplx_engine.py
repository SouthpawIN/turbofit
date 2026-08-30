"""MTPLX discovery, launch contracts, and telemetry normalization."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .managed_engine import EngineLaunch


MTPLX_SPEED_MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
MTPLX_QUALITY_MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
MTPLX_FLASH_BARE_SPEED = "Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed"
MTPLX_FLASH_OPTIMIZED_SPEED = "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
MTPLX_MIN_VERSION = "2.10.1"
DEFAULT_APP_SETTINGS = Path("~/Library/Application Support/MTPLX/settings.json")
DEFAULT_PORTS = (18082, 8000, 18083, 18084, 18085)


def canonical_mtplx_alias(model_path: str | Path) -> str:
    name = Path(model_path).name.lower()
    if "qwen3.8-27b-mtplx-optimized-speed" in name:
        return "qwen3-8-27b-mtplx-optimized-speed"
    if "qwen3.8-27b-mtplx-optimized-quality" in name:
        return "qwen3-8-27b-mtplx-optimized-quality"
    if "qwen3.8-flash-next-mtplx-bare-speed" in name:
        return "qwen3-8-flash-next-mtplx-bare-speed"
    if "qwen3.8-flash-next-mtplx-optimized-speed" in name:
        return "qwen3-8-flash-next-mtplx-optimized-speed"
    raise ValueError(f"unsupported MTPLX model path: {model_path}")


@dataclass(frozen=True)
class MtplxEndpoint:
    host: str
    port: int
    model_id: str
    model_path: str
    pid: int | None
    app_launch_id: str | None
    health: Mapping[str, Any]
    owned_by_turbofit: bool = False


def _json_get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _settings_port(path: Path) -> int | None:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
        port = int(payload.get("port") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return port if 1 <= port <= 65535 else None


def discover_mtplx(
    *,
    settings_path: str | Path = DEFAULT_APP_SETTINGS,
    ports: tuple[int, ...] = DEFAULT_PORTS,
    json_get: Callable[[str, float], Any] = _json_get,
) -> MtplxEndpoint | None:
    app_port = _settings_port(Path(settings_path))
    ordered = ((app_port,) if app_port else ()) + tuple(
        port for port in ports if port != app_port
    )
    for port in ordered:
        try:
            health = json_get(f"http://127.0.0.1:{port}/health", 1.5)
        except (OSError, ValueError):
            continue
        if not isinstance(health, Mapping) or health.get("ok") is not True:
            continue
        model_id = str(health.get("model") or "").strip()
        model_path = str(health.get("model_path") or "").strip()
        if not model_id or not model_path:
            continue
        startup = health.get("startup")
        startup = startup if isinstance(startup, Mapping) else {}
        pid = startup.get("pid")
        return MtplxEndpoint(
            host="127.0.0.1",
            port=port,
            model_id=model_id,
            model_path=model_path,
            pid=int(pid) if isinstance(pid, int) and pid > 0 else None,
            app_launch_id=str(startup.get("launch_id") or "") or None,
            health=health,
        )
    return None


def build_mtplx_launch(
    *,
    executable: str | Path,
    model_path: str | Path,
    model_repo: str,
    model_id: str,
    port: int,
) -> EngineLaunch:
    if model_repo not in {
        MTPLX_SPEED_MODEL,
        MTPLX_QUALITY_MODEL,
        MTPLX_FLASH_BARE_SPEED,
        MTPLX_FLASH_OPTIMIZED_SPEED,
    }:
        raise ValueError(f"unsupported MTPLX model: {model_repo}")
    path = Path(model_path).expanduser().resolve()
    command = (
        str(Path(executable).expanduser().resolve()),
        "serve",
        "--model",
        str(path),
        "--profile",
        "turbo",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-auth",
        "--model-id",
        model_id,
        "--generation-mode",
        "mtp",
        "--fan-mode",
        "default",
        "--no-stats-footer",
    )
    return EngineLaunch(
        engine_id="mtplx",
        model_id=model_id,
        model_path=path,
        port=port,
        context_length=262_144,
        bind_host="127.0.0.1",
        upstream_model_id=model_id,
        command=command,
    )


def fetch_mtplx_metrics(
    endpoint: MtplxEndpoint,
    *,
    json_get: Callable[[str, float], Any] = _json_get,
) -> Mapping[str, Any]:
    try:
        payload = json_get(f"http://{endpoint.host}:{endpoint.port}/metrics", 2.0)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def mtplx_telemetry(
    health: Mapping[str, Any],
    metrics_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scheduler = health.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, Mapping) else {}
    memory = health.get("memory_plan")
    memory = memory if isinstance(memory, Mapping) else {}
    payload = metrics_payload if isinstance(metrics_payload, Mapping) else {}
    metrics = payload.get("latest")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    acceptance = metrics.get("acceptance_rate")
    if acceptance is None:
        acceptance = metrics.get("mtp_acceptance_rate")
    return {
        "engine": "mtplx",
        "model": health.get("model"),
        "generation_mode": health.get("generation_mode"),
        "depth": metrics.get("request_effective_mtp_depth", health.get("depth")),
        "scheduler_mode": scheduler.get("mode"),
        "active_requests": scheduler.get("active_requests"),
        "context_length": memory.get("context_window_resolved"),
        "model_weights_bytes": memory.get("model_weights_bytes"),
        "peak_memory_bytes": metrics.get("peak_memory_bytes"),
        "decode_tokens_per_second": metrics.get("decode_tok_s"),
        "request_tokens_per_second": metrics.get("request_tok_s"),
        "acceptance_rate": acceptance,
    }
