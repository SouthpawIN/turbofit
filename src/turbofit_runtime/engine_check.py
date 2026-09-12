"""Cross-platform capability and availability registry for TurboFit Check engines."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Callable

from .hardware import HardwareFingerprint


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    display_name: str
    openai_port: int
    install_hint: str
    commands: tuple[str, ...]
    modules: tuple[str, ...]
    source_url: str = ""
    source_ref: str = ""
    source_revision: str = ""
    endpoint_discoverable: bool = False


@dataclass(frozen=True)
class EngineCompatibility:
    engine_id: str
    display_name: str
    compatible: bool
    support_mode: str
    reason: str
    openai_port: int
    install_hint: str
    source_url: str
    source_ref: str
    source_revision: str


@dataclass(frozen=True)
class EngineProbe(EngineCompatibility):
    installed: bool
    running: bool
    eligible: bool
    executable: str | None


_SPECS = (
    EngineSpec("llama.cpp", "llama.cpp", 8080, "Install Turbofit's pinned native llama.cpp runtime.", ("llama-server", "llama-server.exe"), ()),
    EngineSpec("mlx", "MLX", 8081, "Install mlx-lm on Apple Silicon macOS.", ("mlx_lm.server",), ("mlx_lm",)),
    EngineSpec(
        "mtplx",
        "MTPLX",
        8000,
        "Install MTPLX 2.10.1 or newer on Apple Silicon macOS.",
        ("mtplx",),
        ("mtplx",),
        "https://github.com/youssofal/MTPLX",
        "v2.10.1",
        "557e637a3aceba4cad7975582979e434aea7c092",
        True,
    ),
    EngineSpec("sglang", "SGLang", 30000, "Install SGLang on native Linux or inside WSL.", ("sglang",), ("sglang",)),
    EngineSpec("vllm", "vLLM", 8000, "Install vLLM on Linux/WSL or vLLM-Metal on Apple Silicon.", ("vllm",), ("vllm",)),
    EngineSpec("freetoken", "FreeToken", 1919, "Install freetoken[accel] on Linux x86_64 with CUDA 13.", ("ft",), ("freetoken",)),
    EngineSpec(
        "turbohaul-manager",
        "Turbohaul Manager",
        11401,
        "Install MrTrenchTrucker/turbohaul-manager from its pinned GitHub release.",
        ("turbohaul-manager",),
        ("turbohaul",),
        "https://github.com/MrTrenchTrucker/turbohaul-manager",
        "v0.7.0",
        "905b4506883313b17e1d4e0480a8e6ca6c63399b",
        True,
    ),
)


def engine_specs() -> tuple[EngineSpec, ...]:
    """Return every engine TurboFit Check reports, in stable preference order."""
    return _SPECS


def parse_driver_major(value: str) -> int | None:
    match = re.match(r"\s*(\d{3,})\.", value)
    return int(match.group(1)) if match else None


def is_wsl_release(value: str) -> bool:
    lowered = value.lower()
    return "microsoft" in lowered or "wsl" in lowered


def detect_driver_major() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (FileNotFoundError, OSError, subprocess.SubprocessError, IndexError):
        return None
    return parse_driver_major(value)


def detect_wsl() -> bool:
    return is_wsl_release(platform.release())


def check_engines(
    hardware: HardwareFingerprint,
    *,
    driver_major: int | None = None,
    is_wsl: bool = False,
) -> tuple[EngineCompatibility, ...]:
    """Classify every engine without installing or launching anything."""
    return tuple(
        _compatibility(spec, hardware, driver_major=driver_major, is_wsl=is_wsl)
        for spec in _SPECS
    )


def probe_engines(
    hardware: HardwareFingerprint,
    *,
    driver_major: int | None = None,
    is_wsl: bool = False,
    which: Callable[[str], str | None] = shutil.which,
    module_available: Callable[[str], bool] | None = None,
    endpoint_ready: Callable[[str], bool] | None = None,
) -> tuple[EngineProbe, ...]:
    """Report compatibility, installation, and live endpoint state for all engines."""
    find_module = module_available or _module_available
    ready = endpoint_ready or _endpoint_ready
    compatibility = {
        item.engine_id: item
        for item in check_engines(hardware, driver_major=driver_major, is_wsl=is_wsl)
    }
    reports: list[EngineProbe] = []
    for spec in _SPECS:
        base = compatibility[spec.engine_id]
        executable = next((path for command in spec.commands if (path := which(command))), None)
        installed = executable is not None or any(find_module(module) for module in spec.modules)
        running = bool(
            base.compatible
            and (installed or spec.endpoint_discoverable)
            and ready(f"http://127.0.0.1:{spec.openai_port}")
        )
        installed = installed or running
        reports.append(
            EngineProbe(
                **base.__dict__,
                installed=installed,
                running=running,
                eligible=base.compatible and installed,
                executable=executable,
            )
        )
    return tuple(reports)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _endpoint_ready(base_url: str) -> bool:
    for path in ("/health", "/v1/models", "/status", "/api/tags"):
        try:
            with urllib.request.urlopen(base_url + path, timeout=2) as response:
                if response.status != 200:
                    continue
                if path in {"/health", "/status"}:
                    return True
                payload = json.load(response)
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue
    return False


def _compatibility(
    spec: EngineSpec,
    hardware: HardwareFingerprint,
    *,
    driver_major: int | None,
    is_wsl: bool,
) -> EngineCompatibility:
    os_name = hardware.os.lower()
    architecture = hardware.architecture.lower()
    backends = set(hardware.backends)
    mode = "wsl" if is_wsl else "native"
    compatible = False
    reason = "unsupported platform"

    if spec.engine_id == "llama.cpp":
        compatible = os_name in {"linux", "windows", "darwin"}
        reason = "native cross-platform runtime" if compatible else reason
    elif spec.engine_id in {"mlx", "mtplx"}:
        compatible = os_name == "darwin" and architecture in {"arm64", "aarch64"} and "metal" in backends
        label = "MTPLX" if spec.engine_id == "mtplx" else "MLX"
        reason = f"Apple Silicon Metal {label} runtime" if compatible else f"{label} requires Apple Silicon macOS with Metal"
    elif spec.engine_id == "sglang":
        compatible = os_name == "linux" and bool(backends & {"cuda", "rocm"})
        reason = "Linux accelerator runtime" if compatible else "SGLang requires Linux/WSL with a supported accelerator"
        if os_name == "windows" and not is_wsl:
            mode = "wsl"
    elif spec.engine_id == "vllm":
        linux = os_name == "linux" and bool(backends & {"cuda", "rocm"})
        metal = os_name == "darwin" and architecture in {"arm64", "aarch64"} and "metal" in backends
        compatible = linux or metal
        reason = "vLLM-Metal runtime" if metal else ("Linux accelerator runtime" if linux else "vLLM requires Linux/WSL or Apple Silicon vLLM-Metal")
        if os_name == "windows" and not is_wsl:
            mode = "wsl"
    elif spec.engine_id == "freetoken":
        mode = "wsl" if is_wsl else ("wsl" if os_name == "windows" else "native")
        if os_name != "linux":
            reason = "FreeToken currently requires Linux/WSL"
        elif architecture not in {"x86_64", "amd64"}:
            reason = "FreeToken currently requires Linux x86_64"
        elif "cuda" not in backends:
            reason = "FreeToken currently requires an NVIDIA CUDA GPU"
        elif driver_major is None or driver_major < 580:
            reason = "FreeToken currently requires NVIDIA driver r580 or newer and CUDA 13"
        else:
            compatible = True
            reason = "Linux x86_64 NVIDIA CUDA 13 runtime"
    elif spec.engine_id == "turbohaul-manager":
        mode = "wsl" if is_wsl else ("wsl" if os_name == "windows" else "native")
        if os_name != "linux":
            reason = "Turbohaul Manager currently requires Linux/WSL"
        elif architecture not in {"x86_64", "amd64"}:
            reason = "Turbohaul Manager currently requires Linux x86_64"
        elif "cuda" not in backends:
            reason = "Turbohaul Manager currently requires an NVIDIA CUDA GPU"
        else:
            compatible = True
            reason = "MrTrenchTrucker Turbohaul Manager on Linux NVIDIA CUDA"

    return EngineCompatibility(
        engine_id=spec.engine_id,
        display_name=spec.display_name,
        compatible=compatible,
        support_mode=mode,
        reason=reason,
        openai_port=spec.openai_port,
        install_hint=spec.install_hint,
        source_url=spec.source_url,
        source_ref=spec.source_ref,
        source_revision=spec.source_revision,
    )


def load_serve_matrix(path: str | None = None) -> dict:
    from pathlib import Path

    target = Path(path) if path else Path(__file__).resolve().parents[2] / "references/engine-serve-matrix.json"
    return json.loads(target.read_text(encoding="utf-8"))


def pair_engine_eligibility(main_alias: str, engine_id: str, *, matrix: dict | None = None) -> tuple[str, str]:
    """Return (status, reason) for serving this model family on this engine."""
    payload = matrix or load_serve_matrix()
    pair = (payload.get("pairs") or {}).get(main_alias) or {}
    engine = (payload.get("engines") or {}).get(engine_id) or {}
    key = "maple_gguf" if "maple" in main_alias else ("ornith_gguf" if "ornith" in main_alias else "qwen38_gguf")
    if engine_id in (pair.get("eligible_engines") or []):
        return "eligible", str(engine.get(key) or "researched GGUF path")
    if engine_id == pair.get("apple_alternate"):
        return "alternate", str(engine.get(key) or "Apple MLX alternate")
    if engine_id in (pair.get("hf_engines") or []):
        return "hf-only", str(engine.get(key) or "requires HuggingFace weights, not this GGUF")
    if engine_id in (pair.get("ineligible_engines") or []):
        return "ineligible", str(engine.get(key) or "no researched serve path")
    return "unknown", "no researched serve path for this pair"


def audition_pair(
    hardware: HardwareFingerprint,
    *,
    main_alias: str,
    aux_alias: str = "auto",
    context: int = 65536,
    probes: tuple[EngineProbe, ...] | None = None,
    matrix: dict | None = None,
) -> list[dict[str, object]]:
    """Audition every Check engine against one main/aux pair using researched serve paths."""
    payload = matrix or load_serve_matrix()
    reports = probes or probe_engines(hardware)
    rows: list[dict[str, object]] = []
    for probe in reports:
        status, reason = pair_engine_eligibility(main_alias, probe.engine_id, matrix=payload)
        if not probe.compatible:
            audition = "incompatible"
        elif status == "ineligible":
            audition = "ineligible"
        elif not probe.installed:
            audition = "not-installed"
        elif probe.running:
            audition = "ready"
        else:
            audition = "installed"
        rows.append(
            {
                "engine_id": probe.engine_id,
                "display_name": probe.display_name,
                "pair_main": main_alias,
                "pair_aux": aux_alias,
                "context": context,
                "compatible": probe.compatible,
                "installed": probe.installed,
                "running": probe.running,
                "eligible": probe.eligible and status in {"eligible", "alternate"},
                "pair_status": status,
                "audition": audition,
                "reason": reason if status != "eligible" else probe.reason,
                "serve_note": reason,
                "openai_port": probe.openai_port,
                "support_mode": probe.support_mode,
            }
        )
    rank = {"ready": 0, "installed": 1, "not-installed": 2, "incompatible": 3, "ineligible": 4}
    rows.sort(key=lambda item: (rank.get(str(item["audition"]), 9), str(item["engine_id"])))
    return rows
