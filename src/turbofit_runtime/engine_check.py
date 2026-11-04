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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

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
    os: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    min_driver: int = 0
    status: str = "released"


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


def _validate_engine_matrix(matrix: Mapping[str, Any]) -> None:
    """Reject a serve matrix that cannot be turned into an engine registry."""
    probe_order = matrix.get("probe_order")
    engines = matrix.get("engines")
    if not isinstance(probe_order, (list, tuple)) or not isinstance(engines, Mapping):
        raise ValueError("engine-serve matrix must define probe_order and engines")
    unknown = [str(engine_id) for engine_id in probe_order if engine_id not in engines]
    if unknown:
        raise ValueError(f"probe_order lists engines missing from engines: {', '.join(unknown)}")
    if set(probe_order) != set(engines):
        raise ValueError("probe_order and engines must list exactly the same engines")


def engine_specs(matrix: dict[str, Any] | None = None) -> tuple[EngineSpec, ...]:
    """Build the engine registry from the data registry (the serve matrix).

    Registering a new inference engine is a data-only edit: add it to
    ``engines`` and ``probe_order`` in references/engine-serve-matrix.json.
    Engines without a hand-written branch in :func:`_compatibility` are
    classified by the generic data-driven fallback from their declared
    ``os``, ``backends`` and ``min_driver``.
    """
    if matrix is None:
        matrix = load_serve_matrix()
    _validate_engine_matrix(matrix)
    specs: list[EngineSpec] = []
    for engine_id in matrix["probe_order"]:
        entry = matrix["engines"][engine_id]
        detect = entry.get("detect") or {}
        specs.append(
            EngineSpec(
                engine_id=str(engine_id),
                display_name=str(entry.get("display_name") or engine_id),
                openai_port=int(entry["openai_port"]),
                install_hint=str(entry.get("install_hint") or "See the engine source."),
                commands=tuple(str(command) for command in detect.get("commands") or ()),
                modules=tuple(str(module) for module in detect.get("modules") or ()),
                source_url=str(entry.get("source") or ""),
                source_ref=str(entry.get("ref") or ""),
                source_revision=str(entry.get("revision") or ""),
                endpoint_discoverable=bool(entry.get("endpoint_discoverable", False)),
                os=tuple(str(value) for value in entry.get("os") or ()),
                backends=tuple(str(value) for value in entry.get("backends") or ()),
                min_driver=int(entry.get("min_driver") or 0),
                status=str(entry.get("status") or "released"),
            )
        )
    return tuple(specs)


def engine_status(engine_id: str, matrix: dict[str, Any] | None = None) -> str:
    """Return an engine's qualification status: 'released' or 'candidate'."""
    payload = matrix if matrix is not None else load_serve_matrix()
    entry = (payload.get("engines") or {}).get(engine_id)
    if not entry:
        raise ValueError(f"unknown engine: {engine_id}")
    status = str(entry.get("status") or "released")
    if status not in _ENGINE_STATUSES:
        raise ValueError(f"invalid engine status for {engine_id}: {status!r}")
    return status


def set_engine_status(matrix_path: str | Path | None, engine_id: str, status: str) -> dict[str, Any]:
    """Flip an engine's qualification status inside the data registry.

    This is the promotion hook for benchmark/qualification gates: when a
    candidate engine clears its gate, its status in
    references/engine-serve-matrix.json flips from "candidate" to
    "released" — a data edit, not a code change.
    """
    if status not in _ENGINE_STATUSES:
        raise ValueError(f"status must be one of {_ENGINE_STATUSES}, got {status!r}")
    path = Path(matrix_path) if matrix_path is not None else _SERVE_MATRIX
    payload = load_serve_matrix(path)
    entry = (payload.get("engines") or {}).get(engine_id)
    if not entry:
        raise ValueError(f"unknown engine: {engine_id}")
    entry["status"] = status
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


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
    matrix: dict[str, Any] | None = None,
) -> tuple[EngineCompatibility, ...]:
    """Classify every engine without installing or launching anything."""
    return tuple(
        _compatibility(spec, hardware, driver_major=driver_major, is_wsl=is_wsl)
        for spec in engine_specs(matrix)
    )


def probe_engines(
    hardware: HardwareFingerprint,
    *,
    driver_major: int | None = None,
    is_wsl: bool = False,
    which: Callable[[str], str | None] = shutil.which,
    module_available: Callable[[str], bool] | None = None,
    endpoint_ready: Callable[[str], bool] | None = None,
    matrix: dict[str, Any] | None = None,
) -> tuple[EngineProbe, ...]:
    """Report compatibility, installation, and live endpoint state for all engines."""
    find_module = module_available or _module_available
    ready = endpoint_ready or _endpoint_ready
    specs = engine_specs(matrix)
    compatibility = {
        item.engine_id: item
        for item in check_engines(hardware, driver_major=driver_major, is_wsl=is_wsl, matrix=matrix)
    }
    reports: list[EngineProbe] = []
    for spec in specs:
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
    elif spec.engine_id == "mlx":
        compatible = os_name == "darwin" and architecture in {"arm64", "aarch64"} and "metal" in backends
        reason = "Apple Silicon Metal runtime" if compatible else "MLX requires Apple Silicon macOS with Metal"
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
    elif spec.engine_id == "strata":
        # Strata 0.2.10 ships signed Windows installers, a notarized macOS dmg
        # (Metal), Linux x86_64 and Linux ARM64 (DGX Spark), plus Intel SYCL --
        # and 8 GB cards start at 128K context. The old Linux-x86_64-CUDA-only
        # rule refused machines Strata now runs on.
        accelerator = bool(backends & {"cuda", "rocm", "metal", "sycl", "vulkan"})
        linux_or_dgx = os_name == "linux" and architecture in {
            "x86_64", "amd64", "aarch64", "arm64",
        }
        windows = os_name == "windows" and architecture in {"x86_64", "amd64", "arm64"}
        mac = os_name == "darwin" and architecture in {"arm64", "aarch64"} and "metal" in backends
        if not (linux_or_dgx or windows or mac):
            reason = "Strata requires Linux (x86_64/ARM64), Windows or Apple Silicon"
        elif not accelerator:
            reason = "Strata needs a CUDA, ROCm, Metal, SYCL or Vulkan accelerator"
        else:
            compatible = True
            reason = "Niko1221 Strata 0.2.10 (DFlash2 drafter, expert tiering, 8 GB+ cards)"
    elif spec.engine_id == "prism-llama.cpp":
        compatible = os_name in {"linux", "windows", "darwin"}
        reason = "PrismML llama.cpp fork (CUDA, Metal, CPU)" if compatible else "Prism llama.cpp requires Linux, Windows, or macOS"
    elif spec.engine_id in {"salt-core", "saliency"}:
        mode = "wsl" if is_wsl else ("wsl" if os_name == "windows" else "native")
        if os_name != "linux":
            reason = f"{spec.display_name} currently requires Linux/WSL"
        elif "cuda" not in backends:
            reason = f"{spec.display_name} currently requires an NVIDIA CUDA GPU"
        elif spec.engine_id == "salt-core":
            compatible = True
            reason = "SALT Core candidate (KV offload + disk streaming, 4 GB VRAM floor)"
        else:
            compatible = True
            reason = "Saliency candidate (dynamic token pruning, 4 GB VRAM floor)"
    else:
        # Generic data-driven fallback: engines registered only in the serve
        # matrix (no hand-written branch above) are classified from their
        # declared os/backends and min_driver.
        declared_os = set(spec.os)
        if declared_os and os_name not in declared_os:
            if os_name == "windows" and "linux" in declared_os:
                mode = "wsl"
                compatible = True
                reason = f"{spec.display_name} requires Linux (WSL)"
            else:
                reason = f"{spec.display_name} requires {'/'.join(sorted(declared_os))}"
        elif (
            spec.min_driver
            and "cuda" in set(spec.backends)
            and (driver_major is None or driver_major < spec.min_driver)
        ):
            reason = f"{spec.display_name} requires NVIDIA driver r{spec.min_driver} or newer"
        elif spec.backends and not (set(spec.backends) & backends):
            host_backends = ", ".join(sorted(backends)) or "CPU only"
            reason = f"{spec.display_name} requires one of {', '.join(sorted(spec.backends))} but the host has {host_backends}"
        else:
            compatible = True
            reason = f"{spec.display_name} supports {os_name}"

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


_SERVE_MATRIX = Path(__file__).resolve().parents[2] / "references/engine-serve-matrix.json"
_ENGINE_STATUSES = ("released", "candidate")


def load_serve_matrix(path: str | Path | None = None) -> dict:
    """Load the engine-serve matrix, the single data registry for engines."""
    target = Path(path) if path is not None else _SERVE_MATRIX
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"engine-serve matrix missing: {target}") from error


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
    if engine_id in (pair.get("low_vram_dense_candidates") or []):
        floor = engine.get("min_vram_gb", 4)
        return "candidate", f"candidate low-VRAM dense engine ({floor} GB VRAM floor) — qualification gate pending"
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
        elif status == "candidate":
            audition = "candidate"
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
    rank = {"ready": 0, "installed": 1, "not-installed": 2, "incompatible": 3, "ineligible": 4, "candidate": 5}
    rows.sort(key=lambda item: (rank.get(str(item["audition"]), 9), str(item["engine_id"])))
    return rows
