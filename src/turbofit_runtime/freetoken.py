"""Optional FreeToken candidate runtime support for NVIDIA MoE models.

FreeToken is deliberately not an Auto authority here. This adapter provides
strict compatibility probing, loopback command construction, and health/stats
inspection so exact model + hardware campaigns can validate it before any lane
is promoted.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

FREETOKEN_REPOSITORY = "https://github.com/FlashML-org/FreeToken.git"
FREETOKEN_REVISION = "0ab982f10905fa775962a4eddcb44caa50065251"
FREETOKEN_VERSION = "0.1.2"
FREETOKEN_DEFAULT_PORT = 1919
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True)
class FreeTokenCompatibility:
    compatible: bool
    status: str
    blockers: tuple[str, ...]
    requirements: tuple[str, ...]


def _major(value: str) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _cuda_version(value: str) -> tuple[int, int] | None:
    match = _VERSION_RE.search(str(value))
    return (int(match.group(1)), int(match.group(2))) if match else None


def evaluate_freetoken_compatibility(
    *,
    os_name: str,
    architecture: str,
    accelerator_backends: tuple[str, ...],
    driver_version: str,
    nvcc_version: str,
) -> FreeTokenCompatibility:
    blockers: list[str] = []
    if os_name.lower() != "linux":
        blockers.append("Linux required")
    if architecture.lower() not in {"x86_64", "amd64"}:
        blockers.append("x86_64 required")
    if "cuda" not in {item.lower() for item in accelerator_backends}:
        blockers.append("NVIDIA CUDA backend required")
    driver = _major(driver_version)
    if driver is None or driver < 580:
        blockers.append("NVIDIA driver 580+ required")
    cuda = _cuda_version(nvcc_version)
    if cuda is None or cuda < (13, 0):
        blockers.append("CUDA toolkit 13+ required")
    return FreeTokenCompatibility(
        compatible=not blockers,
        status="candidate" if not blockers else "blocked",
        blockers=tuple(blockers),
        requirements=(
            "Linux x86_64",
            "NVIDIA RTX CUDA GPU",
            "NVIDIA driver 580+",
            "CUDA toolkit 13+ with nvcc",
            "text-only supported MoE checkpoint",
            "on-box ft bench bw and Turbofit physical campaign",
        ),
    )


def probe_freetoken_compatibility() -> FreeTokenCompatibility:
    backends: tuple[str, ...] = ("cuda",) if shutil.which("nvidia-smi") else ()
    driver = ""
    if backends:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            driver = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    nvcc = ""
    executable = shutil.which("nvcc")
    if executable:
        result = subprocess.run(
            [executable, "--version"], text=True, capture_output=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
            nvcc = match.group(1) if match else result.stdout
    return evaluate_freetoken_compatibility(
        os_name=platform.system().lower(),
        architecture=platform.machine(),
        accelerator_backends=backends,
        driver_version=driver,
        nvcc_version=nvcc,
    )


def build_freetoken_command(
    *,
    binary: str,
    model: str,
    alias: str,
    port: int,
    context: int,
    memory_ratio: float,
    moe: bool,
    vision: bool,
) -> tuple[str, ...]:
    if not moe:
        raise ValueError("FreeToken candidate integration is restricted to MoE models")
    if vision:
        raise ValueError("FreeToken candidate integration is text-only")
    if not binary or not model or not _ALIAS_RE.fullmatch(alias):
        raise ValueError("invalid FreeToken binary, model, or alias")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("FreeToken port must be in 1..65535")
    if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
        raise ValueError("FreeToken context must be positive")
    if isinstance(memory_ratio, bool) or not isinstance(memory_ratio, (int, float)) or not 0 < memory_ratio < 1:
        raise ValueError("FreeToken memory ratio must be between zero and one")
    return (
        binary,
        "serve",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        alias,
        "--max-seq-len-override",
        str(context),
        "--memory-ratio",
        str(memory_ratio),
        "--moe-backend",
        "auto",
    )


Transport = Callable[[str, str, float], tuple[int, dict[str, Any]]]


def _transport(method: str, url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = {"error": str(exc)}
        return exc.code, payload


class FreeTokenClient:
    def __init__(self, base_url: str = "http://127.0.0.1:1919", *, transport: Transport = _transport) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("FreeToken client must use a loopback HTTP endpoint")
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    def _get(self, path: str, *, timeout: float = 5.0) -> dict[str, Any]:
        status, payload = self.transport("GET", self.base_url + path, timeout)
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"FreeToken endpoint failed: {path} ({status})")
        return payload

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> list[dict[str, Any]]:
        payload = self._get("/v1/models")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("FreeToken /v1/models returned invalid data")
        return [dict(item) for item in rows if isinstance(item, dict)]

    def stats(self) -> dict[str, Any]:
        return self._get("/v1/stats")
