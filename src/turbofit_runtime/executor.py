"""Execute one resolved Main:Aux recipe and return a normalized benchmark."""
from __future__ import annotations

import json
import hashlib
import platform
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

from .campaign import RawBenchmark
from .gpu import GPUClearEvent
from .hardware import HardwareFingerprint, probe_hardware
from .recipes import RecipeBook, ResolvedComponent, ResolvedRecipe
from .schema import MatrixRow


class RuntimeBackend(Protocol):
    def start(self, component: ResolvedComponent): ...
    def wait_ready(self, component: ResolvedComponent, handle) -> dict: ...
    def route(self, recipe: ResolvedRecipe, handles: dict[str, object]) -> dict: ...
    def infer(self, role: str, recipe: ResolvedRecipe) -> dict: ...
    def peak_gpu_mb(self) -> dict[int, int]: ...
    def stop(self, component: ResolvedComponent, handle) -> None: ...


def production_recipe_sha256(recipe: ResolvedRecipe, row_payload: dict) -> str:
    payload = {
        "validation_protocol": "turbofit.catalog-physical/v4",
        "physical_fingerprint_schema": "turbofit.physical-hardware/v1",
        "smoke_max_tokens": 128,
        "shared_main_execution": "serial",
        "production_service_lease": "controller-only+continuous-production-gateway+isolated-campaign-gateway/v1",
        "row": row_payload,
        "profile_name": recipe.profile_name,
        "main_alias": recipe.main_alias,
        "aux_alias": recipe.aux_alias,
        "aux_mode": recipe.aux_mode,
        "components": [
            {
                "role": component.role,
                "family": component.family,
                "alias": component.alias,
                "method": component.method,
                "gpu": component.gpu,
                "port": component.port,
                "command": list(component.command),
                "model_path": component.model_path,
                "projector_path": component.projector_path,
            }
            for component in recipe.components
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def capture_physical_hardware() -> dict:
    hardware: HardwareFingerprint = probe_hardware()
    devices = [asdict(device) for device in hardware.devices]
    drivers: dict[str, str] = {}
    if any(device.vendor == "nvidia" for device in hardware.devices):
        command = [
            "nvidia-smi", "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
        for line in completed.stdout.splitlines():
            uuid, separator, version = line.partition(",")
            if not separator or not uuid.strip() or not version.strip():
                raise RuntimeError(f"invalid NVIDIA driver inventory row: {line!r}")
            drivers[uuid.strip()] = version.strip()
        expected = {device.uuid for device in hardware.devices if device.vendor == "nvidia"}
        if set(drivers) != expected:
            raise RuntimeError("NVIDIA driver inventory does not match physical accelerator inventory")
    if any(device.vendor == "amd" for device in hardware.devices):
        completed = subprocess.run(
            ["rocm-smi", "--showdriverversion", "--json"],
            text=True, capture_output=True, check=True, timeout=30,
        )
        if not completed.stdout.strip():
            raise RuntimeError("empty ROCm driver inventory")
        drivers["rocm-smi"] = completed.stdout.strip()
    if any(device.vendor == "apple" for device in hardware.devices):
        completed = subprocess.run(
            ["sw_vers", "-productVersion"],
            text=True, capture_output=True, check=True, timeout=30,
        )
        if not completed.stdout.strip():
            raise RuntimeError("empty Apple platform version")
        drivers["macos"] = completed.stdout.strip()
        drivers["kernel"] = platform.release()
    if hardware.devices and not drivers:
        raise RuntimeError("accelerator driver revision is unavailable")
    fingerprint = {
        "os": hardware.os,
        "architecture": hardware.architecture,
        "system_ram_mb": hardware.system_ram_mb,
        "topology_key": hardware.topology_key,
        "recommendation_key": hardware.recommendation_key,
        "accelerator_memory_mb": hardware.total_vram_mb,
        "unified_memory": hardware.unified_memory,
        "devices": devices,
        "drivers": drivers,
    }
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "turbofit.physical-hardware/v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "fingerprint_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
    }


def validate_physical_hardware(payload: dict) -> None:
    if payload.get("schema") != "turbofit.physical-hardware/v1":
        raise RuntimeError("invalid physical hardware evidence schema")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("physical hardware fingerprint payload is missing")
    if fingerprint.get("devices") and not fingerprint.get("drivers"):
        raise RuntimeError("physical hardware driver revision is missing")
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if payload.get("fingerprint_sha256") != expected:
        raise RuntimeError("physical hardware fingerprint checksum mismatch")


def physical_evidence_current(record: dict, live_fingerprint: str) -> bool:
    raw_path = Path(str(record.get("raw_result_path", "")))
    expected_sha = str(record.get("raw_result_sha256", ""))
    expected_fingerprint = str(record.get("physical_fingerprint", ""))
    if (
        not raw_path.is_file()
        or not expected_sha
        or expected_fingerprint != live_fingerprint
    ):
        return False
    raw_bytes = raw_path.read_bytes()
    if "sha256:" + hashlib.sha256(raw_bytes).hexdigest() != expected_sha:
        return False
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError:
        return False
    return str((payload.get("physical_hardware") or {}).get("fingerprint_sha256", "")) == expected_fingerprint


class LocalPairExecutor:
    def __init__(
        self, *, recipes: RecipeBook, backend: RuntimeBackend, result_dir: Path,
        physical_hardware_probe: Callable[[], dict] = capture_physical_hardware,
    ) -> None:
        self.recipes = recipes
        self.backend = backend
        self.result_dir = result_dir
        self.physical_hardware_probe = physical_hardware_probe
        self._current_physical_fingerprint: str | None = None

    def evidence_is_current(self, record: dict) -> bool:
        return physical_evidence_current(record, self.current_physical_fingerprint())

    def current_physical_fingerprint(self) -> str:
        if self._current_physical_fingerprint is None:
            current = self.physical_hardware_probe()
            validate_physical_hardware(current)
            self._current_physical_fingerprint = str(current["fingerprint_sha256"])
        return self._current_physical_fingerprint

    def prepare(self, item: MatrixRow) -> None:
        acquire = getattr(self.backend, "acquire_campaign_lease", None)
        if acquire is not None:
            acquire()

    def finish(self, item: MatrixRow) -> None:
        release = getattr(self.backend, "release_campaign_lease", None)
        if release is not None:
            release()

    def record_campaign_failure(
        self, item: MatrixRow, error: str, raw_result_path: str | None,
        before: GPUClearEvent | None, after: GPUClearEvent | None,
    ) -> str:
        if "failure_evidence=" in error:
            return error.rsplit("failure_evidence=", 1)[-1].rstrip("')\"")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        failure_dir = self.result_dir / "failures" / item.id / timestamp
        failure_dir.mkdir(parents=True, exist_ok=False)
        logs = []
        for source in sorted(self.result_dir.glob("campaign-*.log")):
            destination = failure_dir / source.name
            shutil.copy2(source, destination)
            logs.append({
                "path": str(destination),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            })
        raw_record = None
        if raw_result_path:
            source = Path(raw_result_path)
            if source.is_file():
                destination = failure_dir / "raw-result.json"
                shutil.copy2(source, destination)
                raw_record = {
                    "path": str(destination),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
        path = failure_dir / "failure.json"
        path.write_text(json.dumps({
            "schema": "turbofit.catalog-failure/v1",
            "timestamp": timestamp,
            "row": {
                "id": item.id, "main": item.main, "auxiliary": item.aux,
                "context": item.context,
            },
            "error": error,
            "gpu_clear_before": asdict(before) if before is not None else None,
            "gpu_clear_after": asdict(after) if after is not None else None,
            "raw_result": raw_record,
            "logs": logs,
        }, indent=2) + "\n", encoding="utf-8")
        return str(path)

    def execute(self, item: MatrixRow) -> RawBenchmark:
        return self._execute(
            recipe=self.recipes.resolve(item), item_id=item.id,
            context=item.context, row_payload=item.to_dict(),
        )

    def execute_catalog(self, item: dict) -> RawBenchmark:
        return self._execute(
            recipe=self.recipes.resolve_catalog_configuration(item),
            item_id=str(item["id"]), context=int(item["context"]), row_payload=dict(item),
        )

    def recipe_sha256(self, item: MatrixRow) -> str:
        return production_recipe_sha256(self.recipes.resolve(item), item.to_dict())

    def catalog_recipe_sha256(self, item: dict) -> str:
        return production_recipe_sha256(self.recipes.resolve_catalog_configuration(item), dict(item))

    def _execute(
        self, *, recipe: ResolvedRecipe, item_id: str,
        context: int, row_payload: dict,
    ) -> RawBenchmark:
        started: list[tuple[ResolvedComponent, object]] = []
        checks: dict[str, dict] = {}
        route: dict = {}
        results: dict[str, dict] = {}
        peak: dict[int, int] = {}
        failure: Exception | None = None
        failure_traceback = ""
        physical_hardware: dict = {}
        try:
            physical_hardware = self.physical_hardware_probe()
            validate_physical_hardware(physical_hardware)
            self._current_physical_fingerprint = str(physical_hardware["fingerprint_sha256"])
            for component in recipe.components:
                handle = self.backend.start(component)
                started.append((component, handle))
            for component, handle in started:
                checks[component.role] = self.backend.wait_ready(component, handle)
            route = self.backend.route(recipe, {component.role: handle for component, handle in started})
            if recipe.aux_mode == "shared-main":
                results = {role: self.backend.infer(role, recipe) for role in ("main", "aux")}
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {role: pool.submit(self.backend.infer, role, recipe) for role in ("main", "aux")}
                    results = {role: future.result() for role, future in futures.items()}
            peak = self.backend.peak_gpu_mb()
        except Exception as exc:
            failure = exc
            failure_traceback = traceback.format_exc()
        finally:
            for component, handle in reversed(started):
                self.backend.stop(component, handle)

        if failure is not None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            failure_dir = self.result_dir / "failures" / item_id / timestamp
            failure_dir.mkdir(parents=True, exist_ok=False)
            logs = []
            for component, handle in started:
                source_value = handle.get("log") if isinstance(handle, dict) else None
                source = Path(str(source_value)) if source_value else None
                if source and source.is_file():
                    destination = failure_dir / f"{component.role}.log"
                    shutil.copy2(source, destination)
                    logs.append({
                        "role": component.role,
                        "path": str(destination),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    })
            gateway_root = getattr(self.backend, "result_dir", None)
            gateway_log = Path(gateway_root) / "campaign-gateway.log" if gateway_root else None
            if gateway_log and gateway_log.is_file():
                destination = failure_dir / "gateway.log"
                shutil.copy2(gateway_log, destination)
                logs.append({
                    "role": "gateway",
                    "path": str(destination),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                })
            evidence_path = failure_dir / "failure.json"
            evidence_path.write_text(json.dumps({
                "schema": "turbofit.catalog-failure/v1",
                "timestamp": timestamp,
                "row": row_payload,
                "profile_name": recipe.profile_name,
                "error": repr(failure),
                "traceback": failure_traceback,
                "components": [
                    {
                        "role": component.role,
                        "family": component.family,
                        "alias": component.alias,
                        "method": component.method,
                        "gpu": component.gpu,
                        "port": component.port,
                        "command": list(component.command),
                        "model_path": component.model_path,
                        "projector_path": component.projector_path,
                    }
                    for component in recipe.components
                ],
                "checks": checks,
                "route": route,
                "results": results,
                "gpu_peak_mb": self.backend.peak_gpu_mb(),
                "physical_hardware": physical_hardware,
                "logs": logs,
            }, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"{failure!r}; failure_evidence={evidence_path}") from failure

        for role, check in checks.items():
            if int(check.get("context", 0)) > 0:
                check["context_source"] = "readiness"
                continue
            result_role = role if role in results else "main"
            inference_context = int(
                ((results.get(result_role) or {}).get("timings") or {}).get("n_ctx", 0)
            )
            if inference_context > 0:
                check["context"] = inference_context
                check["context_source"] = "inference-timings"
        exact_context = all(int(check.get("context", 0)) == context for check in checks.values())
        expected_models = {"main": recipe.main_alias}
        if recipe.aux_mode == "dedicated":
            expected_models["aux"] = recipe.aux_alias
        for role, expected_model in expected_models.items():
            actual_model = str((checks.get(role) or {}).get("model", ""))
            if actual_model != expected_model:
                raise RuntimeError(
                    f"{role} readiness model mismatch: expected {expected_model}, got {actual_model or '<empty>'}"
                )
        main = results.get("main") or {}
        aux = results.get("aux") or {}
        if route.get("main") != recipe.main_alias:
            raise RuntimeError(f"main route mismatch: {route}")
        if route.get("aux") != recipe.aux_alias:
            raise RuntimeError(f"aux route mismatch: {route}")
        methods = sorted({component.method for component in recipe.components})
        method = "+".join(methods)
        runtime_string = f"turbofit-runtime use {recipe.profile_name}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        raw_path = self.result_dir / "attempts" / item_id / timestamp / "result.json"
        payload = {
            "schema_version": 1,
            "row": row_payload,
            "profile_name": recipe.profile_name,
            "components": [
                {
                    "role": component.role,
                    "family": component.family,
                    "alias": component.alias,
                    "kind": component.kind,
                    "method": component.method,
                    "gpu": component.gpu,
                    "port": component.port,
                    "command": list(component.command),
                    "model_path": component.model_path,
                    "projector_path": component.projector_path,
                }
                for component in recipe.components
            ],
            "checks": checks,
            "route": route,
            "results": results,
            "gpu_peak_mb": peak,
            "physical_hardware": physical_hardware,
            "runtime_string": runtime_string,
        }
        raw_path.parent.mkdir(parents=True, exist_ok=False)
        raw_path.write_text(json.dumps(payload, indent=2) + "\n")
        raw_result_sha256 = "sha256:" + hashlib.sha256(raw_path.read_bytes()).hexdigest()
        return RawBenchmark(
            method=method,
            exact_context=exact_context,
            main_health=bool(checks.get("main")),
            aux_health=bool(checks.get("aux")) if recipe.aux_mode == "dedicated" else bool(checks.get("main")),
            main_output=str(main.get("content", "")),
            aux_output=str(aux.get("content", "")),
            main_tps=float((main.get("timings") or {}).get("predicted_per_second", 0)),
            aux_tps=float((aux.get("timings") or {}).get("predicted_per_second", 0)),
            gpu_peak_mb=peak,
            physical_fingerprint=str(physical_hardware["fingerprint_sha256"]),
            raw_result_sha256=raw_result_sha256,
            runtime_string=runtime_string,
            raw_result_path=str(raw_path),
        )
