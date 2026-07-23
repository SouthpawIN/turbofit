"""Register every passing campaign row as a swappable runtime profile."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .evidence import BenchmarkResult
from .recipes import RecipeBook, ResolvedComponent
from .schema import MatrixRow
from .turbohaul import ComponentSpec, TurbohaulCompiler, UnsupportedTurbohaulMethod


class ProfileRegistry:
    def __init__(
        self,
        *,
        recipes: RecipeBook,
        profiles_path: Path,
        turbohaul_dir: Path,
        compiler: TurbohaulCompiler | None = None,
    ) -> None:
        self.recipes = recipes
        self.profiles_path = profiles_path
        self.turbohaul_dir = turbohaul_dir
        self.compiler = compiler or TurbohaulCompiler()

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try: os.unlink(temporary)
            except FileNotFoundError: pass

    @staticmethod
    def _profile_component(component: ResolvedComponent) -> dict:
        common = {
            "role": component.role,
            "kind": component.kind,
            "name": f"turbofit-runtime-{component.role}",
            "gpu": f"device={component.gpu}" if component.kind == "docker" else component.gpu,
            "port": component.port,
            "method": component.method,
        }
        if component.kind == "docker":
            return {
                **common,
                "image": component.image,
                "mounts": list(component.mounts),
                "environment": component.environment or {},
            }
        return {**common, "command": list(component.command)}

    @staticmethod
    def _component_gpu(component: ResolvedComponent) -> int:
        return int(component.gpu.split(",")[0])

    def _component_spec(self, row: MatrixRow, result: BenchmarkResult, component: ResolvedComponent) -> ComponentSpec:
        gpu = self._component_gpu(component)
        expected = result.gpu_peak_mb.get(gpu)
        if expected is None:
            expected = max(result.gpu_peak_mb.values())
        return ComponentSpec(
            role=component.role,
            model_tag=f"{row.id}-{component.role}"[:64],
            model_path=Path(component.model_path),
            projector_path=Path(component.projector_path) if component.projector_path else None,
            context=row.context,
            expected_vram_mb=expected,
            gpu=gpu,
            method=component.method,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            auto_place_eligible=False,
            vision=bool(component.projector_path),
        )

    def register(self, item: MatrixRow, result: BenchmarkResult, evidence_path: Path) -> None:
        recipe = self.recipes.resolve(item)
        try:
            existing = json.loads(self.profiles_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"schema_version": 1, "gateway": "http://127.0.0.1:8091", "profiles": {}}
        existing.setdefault("schema_version", 1)
        existing.setdefault("gateway", "http://127.0.0.1:8091")
        existing.setdefault("profiles", {})

        manifest_paths = []
        model_tags = []
        backend = "turbohaul"
        compiled_payloads: list[tuple[Path, dict]] = []
        try:
            for component in recipe.components:
                compiled = self.compiler.compile_component(self._component_spec(item, result, component))
                manifest_path = self.turbohaul_dir / f"{compiled.manifest['model_tag']}.json"
                compiled_payloads.append((manifest_path, {
                    "manifest": compiled.manifest,
                    "sources": compiled.sources,
                    "matrix_row": item.id,
                    "evidence": str(evidence_path),
                }))
                manifest_paths.append(str(manifest_path))
                model_tags.append(compiled.manifest["model_tag"])
            runtime_string = self.compiler.runtime_string(item.id, tuple(model_tags))
        except UnsupportedTurbohaulMethod:
            backend = "turbohaul-hybrid"
            manifest_paths = []
            runtime_string = result.runtime_string
            compiled_payloads = []

        for path, payload in compiled_payloads:
            self._atomic_json(path, payload)
        existing["profiles"][item.id] = {
            "description": f"{item.main} main with {item.aux} auxiliary at {item.context}",
            "context": item.context,
            "evidence": str(evidence_path),
            "backend": backend,
            "runtime_string": runtime_string,
            "expected": {
                "main_alias": recipe.main_alias,
                "aux_alias": recipe.aux_alias,
                "aux_mode": recipe.aux_mode,
            },
            "metrics": {
                "main_tps": result.main_tps,
                "aux_tps": result.aux_tps,
                "gpu_peak_mb": {str(key): value for key, value in result.gpu_peak_mb.items()},
                "method": result.method,
            },
            "components": [self._profile_component(component) for component in recipe.components],
            "turbohaul_manifests": manifest_paths,
        }
        self._atomic_json(self.profiles_path, existing)
