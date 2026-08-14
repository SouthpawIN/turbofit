"""Register passing native campaign rows as swappable runtime profiles."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .evidence import BenchmarkResult
from .recipes import RecipeBook, ResolvedComponent
from .schema import MatrixRow


class ProfileRegistry:
    def __init__(self, *, recipes: RecipeBook, profiles_path: Path) -> None:
        self.recipes = recipes
        self.profiles_path = profiles_path

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _profile_component(component: ResolvedComponent) -> dict:
        if component.kind != "process":
            raise ValueError(f"unsupported native component kind: {component.kind}")
        return {
            "role": component.role,
            "kind": "process",
            "name": f"turbofit-runtime-{component.role}",
            "gpu": component.gpu,
            "port": component.port,
            "method": component.method,
            "command": list(component.command),
        }

    def register(
        self,
        item: MatrixRow,
        result: BenchmarkResult,
        evidence_path: Path,
        *,
        recipe=None,
    ) -> None:
        recipe = recipe or self.recipes.resolve(item)
        try:
            existing = json.loads(self.profiles_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"schema_version": 2, "gateway": "http://127.0.0.1:8091", "profiles": {}}
        existing["schema_version"] = 2
        existing.setdefault("gateway", "http://127.0.0.1:8091")
        existing.setdefault("profiles", {})
        existing["profiles"][item.id] = {
            "evidence_schema": "turbofit.catalog-physical/v4",
            "description": f"{item.main} main with {item.aux} auxiliary at {item.context}",
            "context": item.context,
            "evidence": str(evidence_path),
            "raw_result": result.raw_result_path,
            "raw_result_sha256": result.raw_result_sha256,
            "physical_fingerprint": result.physical_fingerprint,
            "backend": "native-process",
            "runtime_string": result.runtime_string,
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
        }
        self._atomic_json(self.profiles_path, existing)
