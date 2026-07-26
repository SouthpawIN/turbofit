"""Execute one resolved Main:Aux recipe and return a normalized benchmark."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from .campaign import RawBenchmark
from .recipes import RecipeBook, ResolvedComponent, ResolvedRecipe
from .schema import MatrixRow


class RuntimeBackend(Protocol):
    def start(self, component: ResolvedComponent): ...
    def wait_ready(self, component: ResolvedComponent, handle) -> dict: ...
    def route(self, recipe: ResolvedRecipe, handles: dict[str, object]) -> dict: ...
    def infer(self, role: str, recipe: ResolvedRecipe) -> dict: ...
    def peak_gpu_mb(self) -> dict[int, int]: ...
    def stop(self, component: ResolvedComponent, handle) -> None: ...


class LocalPairExecutor:
    def __init__(self, *, recipes: RecipeBook, backend: RuntimeBackend, result_dir: Path) -> None:
        self.recipes = recipes
        self.backend = backend
        self.result_dir = result_dir

    def execute(self, item: MatrixRow) -> RawBenchmark:
        recipe = self.recipes.resolve(item)
        started: list[tuple[ResolvedComponent, object]] = []
        checks: dict[str, dict] = {}
        route: dict = {}
        results: dict[str, dict] = {}
        peak: dict[int, int] = {}
        try:
            for component in recipe.components:
                handle = self.backend.start(component)
                started.append((component, handle))
            for component, handle in started:
                checks[component.role] = self.backend.wait_ready(component, handle)
            route = self.backend.route(recipe, {component.role: handle for component, handle in started})
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {role: pool.submit(self.backend.infer, role, recipe) for role in ("main", "aux")}
                results = {role: future.result() for role, future in futures.items()}
            peak = self.backend.peak_gpu_mb()
        finally:
            for component, handle in reversed(started):
                self.backend.stop(component, handle)

        exact_context = all(int(check.get("context", 0)) == item.context for check in checks.values())
        main = results.get("main") or {}
        aux = results.get("aux") or {}
        if route.get("main") != recipe.main_alias:
            raise RuntimeError(f"main route mismatch: {route}")
        if route.get("aux") != recipe.aux_alias:
            raise RuntimeError(f"aux route mismatch: {route}")
        methods = sorted({component.method for component in recipe.components})
        method = "+".join(methods)
        runtime_string = f"turbofit-runtime use {recipe.profile_name}"
        raw_path = self.result_dir / f"{item.id}.json"
        payload = {
            "schema_version": 1,
            "row": item.to_dict(),
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
                    "image": component.image,
                    "environment": component.environment or {},
                    "mounts": list(component.mounts),
                }
                for component in recipe.components
            ],
            "checks": checks,
            "route": route,
            "results": results,
            "gpu_peak_mb": peak,
            "runtime_string": runtime_string,
        }
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2) + "\n")
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
            runtime_string=runtime_string,
            raw_result_path=str(raw_path),
        )
