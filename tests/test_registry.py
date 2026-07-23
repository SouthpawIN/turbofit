from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.evidence import BenchmarkResult
from turbofit_runtime.gpu import GPUClearEvent, GPUSample
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.registry import ProfileRegistry
from turbofit_runtime.schema import MatrixRow
from turbofit_runtime.turbohaul import TurbohaulCompiler


ROOT = Path(__file__).resolve().parents[1]


def result(row_id: str, method: str, runtime_string: str) -> BenchmarkResult:
    return BenchmarkResult(
        row_id=row_id, method=method, exact_context=True,
        main_health=True, aux_health=True, main_output="main", aux_output="aux",
        main_tps=50.0, aux_tps=100.0, gpu_peak_mb={0: 14000, 1: 18000},
        runtime_string=runtime_string,
        gpu_clear_after=GPUClearEvent(
            timestamp="2026-07-23T00:00:00+00:00", label="after", passed=True,
            ceilings_mb={0: 1024, 1: 1024},
            snapshot=(GPUSample(gpu=0, total_mb=24576, used_mb=500, free_mb=24076, utilization_pct=0),),
            samples_observed=3,
        ),
        raw_result_path="references/results/raw.json",
    )


def row(main: str, aux: str, context: int) -> MatrixRow:
    return MatrixRow(
        id=MatrixRow.make_id(main, aux, context), main=main, aux=aux, context=context,
        status="pending", method_priority=("dspark", "mtp", "nextn"),
    )


def test_register_mtp_profile_and_turbohaul_manifests(tmp_path: Path) -> None:
    item = row("GRM 2.6 Plus", "Carwin Nano", 65_536)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json"),
        profiles_path=profiles,
        turbohaul_dir=tmp_path / "turbohaul",
        compiler=TurbohaulCompiler(hash_fn=lambda path: "a" * 64),
    )

    registry.register(item, result(item.id, "mtp", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    payload = json.loads(profiles.read_text())
    profile = payload["profiles"][item.id]
    assert profile["backend"] == "turbohaul"
    assert profile["runtime_string"].startswith(f"turbofit-runtime use {item.id} --backend turbohaul")
    assert len(profile["turbohaul_manifests"]) == 2
    assert all((tmp_path / "turbohaul" / Path(path).name).exists() for path in profile["turbohaul_manifests"])


def test_register_dspark_profile_as_hybrid_launcher(tmp_path: Path) -> None:
    item = row("Ternary Bonsai", "1 Bit Bonsai", 65_536)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json"),
        profiles_path=profiles,
        turbohaul_dir=tmp_path / "turbohaul",
        compiler=TurbohaulCompiler(hash_fn=lambda path: "b" * 64),
    )

    registry.register(item, result(item.id, "dspark", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    profile = json.loads(profiles.read_text())["profiles"][item.id]
    assert profile["backend"] == "turbohaul-hybrid"
    assert profile["turbohaul_manifests"] == []
    assert profile["runtime_string"] == f"turbofit-runtime use {item.id}"
    assert profile["components"][0]["method"] == "dspark"
