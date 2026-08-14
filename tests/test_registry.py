from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.evidence import BenchmarkResult
from turbofit_runtime.gpu import GPUClearEvent, GPUSample
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.registry import ProfileRegistry
from turbofit_runtime.schema import MatrixRow


ROOT = Path(__file__).resolve().parents[1]


def result(row_id: str, method: str, runtime_string: str) -> BenchmarkResult:
    return BenchmarkResult(
        row_id=row_id, method=method, exact_context=True,
        main_health=True, aux_health=True, main_output="main", aux_output="aux",
        main_tps=50.0, aux_tps=100.0, gpu_peak_mb={0: 14000, 1: 18000},
        physical_fingerprint="sha256:" + "a" * 64,
        raw_result_sha256="sha256:" + "b" * 64,
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


def recipe_book(tmp_path: Path) -> RecipeBook:
    data = json.loads((ROOT / "references/model-recipes.json").read_text())
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for spec in data["models"].values():
        model_name = Path(spec["model"]).name
        model_path = model_dir / model_name
        model_path.write_bytes(b"test-model")
        spec["model_root"] = str(model_dir)
        spec["model"] = str(model_path)
        if spec.get("draft"):
            draft_name = Path(spec["draft"]).name
            (model_dir / draft_name).write_bytes(b"test-draft")
            spec["draft"] = str(model_dir / draft_name)
        if spec.get("projector"):
            projector_name = Path(spec["projector"]).name
            (model_dir / projector_name).write_bytes(b"test-projector")
            spec["projector"] = str(model_dir / projector_name)
    return RecipeBook(data, platform_name="linux")


def test_register_mtp_native_profile(tmp_path: Path) -> None:
    item = row("qwen3-8-27b-q4-mtp", "Carwin Nano", 65_536)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(recipes=recipe_book(tmp_path), profiles_path=profiles)

    registry.register(item, result(item.id, "mtp", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    payload = json.loads(profiles.read_text())
    profile = payload["profiles"][item.id]
    assert profile["backend"] == "native-process"
    assert profile["runtime_string"] == f"turbofit-runtime use {item.id}"


def test_register_dspark_native_profile(tmp_path: Path) -> None:
    item = row("Ternary Bonsai", "1 Bit Bonsai", 65_536)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(recipes=recipe_book(tmp_path), profiles_path=profiles)

    registry.register(item, result(item.id, "dspark", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    profile = json.loads(profiles.read_text())["profiles"][item.id]
    assert profile["backend"] == "native-process"
    assert profile["runtime_string"] == f"turbofit-runtime use {item.id}"
    assert profile["components"][0]["method"] == "dspark"


def test_register_qwen_dedicated_pair_pins_main_to_second_gpu(tmp_path: Path) -> None:
    item = row("qwen3-8-27b-q4-mtp", "Carwin Nano", 262_144)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(recipes=recipe_book(tmp_path), profiles_path=profiles)

    registry.register(item, result(item.id, "mtp", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    profile = json.loads(profiles.read_text())["profiles"][item.id]
    assert profile["backend"] == "native-process"
    assert profile["runtime_string"] == f"turbofit-runtime use {item.id}"
    main = next(component for component in profile["components"] if component["role"] == "main")
    assert main["gpu"] == "1"
    assert main["command"][main["command"].index("--model-draft") + 1].endswith("mtp-Qwen3.8-27B-Q4_0.gguf")


def test_register_native_profile_preserves_runtime_arguments(tmp_path: Path) -> None:
    item = row("1 Bit Bonsai", "auto", 1_048_576)
    profiles = tmp_path / "profiles.json"
    registry = ProfileRegistry(recipes=recipe_book(tmp_path), profiles_path=profiles)

    registry.register(item, result(item.id, "baseline", f"turbofit-runtime use {item.id}"), tmp_path / "evidence.md")

    profile = json.loads(profiles.read_text())["profiles"][item.id]
    component = profile["components"][0]
    assert component["gpu"] == "0,1"
    assert component["command"][component["command"].index("--rope-scale") + 1] == "4"
