from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from turbofit_runtime.executor import LocalPairExecutor
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.schema import MatrixRow


ROOT = Path(__file__).resolve().parents[1]
FAKE_FINGERPRINT = {
    "topology_key": "2x24576mb",
    "accelerator_memory_mb": 49152,
    "devices": [{"uuid": "gpu-0"}, {"uuid": "gpu-1"}],
    "drivers": {"gpu-0": "test", "gpu-1": "test"},
}
FINGERPRINT = "sha256:" + hashlib.sha256(
    json.dumps(FAKE_FINGERPRINT, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def fake_physical_hardware() -> dict:
    return {
        "schema": "turbofit.physical-hardware/v1",
        "captured_at": "2026-08-10T00:00:00+00:00",
        "fingerprint": FAKE_FINGERPRINT,
        "fingerprint_sha256": FINGERPRINT,
    }


def row(main: str, aux: str, context: int) -> MatrixRow:
    return MatrixRow(
        id=MatrixRow.make_id(main, aux, context), main=main, aux=aux, context=context,
        status="pending", method_priority=("dspark", "mtp", "nextn"),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.actions: list[tuple] = []

    def start(self, component):
        self.actions.append(("start", component.role, component.family))
        return f"handle-{component.role}"

    def wait_ready(self, component, handle):
        self.actions.append(("ready", component.role))
        return {"context": 65_536, "model": component.alias}

    def route(self, recipe, handles):
        self.actions.append(("route", recipe.main_alias, recipe.aux_alias))
        return {"main": recipe.main_alias, "aux": recipe.aux_alias}

    def infer(self, role, recipe):
        self.actions.append(("infer", role))
        return {
            "backend": recipe.main_alias if role == "main" else recipe.aux_alias,
            "content": f"{role} output",
            "timings": {"predicted_per_second": 50.0 if role == "main" else 90.0},
        }

    def peak_gpu_mb(self):
        return {0: 14000, 1: 18000}

    def stop(self, component, handle):
        self.actions.append(("stop", component.role))


def test_executor_starts_aux_before_main_and_stops_reverse(tmp_path: Path) -> None:
    backend = FakeBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(
            ROOT / "references/model-recipes.json", platform_name="linux"
        ),
        backend=backend,
        result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )

    result = executor.execute(row("Ternary Bonsai", "1 Bit Bonsai", 65_536))

    starts = [action for action in backend.actions if action[0] == "start"]
    stops = [action for action in backend.actions if action[0] == "stop"]
    assert starts == [("start", "aux", "1 Bit Bonsai"), ("start", "main", "Ternary Bonsai")]
    assert stops == [("stop", "main"), ("stop", "aux")]
    assert result.exact_context is True
    assert result.main_output == "main output"
    assert result.aux_output == "aux output"
    assert result.main_tps == 50.0
    assert result.aux_tps == 90.0
    assert result.gpu_peak_mb == {0: 14000, 1: 18000}
    assert result.physical_fingerprint == FINGERPRINT
    payload = json.loads(Path(result.raw_result_path).read_text())
    assert payload["physical_hardware"]["fingerprint_sha256"] == FINGERPRINT
    assert result.runtime_string.startswith("turbofit-runtime use ternary-bonsai-1-bit-bonsai-64k")
    record = {
        "raw_result_path": result.raw_result_path,
        "raw_result_sha256": result.raw_result_sha256,
        "physical_fingerprint": result.physical_fingerprint,
    }
    assert executor.evidence_is_current(record) is True
    Path(result.raw_result_path).write_text("{}")
    assert executor.evidence_is_current(record) is False


def test_executor_runs_catalog_variant_configuration(tmp_path: Path) -> None:
    backend = FakeBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(
            ROOT / "references/model-recipes.json", platform_name="linux"
        ),
        backend=backend,
        result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )
    item = {
        "id": "ternary-bonsai-27b-dspark--bonsai-27b-dspark--65536",
        "main": "ternary-bonsai-27b-dspark",
        "auxiliary": "bonsai-27b-dspark",
        "context": 65_536,
        "status": "candidate",
    }

    result = executor.execute_catalog(item)

    assert result.main_output == "main output"
    assert result.aux_output == "aux output"
    assert result.exact_context is True
    assert Path(result.raw_result_path).is_file()
    assert f"attempts/{item['id']}/" in result.raw_result_path


def test_shared_main_smoke_requests_are_serialized(tmp_path: Path) -> None:
    class SerialBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active = threading.Lock()

        def infer(self, role, recipe):
            assert self.active.acquire(blocking=False), "shared-main requests overlapped"
            try:
                time.sleep(0.01)
                return super().infer(role, recipe)
            finally:
                self.active.release()

    backend = SerialBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json", platform_name="linux"),
        backend=backend, result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )

    executor.execute(row("1 Bit Bonsai", "auto", 65_536))

    assert [action for action in backend.actions if action[0] == "infer"] == [
        ("infer", "main"), ("infer", "aux"),
    ]


def test_executor_proves_context_from_inference_when_readiness_omits_it(
    tmp_path: Path,
) -> None:
    class MissingReadinessContext(FakeBackend):
        def wait_ready(self, component, handle):
            result = super().wait_ready(component, handle)
            result["context"] = 0
            return result

        def infer(self, role, recipe):
            result = super().infer(role, recipe)
            result["timings"]["n_ctx"] = 65_536
            return result

    backend = MissingReadinessContext()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json", platform_name="linux"),
        backend=backend, result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )

    result = executor.execute(row("1 Bit Bonsai", "auto", 65_536))
    payload = json.loads(Path(result.raw_result_path).read_text())

    assert result.exact_context is True
    assert payload["checks"]["main"]["context"] == 65_536
    assert payload["checks"]["main"]["context_source"] == "inference-timings"


def test_executor_stops_started_component_when_second_start_fails(tmp_path: Path) -> None:
    class FailingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.result_dir = tmp_path
            (tmp_path / "campaign-gateway.log").write_text("gateway diagnostics")

        def start(self, component):
            if component.role == "main":
                raise RuntimeError("main failed")
            return super().start(component)

    backend = FailingBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(
            ROOT / "references/model-recipes.json", platform_name="linux"
        ),
        backend=backend,
        result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )

    try:
        executor.execute(row("Ternary Bonsai", "1 Bit Bonsai", 65_536))
    except RuntimeError as error:
        assert "main failed" in str(error)
    else:
        raise AssertionError("expected start failure")

    assert ("stop", "aux") in backend.actions
    evidence = list((tmp_path / "failures").glob("**/failure.json"))
    assert len(evidence) == 1
    payload = json.loads(evidence[0].read_text())
    assert payload["schema"] == "turbofit.catalog-failure/v1"
    assert "main failed" in payload["error"]
    assert payload["components"][0]["command"]
    assert any(log["role"] == "gateway" for log in payload["logs"])


def test_executor_rejects_readiness_from_the_wrong_model_alias(tmp_path: Path) -> None:
    class WrongAliasBackend(FakeBackend):
        def wait_ready(self, component, handle):
            value = super().wait_ready(component, handle)
            if component.role == "main":
                value["model"] = "aux-model-that-owned-the-port"
            return value

    executor = LocalPairExecutor(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json", platform_name="linux"),
        backend=WrongAliasBackend(),
        result_dir=tmp_path,
        physical_hardware_probe=fake_physical_hardware,
    )

    with pytest.raises(RuntimeError, match="main readiness model mismatch"):
        executor.execute(row("Ternary Bonsai", "1 Bit Bonsai", 65_536))
