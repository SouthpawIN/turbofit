from __future__ import annotations

from pathlib import Path

from turbofit_runtime.executor import LocalPairExecutor
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.schema import MatrixRow


ROOT = Path(__file__).resolve().parents[1]


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


def test_executor_starts_aux_before_main_and_stops_reverse() -> None:
    backend = FakeBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json"),
        backend=backend,
        result_dir=ROOT / "references/results",
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
    assert result.runtime_string.startswith("turbofit-runtime use ternary-bonsai-1-bit-bonsai-64k")


def test_executor_stops_started_component_when_second_start_fails() -> None:
    class FailingBackend(FakeBackend):
        def start(self, component):
            if component.role == "main":
                raise RuntimeError("main failed")
            return super().start(component)

    backend = FailingBackend()
    executor = LocalPairExecutor(
        recipes=RecipeBook.load(ROOT / "references/model-recipes.json"),
        backend=backend,
        result_dir=ROOT / "references/results",
    )

    try:
        executor.execute(row("Ternary Bonsai", "1 Bit Bonsai", 65_536))
    except RuntimeError as error:
        assert "main failed" in str(error)
    else:
        raise AssertionError("expected start failure")

    assert ("stop", "aux") in backend.actions
