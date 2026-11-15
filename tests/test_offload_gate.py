"""A model that cannot fit is a ``waiting-memory`` row, not a crash."""
from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from turbofit_runtime.campaign import CampaignRunner
from turbofit_runtime.gguf_footprint import ExpertFootprint
from turbofit_runtime.offload_gate import memory_deferred_rows
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.schema import MatrixRow

ROOT = Path(__file__).parents[1]
GIB = 2**30


def footprint(total_gib: float, expert_gib: float, layers: int) -> ExpertFootprint:
    total = int(total_gib * GIB)
    expert = int(expert_gib * GIB)
    return ExpertFootprint(
        total_bytes=total, expert_bytes=expert, expert_layers=layers,
        dense_bytes=total - expert, tensor_count=layers * 3,
        expert_tensor_count=layers * 3,
        expert_tensor_names=tuple(f"blk.{i}.ffn_gate_exps.weight" for i in range(layers)),
        shards=("model.gguf",),
    )


def book(*, vram_gib: float, ram_gib: float,
         fp: ExpertFootprint | None = None) -> RecipeBook:
    entry = {"alias": "test-moe", "kind": "process", "model": "model.gguf",
             "port": 11605, "methods": {"65536": "baseline"}, "auto_cpu_moe": True}
    return RecipeBook(
        {"schema_version": 1, "atomic_binary": "/bin/true",
         "models": {"Test MoE": entry}, "variants": {}},
        backend_name="cuda", platform_name="linux", hardware=None,
        vram_probe=lambda: int(vram_gib * GIB),
        ram_probe=lambda: int(ram_gib * GIB),
        footprint_probe=lambda _path: fp or footprint(80, 40, 40),
    )


def index() -> dict:
    return {"row-a": {"id": "row-a", "main": "Test MoE", "auxiliary": "auto",
                      "context": 65536, "status": "candidate"}}


def test_infeasible_host_defers_the_row_with_the_planner_reason() -> None:
    # 40 GiB of dense weights cannot fit 7 GiB of VRAM.
    recipes = book(vram_gib=7, ram_gib=256)

    deferrals = memory_deferred_rows(index(), recipes)

    assert list(deferrals) == ["row-a"]
    assert deferrals["row-a"].reason.startswith("non-expert weights alone need")
    assert "GiB" in deferrals["row-a"].reason
    assert deferrals["row-a"].to_dict()["status"] == "waiting-memory"


def test_feasible_host_resolves_the_row_normally() -> None:
    recipes = book(vram_gib=48, ram_gib=512, fp=footprint(180, 150, 60))

    assert memory_deferred_rows(index(), recipes) == {}


def test_unresolvable_row_is_not_a_memory_verdict() -> None:
    class Recipes:
        def resolve_catalog_configuration(self, payload):
            raise ValueError("no method recipe for context 65536")

    assert memory_deferred_rows(index(), Recipes()) == {}


class FakeClearGate:
    def __init__(self):
        self.labels = []

    def wait(self, **kwargs):
        self.labels.append(kwargs.get("label"))
        return None


class FakeExecutor:
    def __init__(self):
        self.rows = []

    def recipe_sha256(self, row):
        return "sha256:recipe"

    def execute_catalog(self, payload):
        self.rows.append(payload["id"])
        raise AssertionError("an infeasible row must never be executed")


def _write_matrix(path: Path, rows) -> None:
    path.write_text(json.dumps({"schema_version": 1, "rows": [row.to_dict() for row in rows]}))


def test_infeasible_row_is_never_selected_or_launched(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    ready = MatrixRow.make_id("Carwin Nano", "auto", 65_536)
    too_big = MatrixRow.make_id("Laguna S2.1", "auto", 65_536)
    rows = (
        MatrixRow(id=ready, main="Carwin Nano", aux="auto", context=65_536),
        MatrixRow(id=too_big, main="Laguna S2.1", aux="auto", context=65_536),
    )
    _write_matrix(matrix_path, rows)

    class Publisher:
        def ensure_checklist(self):
            pass

        def publish(self, *args, **kwargs):
            pass

    class Registry:
        def register(self, *args, **kwargs):
            pass

    gate = FakeClearGate()
    executor = FakeExecutor()
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=executor, clear_gate=gate, publisher=Publisher(), registry=Registry(),
        waiting_memory_rows={too_big: "non-expert weights alone need 12.5 GiB but only 7.0 GiB of VRAM is usable"},
    )

    assert [row.id for row in runner.pending_rows()] == [ready]
    assert [row.id for row in runner.unresolved_rows()] == [ready]

    outcome = runner.run_row(rows[1])
    assert outcome.status == "waiting-memory"
    assert "non-expert weights alone need" in outcome.error
    assert gate.labels == []
    assert executor.rows == []


def _load_campaign_script():
    path = ROOT / "scripts" / "turbofit-catalog-campaign"
    spec = importlib.util.spec_from_loader(
        "turbofit_catalog_campaign_script",
        SourceFileLoader("turbofit_catalog_campaign_script", str(path)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_completes_when_the_catalog_contains_an_infeasible_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Regression: one oversized model used to abort `status` outright."""
    campaign = _load_campaign_script()

    class LoadedBook:
        @staticmethod
        def load(path, hardware=None, **_kwargs):
            # The real recipe book, planned against a host too small for it:
            # the catalog really does contain models this describes.
            return RecipeBook.load(
                path, hardware=hardware, backend_name="cuda",
                vram_probe=lambda: 7 * GIB,
                ram_probe=lambda: 256 * GIB,
                footprint_probe=lambda _path: footprint(80, 40, 40),
            )

    monkeypatch.setattr(campaign, "RecipeBook", LoadedBook)
    monkeypatch.setattr(campaign, "MATRIX", tmp_path / "matrix.json")
    monkeypatch.setattr(campaign, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(campaign, "WIKI", tmp_path / "wiki" / "checklist.md")
    monkeypatch.setattr(campaign, "EVIDENCE", tmp_path / "wiki" / "evidence")

    assert campaign.status() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["matrix"]["total"] == 290
    assert report["current_recipe"]["waiting_memory"] >= 1
    assert report["waiting_memory"]["rows"] == report["current_recipe"]["waiting_memory"]
    detail = report["waiting_memory"]["detail"]
    assert detail
    assert all(item["status"] == "waiting-memory" for item in detail)
    assert all("non-expert weights alone need" in item["reason"] for item in detail)
    # A different cause keeps its own count: 'buy more VRAM' (waiting_memory)
    # is not 'this CPU cannot run Colibri' (deferred) or 'not downloaded yet'.
    assert "deferred" in report["current_recipe"]
    assert "waiting_download" in report["current_recipe"]


def test_status_reports_reason_verbatim_for_non_expert_weights(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    campaign = _load_campaign_script()

    class LoadedBook:
        @staticmethod
        def load(path, hardware=None, **_kwargs):
            return RecipeBook.load(
                path, hardware=hardware, backend_name="cuda",
                vram_probe=lambda: 7000 * 2**20,
                ram_probe=lambda: 256 * GIB,
                footprint_probe=lambda _path: footprint(80, 40, 40),
            )

    monkeypatch.setattr(campaign, "RecipeBook", LoadedBook)
    monkeypatch.setattr(campaign, "MATRIX", tmp_path / "matrix.json")
    monkeypatch.setattr(campaign, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(campaign, "WIKI", tmp_path / "wiki" / "checklist.md")
    monkeypatch.setattr(campaign, "EVIDENCE", tmp_path / "wiki" / "evidence")

    assert campaign.status() == 0
    report = json.loads(capsys.readouterr().out)
    reasons = [item["reason"] for item in report["waiting_memory"]["detail"]]
    assert reasons
    assert all("non-expert weights alone need" in reason for reason in reasons)
    assert all("GiB of VRAM is usable" in reason for reason in reasons)
