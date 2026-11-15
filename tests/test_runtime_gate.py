"""A missing engine binary is a ``waiting-runtime`` row, not a model failure.

The campaign must not spend an attempt, a GPU clear and a failure-evidence
directory rediscovering that ``scripts/install.sh`` has not installed the
engine yet. These tests pin the row verdict, the selection gate, and the
status accounting that keeps the campaign totals reconciling.
"""
from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from turbofit_runtime.campaign import CampaignRunner
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.runtime_gate import (
    INSTALL_HINT,
    engine_binary_installed,
    waiting_runtime_rows,
)
from turbofit_runtime.schema import MatrixRow, load_matrix

ROOT = Path(__file__).parents[1]


def book(*, binary: str) -> RecipeBook:
    """One mainline-family recipe whose engine binary is *binary*."""
    entry = {
        "alias": "test-main", "kind": "process", "model": "model.gguf",
        "port": 11605, "methods": {"65536": "baseline"}, "binary": binary,
    }
    return RecipeBook(
        {"schema_version": 1, "atomic_binary": "/bin/true",
         "models": {"Test Main": entry}, "variants": {}},
        backend_name="cuda", platform_name="linux", hardware=None,
    )


def index() -> dict:
    return {"row-a": {"id": "row-a", "main": "Test Main", "auxiliary": "auto",
                      "context": 65536, "status": "candidate"}}


def test_missing_engine_binary_is_waiting_runtime_with_path_and_hint(tmp_path: Path) -> None:
    binary = tmp_path / "runtimes" / "prism-llama.cpp-9ca265a5" / "build-cuda" / "bin" / "llama-server"

    waiting = waiting_runtime_rows(index(), book(binary=str(binary)))

    assert list(waiting) == ["row-a"]
    entry = waiting["row-a"]
    assert entry.to_dict()["status"] == "waiting-runtime"
    assert entry.reason.startswith("waiting-runtime: engine binary not found for Test Main:")
    assert str(binary) in entry.reason
    assert INSTALL_HINT in entry.reason
    assert entry.binary == str(binary)


def test_present_engine_binary_leaves_the_row_runnable(tmp_path: Path) -> None:
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\n")

    waiting = waiting_runtime_rows(index(), book(binary=str(binary)))

    assert waiting == {}


def test_bare_command_name_is_resolved_on_path() -> None:
    assert engine_binary_installed("llama-server", which=lambda name: "/usr/bin/llama-server")
    assert not engine_binary_installed("llama-server", which=lambda name: None)
    assert not engine_binary_installed("")
    # A directory is not a runnable engine.
    assert not engine_binary_installed("/tmp")


def test_row_another_gate_claimed_keeps_its_own_verdict(tmp_path: Path) -> None:
    binary = tmp_path / "missing" / "llama-server"

    waiting = waiting_runtime_rows(
        index(), book(binary=str(binary)), skip={"row-a"},
    )

    assert waiting == {}


class FakeClearGate:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def wait(self, **kwargs):
        self.labels.append(str(kwargs.get("label")))
        raise AssertionError("a waiting-runtime row must never reach the GPU clear gate")


class FakeExecutor:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def recipe_sha256(self, item: MatrixRow) -> str:
        return "sha-" + item.id

    def execute(self, item: MatrixRow):
        self.rows.append(item.id)
        raise AssertionError("a waiting-runtime row must never be launched")


class Publisher:
    def ensure_checklist(self) -> None:
        pass

    def publish(self, *args, **kwargs) -> None:
        pass


class Registry:
    def register(self, *args, **kwargs) -> None:
        pass


def _write_matrix(path: Path, rows: tuple[MatrixRow, ...]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "rows": [row.to_dict() for row in rows]}))


def test_waiting_runtime_row_is_never_selected_or_launched(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    ready = MatrixRow.make_id("Carwin Nano", "auto", 65_536)
    no_engine = MatrixRow.make_id("Ternary Bonsai", "auto", 65_536)
    rows = (
        MatrixRow(id=ready, main="Carwin Nano", aux="auto", context=65_536),
        MatrixRow(id=no_engine, main="Ternary Bonsai", aux="auto", context=65_536),
    )
    _write_matrix(matrix_path, rows)

    gate = FakeClearGate()
    executor = FakeExecutor()
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=executor, clear_gate=gate, publisher=Publisher(), registry=Registry(),
        waiting_runtime_rows={
            no_engine: (
                "waiting-runtime: engine binary not found for Ternary Bonsai: "
                f"{str(tmp_path / 'build-cuda' / 'bin' / 'llama-server')!r}. {INSTALL_HINT}"
            )
        },
    )

    assert [row.id for row in runner.pending_rows()] == [ready]
    assert [row.id for row in runner.unresolved_rows()] == [ready]

    outcome = runner.run_row(rows[1])
    assert outcome.status == "waiting-runtime"
    assert "engine binary not found for Ternary Bonsai" in outcome.error
    assert gate.labels == []
    assert executor.rows == []


def _load_campaign_script():
    path = ROOT / "scripts" / "turbofit-catalog-campaign"
    spec = importlib.util.spec_from_loader(
        "turbofit_catalog_campaign_script_runtime",
        SourceFileLoader("turbofit_catalog_campaign_script_runtime", str(path)),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Download:
    missing_bytes = 1024

    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        self.reason = "waiting-download: 1.0 KiB missing"

    def to_dict(self) -> dict:
        return {"row_id": self.row_id, "status": "waiting-download",
                "reason": self.reason, "missing_bytes": self.missing_bytes}


class _Memory:
    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        self.reason = "non-expert weights alone need 12.5 GiB"

    def to_dict(self) -> dict:
        return {"row_id": self.row_id, "status": "waiting-memory",
                "reason": self.reason}


class _Runtime:
    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        self.reason = "waiting-runtime: engine binary not found"

    def to_dict(self) -> dict:
        return {"row_id": self.row_id, "status": "waiting-runtime",
                "reason": self.reason,
                "binary": "/nonexistent/llama-server"}


def test_status_counts_reconcile_with_all_three_waiting_verdicts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    campaign = _load_campaign_script()
    monkeypatch.setattr(campaign, "MATRIX", tmp_path / "matrix.json")
    monkeypatch.setattr(campaign, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(campaign, "WIKI", tmp_path / "wiki" / "checklist.md")
    monkeypatch.setattr(campaign, "EVIDENCE", tmp_path / "wiki" / "evidence")

    catalog, configurations = campaign.canonical()
    matrix = load_matrix(campaign.MATRIX)
    deferred, _policy = campaign.deferred_rows(configurations, catalog)
    available = [row.id for row in matrix.rows if row.id not in deferred]
    assert len(available) >= 4
    downloading, too_big, no_engine, runnable = available[:4]

    monkeypatch.setattr(
        campaign, "waiting_download_rows",
        lambda index, recipes: {downloading: _Download(downloading)},
    )
    monkeypatch.setattr(
        campaign, "memory_deferred_rows",
        lambda index, recipes: {too_big: _Memory(too_big)},
    )
    seen_skip: list[set[str]] = []

    def fake_runtime(index, recipes, *, skip=()):
        seen_skip.append(set(skip))
        assert {downloading, too_big} <= set(skip)
        return {no_engine: _Runtime(no_engine)}

    monkeypatch.setattr(campaign, "waiting_runtime_rows", fake_runtime)

    assert campaign.status() == 0
    report = json.loads(capsys.readouterr().out)

    current = report["current_recipe"]
    assert current["waiting_download"] == 1
    assert current["waiting_memory"] == 1
    assert current["waiting_runtime"] == 1
    assert (
        current["pending"] + current["resolved"] + current["deferred"]
        + current["waiting_download"] + current["waiting_memory"]
        + current["waiting_runtime"]
    ) == report["matrix"]["total"]
    assert report["waiting_runtime"]["rows"] == 1
    assert report["waiting_runtime"]["install_hint"] == INSTALL_HINT
    assert [item["row_id"] for item in report["waiting_runtime"]["detail"]] == [no_engine]
    assert all(item["status"] == "waiting-runtime"
               for item in report["waiting_runtime"]["detail"])
    # The waiting-runtime row is not pending, so it can never be selected.
    assert no_engine not in {
        row.id for row in campaign.build_runner().pending_rows()
    }
    assert seen_skip and seen_skip[0] == {downloading, too_big}
