from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify-dspark-live"


def load_script():
    loader = SourceFileLoader("verify_dspark_live", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_override_changes_port_and_cpu_offload_without_dropping_dspark() -> None:
    module = load_script()
    original = [
        "/llama-server", "--port", "11607", "-ngl", "auto", "-ngld", "auto", "--fit", "on",
        "--spec-type", "draft-dspark", "--jinja",
    ]

    command = module.prepare_command(
        [*original, "--cpu-moe"],
        port=11731,
        backend="cuda",
        gpu_layers="99",
        draft_gpu_layers="0",
        cpu_moe=True,
    )

    assert command[command.index("--port") + 1] == "11731"
    assert command[command.index("-ngl") + 1] == "99"
    assert command[command.index("-ngld") + 1] == "0"
    assert command.count("--cpu-moe") == 1
    assert command[command.index("--spec-type") + 1] == "draft-dspark"
    assert "--jinja" in command


def test_tool_verification_reserves_reasoning_room() -> None:
    module = load_script()

    assert module.tool_token_budget(32) == 128
    assert module.tool_token_budget(256) == 256


def test_matrix_row_selection_supports_each_model_and_requested_context() -> None:
    module = load_script()
    matrix = {
        "rows": [
            {"main": "deepseek-v4-flash-0731-q8-dspark", "auxiliary": "auto", "context": 65536},
            {"main": "deepseek-v4-flash-0731-q4-dspark", "auxiliary": "auto", "context": 1048576},
        ]
    }

    row = module.select_matrix_row(
        matrix,
        main_id="deepseek-v4-flash-0731-q4-dspark",
        context=1048576,
    )

    assert row["main"] == "deepseek-v4-flash-0731-q4-dspark"
    assert row["context"] == 1048576
