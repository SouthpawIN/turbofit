from __future__ import annotations

from pathlib import Path

from turbofit_runtime.apple_mlx import (
    MLX_LM_VERSION,
    MLX_VERSION,
    MODEL_FAMILY,
    MODEL_REVISION,
    MODEL_REPO,
    build_apple_mlx_launch,
    load_python_runtime,
)


ROOT = Path(__file__).parents[1]


def test_apple_mlx_runtime_and_model_are_immutably_pinned(tmp_path: Path) -> None:
    runtime = load_python_runtime(ROOT / "references/python-runtimes.json", "mlx-lm")

    assert runtime["packages"] == {
        "mlx": f"mlx=={MLX_VERSION}",
        "mlx-lm": f"mlx-lm=={MLX_LM_VERSION}",
    }
    assert MODEL_REPO == "orcarouter/Qwen3.8-27B-Uncensored-MLX"
    assert MODEL_REVISION == "b4603df5fd2a51e7fed2560ee7090caa4e13e4b7"
    assert MODEL_FAMILY == "qwen3-8-27b-uncensored-mlx-8bit"


def test_build_apple_mlx_launch_uses_loopback_and_real_upstream_model_id(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtimes"
    python = runtime_root / f"mlx-lm-{MLX_LM_VERSION}" / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("python")
    python.chmod(0o755)
    model_root = tmp_path / "models"
    model = model_root / "Qwen3.8-27B-Uncensored-MLX/8-bit"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")

    launch = build_apple_mlx_launch(
        model_root=model_root,
        runtime_root=runtime_root,
        port=18081,
    )

    assert launch.engine_id == "mlx"
    assert launch.model_id == MODEL_FAMILY
    assert launch.upstream_model_id == str(model)
    assert launch.context_length == 262_144
    assert launch.command == (
        str(python),
        "-m",
        "mlx_lm",
        "server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "18081",
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--max-tokens",
        "32768",
    )
