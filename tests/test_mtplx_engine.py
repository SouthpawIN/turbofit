from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.engine_check import engine_specs
from turbofit_runtime.mtplx_engine import (
    MTPLX_QUALITY_MODEL,
    MTPLX_SPEED_MODEL,
    MTPLX_FLASH_BARE_SPEED,
    MTPLX_FLASH_OPTIMIZED_SPEED,
    build_mtplx_launch,
    canonical_mtplx_alias,
    discover_mtplx,
    mtplx_telemetry,
)


ROOT = Path(__file__).parents[1]


def test_distribution_includes_mtplx_runtime_adapter() -> None:
    distribution = (ROOT / "distribution.yaml").read_text()
    assert "  - scripts/turbofit-mtplx-runtime\n" in distribution


def test_engine_registry_exposes_mtplx_as_endpoint_discoverable() -> None:
    specs = {item.engine_id: item for item in engine_specs()}

    assert specs["mtplx"].openai_port == 8000
    assert specs["mtplx"].endpoint_discoverable is True
    assert specs["mtplx"].commands == ("mtplx",)


def test_discovery_reads_app_port_and_requires_mtplx_health(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"host": "127.0.0.1", "port": 18086}))
    calls: list[str] = []

    result = discover_mtplx(
        settings_path=settings,
        json_get=lambda url, _timeout: calls.append(url) or {
            "ok": True,
            "model": "qwen38-mtplx-speed",
            "model_path": "/models/speed",
            "generation_mode": "mtp",
            "depth": 3,
            "startup": {"pid": 4321, "launch_id": "app-owned"},
        },
    )

    assert calls == ["http://127.0.0.1:18086/health"]
    assert result.port == 18086
    assert result.pid == 4321
    assert result.owned_by_turbofit is False
    assert result.app_launch_id == "app-owned"


def test_discovery_probes_turbofit_owned_mtplx_port_without_app_settings(tmp_path: Path) -> None:
    calls: list[str] = []

    def get(url, _timeout):
        calls.append(url)
        if url != "http://127.0.0.1:18082/health":
            raise OSError("closed")
        return {
            "ok": True,
            "model": "qwen3-8-27b-mtplx-optimized-speed",
            "model_path": "/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
            "startup": {"pid": 77},
        }

    result = discover_mtplx(settings_path=tmp_path / "missing.json", json_get=get)

    assert result is not None
    assert result.port == 18082
    assert "http://127.0.0.1:18082/health" in calls


def test_owned_mtplx_launch_uses_published_model_contract(tmp_path: Path) -> None:
    executable = tmp_path / "mtplx"
    executable.write_text("mtplx")
    executable.chmod(0o755)
    model = tmp_path / "quality"
    model.mkdir()
    (model / "config.json").write_text("{}")

    launch = build_mtplx_launch(
        executable=executable,
        model_path=model,
        model_repo=MTPLX_QUALITY_MODEL,
        model_id="qwen38-mtplx-quality",
        port=18082,
    )

    assert MTPLX_SPEED_MODEL == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert launch.engine_id == "mtplx"
    assert launch.model_id == "qwen38-mtplx-quality"
    assert launch.upstream_model_id == "qwen38-mtplx-quality"
    assert launch.command == (
        str(executable),
        "serve",
        "--model",
        str(model),
        "--profile",
        "turbo",
        "--host",
        "127.0.0.1",
        "--port",
        "18082",
        "--no-auth",
        "--model-id",
        "qwen38-mtplx-quality",
        "--generation-mode",
        "mtp",
        "--fan-mode",
        "default",
        "--no-stats-footer",
    )


def test_cached_mtplx_paths_map_to_stable_turbofit_aliases() -> None:
    assert canonical_mtplx_alias(
        "/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
    ) == "qwen3-8-27b-mtplx-optimized-speed"
    assert canonical_mtplx_alias(
        "/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality"
    ) == "qwen3-8-27b-mtplx-optimized-quality"
    assert canonical_mtplx_alias(
        "/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Bare-Speed"
    ) == "qwen3-8-flash-next-mtplx-bare-speed"
    assert canonical_mtplx_alias(
        "/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    ) == "qwen3-8-flash-next-mtplx-optimized-speed"


def test_flash_next_models_are_supported_engine_candidates(tmp_path: Path) -> None:
    executable = tmp_path / "mtplx"
    executable.write_text("mtplx")
    executable.chmod(0o755)
    model = tmp_path / "flash"
    model.mkdir()
    (model / "config.json").write_text("{}")

    for repository, alias in (
        (MTPLX_FLASH_BARE_SPEED, "qwen3-8-flash-next-mtplx-bare-speed"),
        (MTPLX_FLASH_OPTIMIZED_SPEED, "qwen3-8-flash-next-mtplx-optimized-speed"),
    ):
        launch = build_mtplx_launch(
            executable=executable,
            model_path=model,
            model_repo=repository,
            model_id=alias,
            port=18082,
        )
        assert launch.model_id == alias
        assert launch.context_length == 262_144


def test_telemetry_preserves_measured_fields_without_inventing_tps() -> None:
    payload = mtplx_telemetry({
        "ok": True,
        "model": "qwen38-mtplx-speed",
        "generation_mode": "mtp",
        "depth": 3,
        "scheduler": {"mode": "serial", "active_requests": 0},
        "memory_plan": {"model_weights_bytes": 123, "context_window_resolved": 262144},
    }, {
        "latest": {
            "decode_tok_s": 58.2,
            "request_tok_s": 55.1,
            "request_effective_mtp_depth": 3,
            "peak_memory_bytes": 456,
        }
    })

    assert payload["decode_tokens_per_second"] == 58.2
    assert payload["request_tokens_per_second"] == 55.1
    assert payload["peak_memory_bytes"] == 456
    assert payload["acceptance_rate"] is None
    assert payload["context_length"] == 262144
    assert payload["scheduler_mode"] == "serial"

    unmeasured = mtplx_telemetry({"ok": True, "model": "qwen38-mtplx-speed"})
    assert unmeasured["decode_tokens_per_second"] is None
