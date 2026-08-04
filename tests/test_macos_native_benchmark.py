from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

from turbofit_runtime.benchmark_schema import load_suite, require_promotion


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_macos_native",
    ROOT / "scripts" / "benchmark-macos-native",
    loader=SourceFileLoader(
        "benchmark_macos_native",
        str(ROOT / "scripts" / "benchmark-macos-native"),
    ),
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metal_evidence_compiles_to_a_promotion_record_without_power() -> None:
    identity = "sha256:" + "a" * 64
    evidence = {
        "candidate": "bonsai-27b-1bit-64k-main",
        "status": "pass",
        "evidence_sha256": identity,
        "hardware": {
            "platform": "macOS",
            "machine": "arm64",
            "accelerator_memory_kind": "unified",
        },
        "summary": {
            "effective_output_tokens_per_second": 1.2,
            "mean_ttft_ms": 1500.0,
            "peak_gpu_memory_used_mib": [9000],
            "quality_pass_rate": 1.0,
            "context_pass_rate": 1.0,
        },
        "cases": [
            {
                "category": "throughput",
                "timings": {"predicted_per_second": 1.25},
            }
        ],
    }
    props = {
        "context_length": 65536,
        "default_generation_settings": {"n_ctx": 65536},
    }

    record = MODULE._promotion_record(evidence, props, "b" * 64)
    decision = require_promotion(
        record,
        load_suite(ROOT / "benchmarks" / "suite-metal.json"),
    )

    assert decision.promoted is True
    assert record.observed_context == 65536
    assert record.power_w_by_card == ()
    assert record.per_card_vram_mb == (9000,)


def test_gateway_props_supply_runtime_build_identity() -> None:
    props = {
        "build_info": "b10173-e9fa0781f",
        "context_length": 65536,
    }

    assert MODULE._configuration(props) == "metal/native/b10173-e9fa0781f/64k/q4-kv"
