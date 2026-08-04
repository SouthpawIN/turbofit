from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from turbofit_runtime.benchmark_schema import (
    BenchmarkRecord,
    BenchmarkSuite,
    PromotionRejected,
    StageResult,
    load_record,
    load_suite,
    require_promotion,
)


ROOT = Path(__file__).parents[1]
DIGEST = "sha256:" + "a" * 64
STAGES = ("artifact", "runtime", "performance", "quality", "pressure-self-heal")


def valid_record() -> BenchmarkRecord:
    return BenchmarkRecord(
        candidate_id="grm-carwin-262k",
        artifact_hashes=(DIGEST, "sha256:" + "b" * 64),
        host_fingerprint="sha256:" + "c" * 64,
        observed_context=262144,
        throughput_tps=60.5,
        ttft_ms=180.0,
        per_card_vram_mb=(21817, 22814),
        power_w_by_card=(310.0, 285.0),
        quality_score=0.87,
        raw_result_identity="sha256:" + "d" * 64,
        stages=tuple(
            StageResult(stage=stage, passed=True, evidence_identity="sha256:" + f"{index:x}" * 64)
            for index, stage in enumerate(STAGES, start=1)
        ),
    )


def test_canonical_suite_defines_all_required_promotion_stages() -> None:
    suite = load_suite(ROOT / "benchmarks" / "suite.yaml")

    assert tuple(stage.id for stage in suite.stages) == STAGES
    assert all(stage.required for stage in suite.stages)


def test_complete_record_passes_every_promotion_gate() -> None:
    suite = load_suite(ROOT / "benchmarks" / "suite.yaml")

    decision = require_promotion(valid_record(), suite)

    assert decision.promoted is True
    assert decision.failures == ()


def test_metal_suite_allows_unified_memory_without_privileged_power_sampling() -> None:
    mapping = valid_record().to_mapping()
    mapping["power_w_by_card"] = []
    record = BenchmarkRecord.from_mapping(mapping)
    suite = load_suite(ROOT / "benchmarks" / "suite-metal.json")

    decision = require_promotion(record, suite)

    assert decision.promoted is True


def test_failed_or_missing_stage_cannot_promote() -> None:
    suite = load_suite(ROOT / "benchmarks" / "suite.yaml")
    record = valid_record()
    failed = BenchmarkRecord(
        **{
            **record.__dict__,
            "stages": tuple(
                StageResult(item.stage, False, item.evidence_identity)
                if item.stage == "quality"
                else item
                for item in record.stages
            ),
        }
    )

    with pytest.raises(PromotionRejected, match="quality"):
        require_promotion(failed, suite)

    missing = BenchmarkRecord(**{**record.__dict__, "stages": record.stages[:-1]})
    with pytest.raises(PromotionRejected, match="pressure-self-heal"):
        require_promotion(missing, suite)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_hashes", ("not-a-hash",)),
        ("host_fingerprint", "host-name"),
        ("observed_context", 0),
        ("throughput_tps", 0),
        ("ttft_ms", -1),
        ("per_card_vram_mb", ()),
        ("power_w_by_card", (0.0, 0.0)),
        ("quality_score", -0.1),
        ("raw_result_identity", "result.json"),
    ],
)
def test_required_measurement_fields_are_strict(field: str, value: object) -> None:
    record = valid_record()
    with pytest.raises(ValueError, match=field):
        BenchmarkRecord(**{**record.__dict__, field: value})


def test_record_json_round_trip_is_deterministic(tmp_path: Path) -> None:
    record = valid_record()
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record.to_mapping(), sort_keys=True))

    loaded = load_record(path)

    assert loaded == record
    assert loaded.to_mapping() == record.to_mapping()


def test_suite_rejects_duplicate_or_incomplete_stage_definitions() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkSuite((
            load_suite(ROOT / "benchmarks" / "suite.yaml").stages[0],
            load_suite(ROOT / "benchmarks" / "suite.yaml").stages[0],
        ))


def test_matrix_success_marking_requires_a_passing_promotion_record(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "matrix_benchmark_gate", ROOT / "scripts" / "matrix-benchmark.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite_path = str(ROOT / "benchmarks" / "suite.yaml")

    allowed, reason = module.promotion_allowed(None, suite_path)
    assert allowed is False
    assert "promotion-record" in reason

    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(valid_record().to_mapping()))
    allowed, reason = module.promotion_allowed(str(record_path), suite_path)
    assert allowed is True
    assert reason is None
    assert "/home/sovthpaw" not in (ROOT / "scripts" / "matrix-benchmark.py").read_text()
