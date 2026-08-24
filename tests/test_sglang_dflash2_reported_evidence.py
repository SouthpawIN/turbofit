from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reported_spark_sglang_lane_is_preserved_without_false_promotion() -> None:
    payload = json.loads(
        (
            ROOT
            / "references/results/qwen38-nvfp4-dflash2-sglang-spark-c10-20260824.json"
        ).read_text()
    )

    assert payload["schema"] == "turbofit.reported-throughput/v1"
    assert payload["status"] == "candidate-user-reported"
    assert payload["target"]["repo"] == "RadixArk/Qwen3.8-27B-NVFP4"
    assert payload["target"]["reported_revision"] == "554ebba"
    assert payload["draft"]["revision"] == "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
    assert payload["runtime"]["upstream_dflash2_merge"] == "c14312a66420b75ca9a11bf1817c4db1fa26b097"
    assert payload["launch"]["context_length"] == 262144
    assert payload["launch"]["max_running_requests"] == 10
    assert payload["measurement"]["tokens_per_second"] == 116.0
    assert payload["measurement"]["concurrency"] == 10
    assert payload["measurement"]["metric"] == "aggregate_output_throughput"
    assert payload["promotion"]["eligible"] is False
    assert payload["source"]["raw_harness_attached"] is False
