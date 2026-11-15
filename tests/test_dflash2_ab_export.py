from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/turbofit-dflash2-ab-evidence"


def module() -> dict:
    return runpy.run_path(str(SCRIPT), run_name="turbofit_dflash2_ab_test")


def write_result(
    path: Path,
    *,
    row_id: str,
    tps: float,
    peak: int,
    draft: bool,
    fingerprint: str = "sha256:fingerprint",
) -> str:
    timings = {
        "predicted_per_second": tps,
        "predicted_n": 128,
        "prompt_n": 26,
    }
    if draft:
        timings.update({"draft_n": 150, "draft_n_accepted": 105})
    payload = {
        "row": {"id": row_id, "context": 65536},
        "components": [{"role": "main", "command": ["/runtime/llama-server"]}],
        "results": {
            "main": {
                "usage": {"prompt_tokens": 26, "completion_tokens": 128},
                "timings": timings,
            }
        },
        "gpu_peak_mb": {"0": 486, "1": peak},
        "physical_hardware": {
            "captured_at": "2026-08-24T08:00:00+00:00",
            "fingerprint": {"topology_key": "2x24576mb", "devices": [{"name": "RTX 3090"}, {"name": "RTX 3090"}]},
            "fingerprint_sha256": fingerprint,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_evidence_exports_comparable_hash_bound_ab(tmp_path: Path) -> None:
    baseline_id = "baseline"
    dflash_id = "dflash2"
    baseline_path = tmp_path / "baseline.json"
    dflash_path = tmp_path / "dflash.json"
    baseline_sha = write_result(baseline_path, row_id=baseline_id, tps=35.0, peak=20341, draft=False)
    dflash_sha = write_result(dflash_path, row_id=dflash_id, tps=45.5, peak=23181, draft=True)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "rows": {
                    baseline_id: {
                        "status": "success",
                        "raw_result_path": str(baseline_path),
                        "raw_result_sha256": baseline_sha,
                        "recipe_sha256": "sha256:baseline-recipe",
                        "physical_fingerprint": "sha256:fingerprint",
                    },
                    dflash_id: {
                        "status": "success",
                        "raw_result_path": str(dflash_path),
                        "raw_result_sha256": dflash_sha,
                        "recipe_sha256": "sha256:dflash-recipe",
                        "physical_fingerprint": "sha256:fingerprint",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    evidence = module()["build_evidence"](state_path, baseline_id, dflash_id)

    assert evidence["schema"] == "turbofit.dflash2-ab/v2"
    assert evidence["hardware"]["fingerprint_sha256"] == "sha256:fingerprint"
    assert evidence["arms"]["baseline"]["raw_result_sha256"] == baseline_sha
    assert evidence["arms"]["dflash2"]["raw_result_sha256"] == dflash_sha
    assert evidence["comparison"]["baseline_main_tps"] == 35.0
    assert evidence["comparison"]["dflash2_main_tps"] == 45.5
    assert evidence["comparison"]["speedup"] == 1.3
    assert evidence["comparison"]["percent_gain"] == 30.0
    assert evidence["comparison"]["additional_peak_gpu_mb"] == 2840
    assert evidence["comparison"]["draft_acceptance_fraction"] == 0.7
    assert evidence["promotion"]["eligible"] is True
    assert evidence["evidence_sha256"].startswith("sha256:")


def test_build_evidence_refuses_different_hardware(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    dflash_path = tmp_path / "dflash.json"
    baseline_sha = write_result(baseline_path, row_id="baseline", tps=35.0, peak=20341, draft=False)
    dflash_sha = write_result(dflash_path, row_id="dflash2", tps=45.5, peak=23181, draft=True)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "rows": {
                    "baseline": {"status": "success", "raw_result_path": str(baseline_path), "raw_result_sha256": baseline_sha, "recipe_sha256": "sha256:a", "physical_fingerprint": "sha256:one"},
                    "dflash2": {"status": "success", "raw_result_path": str(dflash_path), "raw_result_sha256": dflash_sha, "recipe_sha256": "sha256:b", "physical_fingerprint": "sha256:two"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="physical fingerprint"):
        module()["build_evidence"](state_path, "baseline", "dflash2")


def test_build_evidence_refuses_raw_fingerprint_that_disagrees_with_state(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    dflash_path = tmp_path / "dflash.json"
    baseline_sha = write_result(baseline_path, row_id="baseline", tps=35.0, peak=20341, draft=False)
    dflash_sha = write_result(
        dflash_path,
        row_id="dflash2",
        tps=45.5,
        peak=23181,
        draft=True,
        fingerprint="sha256:other-hardware",
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "rows": {
                    "baseline": {"status": "success", "raw_result_path": str(baseline_path), "raw_result_sha256": baseline_sha, "recipe_sha256": "sha256:a", "physical_fingerprint": "sha256:fingerprint"},
                    "dflash2": {"status": "success", "raw_result_path": str(dflash_path), "raw_result_sha256": dflash_sha, "recipe_sha256": "sha256:b", "physical_fingerprint": "sha256:fingerprint"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embedded physical fingerprint"):
        module()["build_evidence"](state_path, "baseline", "dflash2")
