from __future__ import annotations

from turbofit_runtime.turbofit_list import build_turbofit_list, render_turbofit_list


def report() -> dict:
    tiers = []
    for capacity in (8, 16, 24, 48, 64, 96, 200, 300):
        winner = None
        status = "catalog-candidates-only"
        if capacity == 48:
            status = "physically-validated"
            winner = {
                "configuration_id": "qwen--ornith--128k",
                "main": {"name": "Qwen"},
                "auxiliary": {"name": "Ornith"},
                "context": 131072,
                "fit": {
                    "physically_demonstrated": True,
                    "evidence": "sha256:" + "a" * 64,
                    "hardware_fingerprint": "sha256:" + "b" * 64,
                },
                "intelligence_score": 62.5,
                "measured_tps": 38.4,
                "balanced_score": 68.0,
            }
        tiers.append({
            "id": f"hardware-{capacity}gb",
            "capacity_gb": capacity,
            "status": status,
            "recommendations": {
                "measured_winner": winner,
                "smartest": winner,
                "fastest": winner,
                "balanced": winner,
            },
            "candidates": [],
        })
    return {"schema": "turbofit.hardware-tier-report/v1", "tiers": tiers}


def test_turbofit_list_contains_only_exact_evidence_backed_hardware_winners() -> None:
    payload = build_turbofit_list(report())

    assert payload["schema"] == "turbofit.list/v1"
    assert payload["name"] == "TurboFit List"
    assert [item["hardware_level_gb"] for item in payload["levels"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    assert next(item for item in payload["levels"] if item["hardware_level_gb"] == 48)["winner"]["configuration_id"] == "qwen--ornith--128k"
    assert next(item for item in payload["levels"] if item["hardware_level_gb"] == 8)["status"] == "pending-benchmarks"


def test_turbofit_list_rejects_nonphysical_balanced_candidate() -> None:
    payload = report()
    tier8 = payload["tiers"][0]
    tier8["recommendations"]["balanced"] = {
        "configuration_id": "portable-only",
        "fit": {"physically_demonstrated": False},
        "intelligence_score": 90,
        "measured_tps": 100,
        "balanced_score": 94,
    }

    result = build_turbofit_list(payload)

    assert result["levels"][0]["winner"] is None


def test_rendered_list_calls_scan_to_configuration_turbofit_check() -> None:
    text = render_turbofit_list(build_turbofit_list(report()))

    assert text.startswith("# TurboFit List")
    assert "TurboFit Check" in text
    assert "system scan-to-configuration process" in text
    assert "Qwen" in text
