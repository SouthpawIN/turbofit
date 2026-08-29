from __future__ import annotations

import runpy
from pathlib import Path

from turbofit_runtime.downloads import DownloadCatalog


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/turbofit-completion-check"


def module() -> dict:
    return runpy.run_path(str(SCRIPT), run_name="turbofit_completion_check_test")


def test_production_download_gate_uses_current_manifest_group() -> None:
    gate = module()["production_download_gate"](
        ROOT / "runtime-profiles/downloads.json",
        ROOT / "references/artifact-manifest.json",
    )

    assert gate["ok"] is True
    assert gate["group"] == "production-floor"
    assert gate["expected"] == 2
    assert gate["verified"] == 2
    assert gate["missing"] == []
    assert gate["mismatched"] == []


def test_operator_docs_match_current_catalog_size_and_models() -> None:
    campaign_docs = (ROOT / "docs/campaigns.md").read_text(encoding="utf-8")
    backend_docs = (ROOT / "docs/runtime-backends.md").read_text(encoding="utf-8")
    matrix = __import__("json").loads((ROOT / "references/configuration-matrix.json").read_text(encoding="utf-8"))

    assert len(matrix["rows"]) == 552
    assert "552 active rows" in campaign_docs
    assert "currently 552 rows" in backend_docs
    assert "1,620" not in campaign_docs + backend_docs
    assert "--group deepseek-v4-flash-0731-q8-dspark" not in backend_docs
    assert "Qwen 3.8 Q4 plus DFlash2" in backend_docs


def test_qwen_dflash2_variant_has_complete_download_group() -> None:
    catalog = DownloadCatalog.load(ROOT / "runtime-profiles/downloads.json")
    files = catalog.files_for_group("qwen3-8-27b-q4-dflash2")

    assert {item.destination for item in files} == {
        "Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf",
        "Qwen3.8-27B/mmproj-Qwen3.8-27B-Q8_0.gguf",
        "Qwen3.8-27B/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    }


def test_release_gate_derives_current_matrix_candidate_and_version() -> None:
    loaded = module()
    loaded["_command_ok"] = lambda _command: (False, "not exercised in metadata test")
    loaded["_json_command"] = lambda command: (
        {
            "matrix": {"total": 552},
            "current_recipe": {"pending": 552},
        }
        if "catalog-campaign" in command[0]
        else {"levels": {}}
        if "intelligence-campaign" in command[0]
        else {"tiers": []}
    )

    checks = loaded["report"]()["checks"]

    assert checks["canonical_configuration_matrix"]["ok"] is True
    assert checks["canonical_configuration_matrix"]["expected"] == 552
    assert checks["qwen38_day_zero_lane"]["ok"] is True
    assert checks["qwen38_day_zero_lane"]["replacement_candidate"] == "qwen3-8-27b-unleashed-ud-q3-k-xl"
    assert checks["hermes_plugin_and_setup_command"]["ok"] is True
    assert checks["hermes_plugin_and_setup_command"]["version"] == "2.3.1"
    assert "deepseek_artifacts" not in checks
    assert checks["production_floor_artifacts"]["ok"] is True
    assert "bundled_sirvir_customer_service" not in checks
    assert checks["github_sirvir_customer_service"]["ok"] is True
    assert "dspark_live_tool_call" not in checks
    assert checks["speculative_runtime_evidence"]["ok"] is True
    assert checks["speculative_runtime_evidence"]["schema"] == "turbofit.dflash2-ab/v2"
    assert checks["speculative_runtime_evidence"]["hash_bound"] is True
    assert "h3_promo_and_senter_delivery" not in checks
