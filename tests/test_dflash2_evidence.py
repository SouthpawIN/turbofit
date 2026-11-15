from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_dflash2_real_ab_is_hash_bound_and_faster_than_baseline() -> None:
    payload = json.loads((ROOT / "references/results/qwen38-dflash2-2x24-20260823.json").read_text())
    comparison = payload["comparison"]

    assert payload["schema"] == "turbofit.dflash2-ab/v1"
    assert payload["hardware"]["class"] == "2x24"
    assert payload["draft"]["sha256"] == "18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594"
    assert payload["runtime"]["revision"] == "1deefcca395743049c3820ab8f9b15043f3e9446"
    assert comparison["baseline_main_tps"] > 0
    assert comparison["dflash2_main_tps"] > comparison["baseline_main_tps"]
    assert comparison["speedup"] > 1.0
    assert 0 < comparison["accepted_draft_tokens"] <= comparison["draft_tokens"]
    assert payload["evidence_sha256"].startswith("sha256:")
