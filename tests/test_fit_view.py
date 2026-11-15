"""The /fit view: what to run on this host, and what is missing to run it.

These are pure-logic tests. They drive the real recipes, artifact manifest and
CPU-feature probe against a temporary model root and an injected CPU flag set,
so no GPU, gateway or network is involved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from turbofit_runtime.artifacts import expected_size_for_model_root
from turbofit_runtime.cpu_features import STRATA_REQUIRED
from turbofit_runtime.fit_view import (
    MAPLE_AUX_ID,
    STATUS_ENGINE,
    STATUS_MISSING,
    STATUS_READY,
    STATUS_UNMEASURED,
    build_fit_view,
    engine_label,
    load_fit_view,
)
from turbofit_runtime.recipes import RecipeBook

ROOT = Path(__file__).parents[1]
REFS = ROOT / "references"

MAIN_ID = "qwen3-8-27b-gsq-rco-iq3s--auto--262k"
MAIN_FAMILY = "qwen3-8-27b-gsq-rco-iq3s"
STRATA_ID = "qwen3-8-flash-next-strata--maple-preview-tq2--128k"
STRATA_FAMILY = "qwen3-8-flash-next-strata"

AVX512 = frozenset(STRATA_REQUIRED)
NO_AVX512 = frozenset({"sse4_2", "avx", "avx2"})


def _recommended(configuration_id: str, main: str, auxiliary: str, context: int) -> dict:
    return {
        "configuration_id": configuration_id,
        "main": main,
        "auxiliary": auxiliary,
        "context": context,
        "intelligence_score": 46.875,
        "throughput_tps": 39.4,
        "level": "screening",
        "recorded_at": "2026-08-23T05:38:24.951167+00:00",
    }


def _tier(gb: int, configuration_id: str, main: str, auxiliary: str, context: int) -> dict:
    return {
        "hardware_tier_gb": gb,
        "recommended": _recommended(configuration_id, main, auxiliary, context),
        "options": [
            {
                "configuration_id": MAIN_ID,
                "main": MAIN_FAMILY,
                "intelligence_score": 40.0,
                "throughput_tps": 22.5,
                "level": "screening",
            }
        ],
        "measured": 2,
    }


def _book(tmp_path: Path) -> RecipeBook:
    book = RecipeBook.load(REFS / "model-recipes.json", backend_name="cpu")
    book.model_root = tmp_path
    return book


def _materialize(path: str, size: int | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    os.truncate(target, size if size is not None else 2 * 1024 * 1024)
    return target


def _view(tmp_path, tiers, *, flags: frozenset[str] | None = AVX512, tier_gb=None, installed=lambda _: True):
    book = _book(tmp_path)
    return build_fit_view(
        leaderboard={"schema": "turbofit.intelligence-leaderboard/v1", "tiers": tiers},
        recipe_book=book,
        configurations=json.loads((REFS / "configuration-matrix.json").read_text())["rows"],
        flags=flags,
        binary_installed=installed,
        tier_gb=tier_gb,
        model_names={MAIN_FAMILY: "Qwen 3.8 27B GSQ-RCO IQ3_S", STRATA_FAMILY: "Qwen 3.8 Flash Next (Strata)"},
    )


def test_measured_tier_yields_recommendation_and_options(tmp_path):
    view = _view(tmp_path, [_tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144)])
    assert view["ok"] is True
    assert view["schema"] == "turbofit.fit-view/v1"
    assert view["maple_aux_id"] == MAPLE_AUX_ID
    tier = view["tiers"][0]
    assert tier["hardware_tier_gb"] == 48
    assert tier["measured"] is True
    recommendation = tier["recommendation"]
    assert recommendation is not None
    assert recommendation["configuration_id"] == MAIN_ID
    assert recommendation["main"] == "Qwen 3.8 27B GSQ-RCO IQ3_S"
    assert recommendation["context"] == 262144
    assert recommendation["engine"]["name"] == "llama.cpp"
    assert recommendation["engine"]["usable"] is True
    assert tier["options"][0]["configuration_id"] == MAIN_ID


def test_unmeasured_tier_says_no_measurements_instead_of_inventing_one(tmp_path):
    view = _view(tmp_path, [], tier_gb=24)
    tier = view["tiers"][0]
    assert tier["status"] == STATUS_UNMEASURED
    assert tier["recommendation"] is None
    assert "No measurements yet" in tier["message"]
    assert "24 GB" in tier["message"]
    assert view["summary"].startswith("No measured configuration can run here yet")


def test_missing_artifact_is_listed_with_its_size(tmp_path):
    view = _view(tmp_path, [_tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144)])
    recommendation = view["tiers"][0]["recommendation"]
    assert view["tiers"][0]["status"] == STATUS_MISSING
    assert recommendation["runnable"] is False
    missing_roles = {item["role"] for item in recommendation["missing"]}
    assert "main:model" in missing_roles
    model_state = next(item for item in recommendation["missing"] if item["role"] == "main:model")
    assert model_state["present"] is False
    assert model_state["missing_bytes"] and model_state["missing_bytes"] > 1024**3
    assert view["tiers"][0]["missing_bytes"] >= model_state["missing_bytes"]
    assert "Missing" in view["tiers"][0]["message"]
    assert "GiB" in view["tiers"][0]["message"]


def test_placeholder_weights_are_not_counted_as_installed(tmp_path):
    book = _book(tmp_path)
    component, _ = (
        book.resolve_component(MAIN_FAMILY, role="main", gpu="0", port=11605, context=262144),
        None,
    )
    _materialize(component.model_path, 4096)  # a truncated download, not weights
    view = _view(tmp_path, [_tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144)])
    missing = {item["role"]: item for item in view["tiers"][0]["recommendation"]["missing"]}
    assert missing["main:model"]["present"] is False
    assert missing["main:model"]["reason"].startswith("stub below")


def test_strata_engine_on_a_cpu_without_avx512_is_unavailable(tmp_path):
    view = _view(
        tmp_path,
        [_tier(64, STRATA_ID, STRATA_FAMILY, MAPLE_AUX_ID, 131072)],
        flags=NO_AVX512,
    )
    tier = view["tiers"][0]
    assert tier["status"] == STATUS_ENGINE
    assert tier["recommendation"]["runnable"] is False
    engine = tier["recommendation"]["engine"]
    assert engine["flavor"] == "strata"
    assert engine["usable"] is False
    assert "AVX-512" in engine["reason"]
    assert "avx512f" in engine["reason"]
    assert "AVX-512" in tier["message"]
    assert view["cpu"]["strata_supported"] is False
    assert "avx512vbmi" in view["cpu"]["missing_strata_flags"]


def test_strata_is_usable_when_the_cpu_has_the_features(tmp_path):
    view = _view(tmp_path, [_tier(64, STRATA_ID, STRATA_FAMILY, MAPLE_AUX_ID, 131072)])
    engine = view["tiers"][0]["recommendation"]["engine"]
    assert engine["flavor"] == "strata"
    assert engine["usable"] is True
    assert view["cpu"]["strata_supported"] is True


def test_ready_when_every_artifact_and_engine_is_present(tmp_path):
    book = _book(tmp_path)
    main, aux = (
        book.resolve_component(MAPLE_AUX_ID, role="main", gpu="0", port=11605, context=65536),
        None,
    )
    expected = expected_size_for_model_root(main.model_path, model_root=tmp_path)
    assert expected and expected > 0
    _materialize(main.model_path, expected)

    view = _view(
        tmp_path,
        [_tier(8, f"{MAPLE_AUX_ID}--auto--64k", MAPLE_AUX_ID, f"auto:{MAPLE_AUX_ID}", 65536)],
    )
    tier = view["tiers"][0]
    assert tier["status"] == STATUS_READY
    assert tier["recommendation"]["runnable"] is True
    assert tier["recommendation"]["auxiliary"]["source"] == "shared-main"
    assert tier["message"].startswith("Ready: run")
    assert MAPLE_AUX_ID in tier["message"]
    assert view["summary"].startswith("On this machine, run")


def test_maple_aux_is_pinned_even_when_the_measurement_used_auto(tmp_path):
    view = _view(tmp_path, [_tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144)])
    recommendation = view["tiers"][0]["recommendation"]
    assert recommendation["auxiliary"]["id"] == MAPLE_AUX_ID
    assert recommendation["auxiliary"]["measured_auxiliary"] == f"auto:{MAIN_FAMILY}"
    aux_artifacts = recommendation["auxiliary"]["artifacts"]
    assert aux_artifacts and all(item["role"].startswith("aux:") for item in aux_artifacts)


def test_tier_filter_returns_only_the_requested_tier(tmp_path):
    tiers = [
        _tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144),
        _tier(64, STRATA_ID, STRATA_FAMILY, MAPLE_AUX_ID, 131072),
    ]
    view = _view(tmp_path, tiers, tier_gb=64)
    assert [row["hardware_tier_gb"] for row in view["tiers"]] == [64]
    unknown = _view(tmp_path, tiers, tier_gb=12)
    assert unknown["tiers"][0]["status"] == STATUS_UNMEASURED
    assert "12 GB" in unknown["tiers"][0]["message"]


def test_messages_are_sentences_not_status_codes(tmp_path):
    view = _view(
        tmp_path,
        [
            _tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144),
            _tier(64, STRATA_ID, STRATA_FAMILY, MAPLE_AUX_ID, 131072),
        ],
        flags=NO_AVX512,
    )
    for tier in view["tiers"]:
        message = tier["message"]
        assert message.endswith(".")
        assert tier["status"] not in message
        assert "error" not in message.lower()
        assert len(message) > 40


def test_missing_engine_binary_is_named(tmp_path):
    view = _view(
        tmp_path,
        [_tier(48, MAIN_ID, MAIN_FAMILY, f"auto:{MAIN_FAMILY}", 262144)],
        installed=lambda _path: False,
    )
    engine = view["tiers"][0]["recommendation"]["engine"]
    assert engine["installed"] is False
    assert engine["usable"] is False
    assert "not installed" in engine["reason"]


def test_unprobed_host_reports_cpu_support_as_unknown(tmp_path):
    book = _book(tmp_path)
    view = build_fit_view(
        leaderboard={"tiers": [_tier(64, STRATA_ID, STRATA_FAMILY, MAPLE_AUX_ID, 131072)]},
        recipe_book=book,
        configurations=json.loads((REFS / "configuration-matrix.json").read_text())["rows"],
        probe_flags=False,
        binary_installed=lambda _: True,
    )
    assert view["cpu"]["flags_known"] is False
    assert view["cpu"]["strata_supported"] is None
    assert "could not be checked" in view["cpu"]["statement"]


def test_engine_labels():
    assert engine_label("mainline") == "llama.cpp"
    assert engine_label("strata") == "strata"
    assert engine_label("") == "unknown engine"


def test_recipes_expose_engine_and_draft_for_fit_checks(tmp_path):
    book = _book(tmp_path)
    component = book.resolve_component(MAIN_FAMILY, role="main", gpu="0", port=11605, context=262144)
    assert component.engine == "mainline"
    assert component.draft_path.endswith(".gguf")
    strata = book.resolve_component(STRATA_FAMILY, role="main", gpu="0", port=11605, context=131072)
    assert strata.engine == "strata"
    assert "${" not in " ".join(strata.command)


def test_committed_references_answer_the_question_for_this_host() -> None:
    """The shipped leaderboard resolves end to end, whatever is on disk here."""
    view = load_fit_view(ROOT)

    assert view["ok"] is True
    assert view["maple_aux_id"] == MAPLE_AUX_ID
    measured = [item for item in view["tiers"] if item["measured"]]
    assert measured and measured[0]["hardware_tier_gb"] == 48
    recommendation = measured[0]["recommendation"]
    assert recommendation["auxiliary"]["id"] == MAPLE_AUX_ID
    assert recommendation["engine"]["name"] == "llama.cpp"
    assert measured[0]["status"] in {STATUS_READY, STATUS_MISSING, STATUS_ENGINE}
    for tier in view["tiers"]:
        assert tier["message"].endswith(".")
        row = tier["hardware_tier_gb"]
        assert row in {48}
