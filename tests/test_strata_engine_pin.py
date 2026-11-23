"""The Strata pin must not silently drift back to an older engine.

A stale engine invalidates every throughput measurement taken on it, so the
pin is a contract, not a default. These tests fail if someone reverts it
without also reverting the release evidence that justified it.
"""
import json
import pathlib

from turbofit_runtime import strata as S

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_app_and_engine_pins_are_current():
    assert S.STRATA_APP_VERSION == "0.2.19"
    assert S.STRATA_ENGINE_VERSION == "0.1.65"


def test_app_and_engine_versions_move_independently():
    # They are different numbers on purpose; conflating them is the original bug.
    assert S.STRATA_APP_VERSION != S.STRATA_ENGINE_VERSION


def test_engine_serve_matrix_agrees_with_the_module_pin():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    entry = d["engines"]["strata"]
    assert entry["ref"] == S.STRATA_ENGINE_VERSION
    assert entry["app_version"] == S.STRATA_APP_VERSION


def test_inkling_shape_engine_is_declared():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    assert "strata-inkling" in d["engines"]["strata"]["shape_engines"]


def test_sm86_gains_are_recorded_with_their_provenance():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    note = d["engines"]["strata"]["sm86_gains"]
    assert "sm_86" in note and "bit-exact" in note
    # A vendor number must never be presented as our own measurement.
    assert "Vendor-reported" in note


def test_dual_gpu_expert_parallelism_is_distinct_from_layer_split():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    note = d["engines"]["strata"]["dual_gpu"]
    assert "--gpus 2" in note
    assert "layer split" in note


def test_dgx_spark_platform_is_named():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    assert "DGX Spark" in d["engines"]["strata"]["platforms_note"]


def test_native_runtime_revision_matches_the_app_pin():
    d = json.loads((ROOT / "references/native-runtimes.json").read_text())
    blob = json.dumps(d)
    assert f"v{S.STRATA_APP_VERSION}" in blob


def test_release_source_is_cited():
    d = json.loads((ROOT / "references/engine-serve-matrix.json").read_text())
    assert d["engines"]["strata"]["source_release"].endswith("/v0.2.19")
