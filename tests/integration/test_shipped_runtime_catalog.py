"""Exercise the production stores together, not a hand-picked fixture catalog."""
import json
from pathlib import Path

import pytest

from turbofit_runtime.controller import load_rung_requirements
from turbofit_runtime.profile_io import load_profile
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.routes import build_route_state, load_runtime_resolutions
from turbofit_runtime.runtime_profile import AuxMode
from turbofit_runtime.selection import ProfileCatalog

ROOT = Path(__file__).resolve().parents[2]
TIERS = sorted((ROOT / "runtime-profiles").glob("*gb.yaml"))


def test_complete_shipped_catalog_loads():
    catalog = ProfileCatalog.from_paths(TIERS)
    assert len(catalog.profiles) == len(TIERS)


def test_shipped_runtime_resolutions_load():
    assert load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json")


@pytest.mark.parametrize("path", TIERS, ids=lambda path: path.stem)
def test_every_selectable_rung_has_bound_requirements_and_native_recipe(path):
    profile = load_profile(path)
    resolutions = load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json")
    load_rung_requirements(ROOT / "runtime-profiles/rung-requirements.json", profile)
    recipes = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cuda")
    index = json.loads((ROOT / "runtime-profiles/class-evidence-index.json").read_text())
    for number, rung in enumerate(profile.rungs):
        build_route_state(profile, number, resolutions, manager_port=8092)
        assert rung.evidence in index
        assert profile.id in index[rung.evidence]["profiles"]
        if rung.aux_mode is AuxMode.API:
            continue
        assert index[rung.evidence]["status"] in {"auto-promoted", "validated-native-tournament"}
        roles = resolutions[profile.id][rung.id]
        assert set(roles) == ({"main", "aux"} if rung.aux_mode is AuxMode.DEDICATED else {"main"})
        for role, target in roles.items():
            component = recipes.resolve_component(
                str(target["family"]), role=role, gpu=str(target["gpu"]),
                port=int(target["port"]), context=rung.context,
                alias=str(target["model_tag"]),
            )
            assert component.method == "baseline"  # No unpromoted DFlash/DSpark substitution.
            assert "bonsai" not in component.family.lower()
            assert rung.main_manifest != "sha256:" + "0" * 64


def test_48gb_winner_matches_canonical_promotion():
    profile = load_profile(ROOT / "runtime-profiles/48gb.yaml")
    promotion = json.loads((ROOT / "references/results/hardware-tier-winner-48gb-qwen38-unleashed-20260823.json").read_text())
    rung = profile.rungs[0]
    assert rung.evidence == promotion["physical_evidence_sha256"]
    assert rung.main_manifest == promotion["recipe_sha256"]
    assert rung.context == promotion["checks"]["main"]["context"]
    target = load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json")[profile.id][rung.id]["main"]
    component = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cuda").resolve_component(
        str(target["family"]), role="main", gpu=str(target["gpu"]), port=int(target["port"]),
        context=rung.context, alias=str(target["model_tag"]),
    )
    expected = promotion["components"][0]
    assert component.gpu == expected["gpu"]
    assert component.port == expected["port"]
    assert component.alias == expected["alias"]
    assert component.method == expected["method"]
    assert Path(component.command[0]).parent.parent.parent.name == "llama.cpp-1c3c9674de4d455f1e571bed808252af54932767"
    # Portable roots may differ, but every launch flag/artifact must match promotion.
    def portable_command(command):
        return [Path(token).name if "/" in token else token for token in command]
    assert portable_command(component.command) == portable_command(expected["command"])


@pytest.mark.parametrize("command", ["list", "status"])
def test_runtime_entrypoint_reads_complete_production_stores(command, tmp_path, monkeypatch, capsys):
    import runpy
    from turbofit_runtime.hardware import HardwareFingerprint

    loaded = runpy.run_path(str(ROOT / "scripts/turbofit-runtime"))
    entrypoint = loaded["main"]
    monkeypatch.setitem(entrypoint.__globals__, "CONFIG_DIR", tmp_path)
    monkeypatch.setitem(entrypoint.__globals__, "probe_hardware", lambda: HardwareFingerprint("linux", "x86_64", 65536))
    monkeypatch.setenv("TURBOFIT_SELECTION_STATE", str(tmp_path / "selection.json"))
    code = entrypoint(["turbofit-runtime", command])
    output = json.loads(capsys.readouterr().out)
    assert code == (0 if command == "list" else 1)
    if command == "list":
        assert {row["id"] for row in output} == {load_profile(path).id for path in TIERS}
        assert {row["id"] for row in output if row["local"]} == {"hardware-8gb", "hardware-48gb"}
    else:
        assert output["configured"] is False
        assert "error" not in output


@pytest.mark.parametrize("path", [p for p in TIERS if p.stem not in {"8gb", "48gb"}], ids=lambda path: path.stem)
def test_unpromoted_hardware_tiers_remain_api_only(path):
    profile = load_profile(path)
    assert profile.policy.recommendation == "api-only-unproven-local"
    assert all(rung.aux_mode is AuxMode.API for rung in profile.rungs)
