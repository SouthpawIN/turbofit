"""First run: scan → recommend one model → download only that.

These tests are offline and GPU-free. They drive the committed references, a
temporary model root, and an injected hardware fingerprint, so the properties
under test are the plan's, not this container's.

The properties, in the order the product needs them:

* the plan holds the recommendation plus whatever fallback rungs were accepted
  -- never the catalog;
* presence is checked, not assumed: complete files are skipped and counted,
  missing ones are sized;
* the fallback rungs are opt-in, and opting in adds exactly their artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from turbofit_runtime.cpu_features import STRATA_REQUIRED
from turbofit_runtime.fit_view import STATUS_MEMORY, build_fit_view
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.onboard import (
    PLAN_BLOCKED,
    PLAN_COMPLETE,
    PLAN_READY,
    FallbackPolicyError,
    build_plan,
    execute_plan,
    parse_fallback_policy,
    render_plan,
    scan_host,
)
from turbofit_runtime.recipes import RecipeBook

ROOT = Path(__file__).parents[1]
REFS = ROOT / "references"
GIB = 2 ** 30
AVX512 = frozenset(STRATA_REQUIRED)
NO_AVX512 = frozenset({"sse4_2", "avx", "avx2"})


def _host(vram_mb: int = 48 * 1024, ram_mb: int = 128 * 1024) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=ram_mb,
        devices=(
            AcceleratorDevice(
                index=0,
                uuid="GPU-test",
                name="Test Accelerator",
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=vram_mb,
                compute_capability="8.9",
                bus_id="0000:01:00.0",
            ),
        ),
    )


def _plan(tmp_path: Path, *, include_fallback="none", flags=AVX512, tier_gb=48, **kwargs):
    model_root = kwargs.pop("model_root", tmp_path / "models")
    return build_plan(
        ROOT,
        hardware=kwargs.pop("hardware", _host()),
        ram_bytes=kwargs.pop("ram_bytes", 128 * GIB),
        flags=flags,
        tier_gb=tier_gb,
        include_fallback=include_fallback,
        model_root=model_root,
        **kwargs,
    )


def _manifest_rows() -> list[dict]:
    return json.loads((REFS / "artifact-manifest.json").read_text(encoding="utf-8"))["artifacts"]


def _row_for(destination: str) -> dict:
    for row in _manifest_rows():
        if row["destination"] == destination:
            return row
    raise AssertionError(f"no manifest row for {destination}")


def _materialize(model_root: Path, plan: dict, *, only=None) -> None:
    """Create sparse files at their pinned sizes, i.e. "already installed"."""
    for entry in plan["recommendation"]["artifacts"]:
        if only is not None and entry["path"] not in only:
            continue
        destination = entry["download"]["destination"]
        row = _row_for(destination)
        target = model_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        os.truncate(target, row["size_bytes"])


def test_scan_uses_the_published_tier_for_this_capacity() -> None:
    scan = scan_host(_host(), ram_bytes=128 * GIB, flags=AVX512)
    assert scan.tier_gb == 48
    assert scan.tier_source == "host"
    assert scan.lane == "dedicated"
    assert scan.per_device_gb == 48
    assert scan.as_dict()["topology_key"] == "1x49152mb"


def test_scan_lets_an_explicit_tier_win() -> None:
    scan = scan_host(_host(), ram_bytes=8 * GIB, flags=AVX512, tier_gb=24)
    assert (scan.tier_gb, scan.tier_source) == (24, "explicit")


def test_plan_downloads_only_the_recommendation_with_its_sizes(tmp_path) -> None:
    plan = _plan(tmp_path)

    assert plan["ok"] is True
    assert plan["schema"] == "turbofit.onboard-plan/v1"
    assert plan["status"] == PLAN_READY
    assert plan["mode"] == "dry-run"
    recommendation = plan["recommendation"]
    assert recommendation["source"] == "tier-measurement"
    assert recommendation["evidence"] == "measured-on-box"

    planned = {entry["download"]["destination"] for entry in plan["download"]["entries"]}
    expected = {
        entry["download"]["destination"] for entry in recommendation["artifacts"]
    }
    assert planned == expected
    # main weights, projector, MTP draft and the Maple auxiliary -- and nothing
    # from the other 84 catalog families.
    assert len(planned) == 4
    assert plan["total"]["artifacts_to_download"] == 4
    assert plan["total"]["download_bytes"] == sum(
        int(entry["download_bytes"]) for entry in plan["download"]["entries"]
    )
    assert plan["total"]["download_bytes"] == recommendation["download_bytes"]
    assert plan["total"]["download_bytes"] > 20 * GIB
    # Sizes come from the pinned manifest (or the measured footprint), never 0.
    for entry in plan["download"]["entries"]:
        assert entry["full_bytes"] and entry["full_bytes"] > 0
        assert entry["bytes_basis"] in {"manifest-pinned", "measured-footprint"}
        assert entry["action"] == "download"


def test_plan_is_a_dry_run_until_executed(tmp_path) -> None:
    model_root = tmp_path / "models"
    plan = _plan(tmp_path)
    assert not model_root.exists() or not any(model_root.iterdir())
    assert plan["download"]["bytes"] > 0
    assert "scripts/turbofit-onboard --download" in " ".join(plan["next"])


def test_present_artifacts_are_skipped_and_reported(tmp_path) -> None:
    model_root = tmp_path / "models"
    first = _plan(tmp_path)
    _materialize(model_root, first)

    plan = _plan(tmp_path)

    assert plan["status"] == PLAN_COMPLETE
    assert plan["download"]["entries"] == []
    assert plan["total"]["download_bytes"] == 0
    assert plan["total"]["artifacts_present"] == 4
    assert "already on this machine" in plan["message"]
    assert "nothing will be downloaded" in plan["message"]


def test_partial_presence_counts_only_the_missing_bytes(tmp_path) -> None:
    model_root = tmp_path / "models"
    first = _plan(tmp_path)
    kept = first["recommendation"]["artifacts"][0]["path"]
    _materialize(model_root, first, only={kept})

    plan = _plan(tmp_path)

    assert plan["status"] == PLAN_READY
    assert plan["total"]["artifacts_to_download"] == 3
    assert plan["total"]["artifacts_present"] == 1
    assert all(entry["path"] != kept for entry in plan["download"]["entries"])
    assert 0 < plan["total"]["download_bytes"] < first["total"]["download_bytes"]


def test_fallback_rungs_are_excluded_by_default(tmp_path) -> None:
    plan = _plan(tmp_path)
    assert plan["fallback"]["policy"] == "none"
    assert plan["fallback"]["included_rungs"] == []
    assert all(rung["included"] is False for rung in plan["fallback"]["rungs"])
    assert plan["download"]["bytes"] == plan["recommendation"]["download_bytes"]


def test_include_fallback_one_adds_exactly_one_rung(tmp_path) -> None:
    base = _plan(tmp_path)
    plan = _plan(tmp_path, include_fallback=1)

    assert plan["fallback"]["included_rungs"] == [1]
    included = next(rung for rung in plan["fallback"]["rungs"] if rung["index"] == 1)
    extra = {
        entry["download"]["destination"]
        for entry in included["artifacts"]
        if entry["action"] == "download"
    }
    assert extra, "the first rung must bring artifacts of its own"
    planned = {entry["download"]["destination"] for entry in plan["download"]["entries"]}
    recommended = {
        entry["download"]["destination"] for entry in plan["recommendation"]["artifacts"]
    }
    assert planned == recommended | extra
    assert plan["total"]["download_bytes"] == (
        base["total"]["download_bytes"] + included["incremental_bytes"]
    )
    assert included["incremental_bytes"] == sum(
        int(entry["download_bytes"])
        for entry in included["artifacts"]
        if entry["action"] == "download"
    )


def test_include_fallback_all_adds_every_selectable_rung(tmp_path) -> None:
    plan = _plan(tmp_path, include_fallback="all")
    selectable = [rung for rung in plan["fallback"]["rungs"] if rung["selectable"]]
    assert selectable
    assert plan["fallback"]["included_rungs"] == [rung["index"] for rung in selectable]
    planned = {entry["download"]["destination"] for entry in plan["download"]["entries"]}
    for rung in selectable:
        for entry in rung["artifacts"]:
            if entry["action"] == "download":
                assert entry["download"]["destination"] in planned


def test_a_rung_that_needs_more_vram_than_this_card_is_not_selectable(tmp_path) -> None:
    plan = _plan(tmp_path, hardware=_host(vram_mb=8 * 1024), tier_gb=8, include_fallback="all")
    # Nothing on an 8 GB dedicated card closes over the chain's rungs.
    assert plan["fallback"]["rungs"] == [] or all(
        rung["selectable"] is False for rung in plan["fallback"]["rungs"]
    )


def test_no_plan_ever_contains_the_whole_catalog(tmp_path) -> None:
    catalog = {row["destination"] for row in _manifest_rows()}
    plan = _plan(tmp_path, include_fallback="all")
    planned = {entry["download"]["destination"] for entry in plan["download"]["entries"]}

    assert planned, "the plan must contain something"
    assert planned < catalog, "the plan must be a strict subset of the manifest"
    assert len(planned) < len(catalog) // 4
    for entry in plan["download"]["entries"]:
        assert entry.get("shared_with") is None or entry["shared_with"] == "recommendation"


def test_shared_artifacts_are_counted_once(tmp_path) -> None:
    plan = _plan(tmp_path, include_fallback="all")
    destinations = [entry["download"]["destination"] for entry in plan["download"]["entries"]]
    assert len(destinations) == len(set(destinations))


def test_offload_placement_is_labelled_with_its_ram_requirement(tmp_path) -> None:
    """/fit and the plan both have to say a model only fits by spilling experts to RAM."""
    from turbofit_runtime.gguf_footprint import ExpertFootprint

    footprint = ExpertFootprint(
        total_bytes=60 * GIB,
        expert_bytes=30 * GIB,
        expert_layers=40,
        dense_bytes=30 * GIB,
        tensor_count=120,
        expert_tensor_count=120,
        expert_tensor_names=(),
        shards=("model.gguf",),
    )
    book = RecipeBook.load(
        REFS / "model-recipes.json",
        backend_name="cuda",
        vram_probe=lambda: 48 * GIB,
        ram_probe=lambda: 256 * GIB,
        footprint_probe=lambda _path: footprint,
    )
    book.model_root = tmp_path / "models"
    view = build_fit_view(
        leaderboard={
            "schema": "turbofit.intelligence-leaderboard/v1",
            "tiers": [
                {
                    "hardware_tier_gb": 48,
                    "recommended": {
                        "configuration_id": "minimax-m3--auto--64k",
                        "main": "MiniMax M3",
                        "auxiliary": "auto:MiniMax M3",
                        "context": 65536,
                        "intelligence_score": 41.0,
                        "throughput_tps": 21.0,
                    },
                    "options": [],
                    "measured": 1,
                }
            ],
        },
        recipe_book=book,
        configurations=json.loads((REFS / "configuration-matrix.json").read_text())["rows"],
        flags=AVX512,
        binary_installed=lambda _: True,
        model_names={"MiniMax M3": "MiniMax M3"},
    )
    tier = view["tiers"][0]
    assert tier["status"] != STATUS_MEMORY
    placement = tier["recommendation"]["placement"]
    assert placement["kind"] == "offload"
    assert placement["n_cpu_moe"] > 0
    assert placement["ram_requirement"], "offload must name the RAM it needs"
    assert "expert layers in RAM" in placement["label"]

    plan = _plan(tmp_path, view=view, recipe_book=book, model_root=tmp_path / "models")
    assert plan["recommendation"]["placement"]["kind"] == "offload"
    assert plan["recommendation"]["placement"]["ram_requirement"]
    assert "Placement: offload" in plan["message"]
    assert "offload" in render_plan(plan)


def test_unmeasured_tier_offers_a_clearly_labelled_chain_rung(tmp_path) -> None:
    plan = _plan(tmp_path, tier_gb=96)
    if plan["recommendation"] is None:
        # Nothing on this host both fits and ships a fetchable recipe: the plan
        # must say so rather than invent an option.
        assert plan["status"] in {PLAN_BLOCKED, "no-measurements"}
        assert plan["download"]["entries"] == []
        assert "No measurements yet" in plan["message"]
        return
    assert plan["tier"]["measured"] is False
    assert plan["recommendation"]["source"] == "fallback-chain"
    assert plan["recommendation"]["evidence"] == "portable-fit"
    assert "No measurements yet" in plan["message"]
    assert "not a benchmarked winner" in plan["message"]


def test_strata_only_recommendation_is_blocked_without_avx512(tmp_path) -> None:
    plan = _plan(tmp_path, tier_gb=96, flags=NO_AVX512)
    if plan["recommendation"] is None:
        assert plan["status"] in {PLAN_BLOCKED, "no-measurements"}
        return
    # Any recommendation that survives must not be a Strata placement on a CPU
    # that cannot decode it.
    assert plan["recommendation"]["engine"]["flavor"] != "strata"


def test_execute_plan_fetches_exactly_the_planned_rows(tmp_path) -> None:
    plan = _plan(tmp_path)
    model_root = tmp_path / "models"
    seen: list[dict] = []

    def fetcher(rows, root):
        seen.extend(rows)
        for row in rows:
            target = root / row["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            os.truncate(target, row["size_bytes"] or 1)
        return {
            "artifacts": len(rows),
            "artifacts_fetched": len(rows),
            "missing_bytes": 0,
            "failed": {},
            "families_detail": {},
        }

    result = execute_plan(plan, model_root=model_root, fetcher=fetcher)

    assert result["ok"] is True
    assert {row["destination"] for row in seen} == set(plan["download"]["destinations"])
    assert result["downloaded"] == len(plan["download"]["entries"])
    assert result["verified"] is True
    assert "verified it on disk" in result["message"]


def test_execute_plan_does_not_claim_success_when_files_are_absent(tmp_path) -> None:
    plan = _plan(tmp_path)
    result = execute_plan(
        plan,
        model_root=tmp_path / "models",
        fetcher=lambda rows, root: {
            "artifacts": len(rows),
            "artifacts_fetched": len(rows),
            "missing_bytes": 0,
            "failed": {},
            "families_detail": {},
        },
    )
    assert result["ok"] is False
    assert result["verified"] is False
    assert "not complete on disk" in result["message"]


def test_execute_plan_with_nothing_to_do_is_a_no_op(tmp_path) -> None:
    _materialize(tmp_path / "models", _plan(tmp_path))
    plan = _plan(tmp_path)
    assert plan["status"] == PLAN_COMPLETE
    result = execute_plan(plan, model_root=tmp_path / "models", fetcher=pytest.fail)
    assert result["ok"] is True
    assert result["downloaded"] == 0


def test_fallback_policy_parsing() -> None:
    assert parse_fallback_policy(None) == "none"
    assert parse_fallback_policy("none") == "none"
    assert parse_fallback_policy("all") == "all"
    assert parse_fallback_policy("2") == 2
    assert parse_fallback_policy(3) == 3
    for bad in ("-1", "yes", "first"):
        with pytest.raises(FallbackPolicyError):
            parse_fallback_policy(bad)


def _load_cli():
    script = ROOT / "scripts/turbofit-onboard"
    loader = SourceFileLoader("turbofit_onboard_cli", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_to_download_the_whole_library() -> None:
    module = _load_cli()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--all"])
    assert excinfo.value.code == 2


def test_cli_rejects_a_bad_fallback_selector() -> None:
    module = _load_cli()
    with pytest.raises(SystemExit):
        module.main(["--include-fallback", "half"])


def test_plan_renders_for_a_human(tmp_path) -> None:
    text = render_plan(_plan(tmp_path, include_fallback=1))
    assert "TurboFit first run" in text
    assert "recommendation" in text
    assert "optional fallback-chain rungs" in text
    assert "total download:" in text


def _dashboard():
    pytest.importorskip("fastapi")
    spec = importlib.util.spec_from_file_location(
        "turbofit_dashboard_onboard_api", ROOT / "dashboard/plugin_api.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_payload_carries_the_first_run_download_plan() -> None:
    module = _dashboard()
    payload = module._fit_rows()

    assert payload["schema"] == "turbofit.fit-view/v1"
    onboard = payload["onboard"]
    assert onboard["url"] == "/api/plugins/turbofit/onboard"
    assert onboard["download_url"] == "/api/plugins/turbofit/onboard/download"
    assert isinstance(onboard["download_bytes"], int)
    assert onboard["download_gib"] == pytest.approx(onboard["download_bytes"] / GIB, abs=0.01)
    if onboard["ok"]:
        assert onboard["message"]
        assert onboard["placement"] is not None
        assert all("model" in rung for rung in onboard["rungs"])
    else:
        # A host with nothing to recommend must say so, not show a fake total.
        assert onboard["download_bytes"] == 0
        assert "measurements" in onboard["message"] or "cannot" in onboard["message"]


def test_onboard_plan_payload_is_the_same_shape_the_cli_prints() -> None:
    module = _dashboard()
    plan = module._onboard_plan("all")
    assert plan["schema"] == "turbofit.onboard-plan/v1"
    assert plan["fallback"]["policy"] == "all"
    assert set(plan["total"]) >= {"download_bytes", "download_gib", "artifacts_to_download"}


def test_onboard_routes_validate_the_fallback_selector() -> None:
    module = _dashboard()
    import asyncio

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as bad_policy:
        asyncio.run(module.get_onboard(include_fallback="half"))
    assert bad_policy.value.status_code == 422
    with pytest.raises(HTTPException) as bad_tier:
        asyncio.run(module.get_onboard(tier=0))
    assert bad_tier.value.status_code == 422
    with pytest.raises(HTTPException) as bad_body:
        asyncio.run(module.post_onboard_download({"include_fallback": True}))
    assert bad_body.value.status_code == 422

    plan = asyncio.run(module.get_onboard(include_fallback="none"))
    assert plan["schema"] == "turbofit.onboard-plan/v1"


def test_dashboard_bundle_exposes_the_download_trigger() -> None:
    bundle = (ROOT / "dashboard/dist/index.js").read_text()
    assert "/api/plugins/turbofit/onboard/download" in bundle
    assert "This run would download " in bundle
    assert "Also download the fallback-chain rungs" in bundle


def test_execute_plan_defaults_to_the_shared_shard_aware_materializer(monkeypatch, tmp_path) -> None:
    """The default fetcher is scripts/download-artifacts, not a second downloader."""
    from types import SimpleNamespace

    from turbofit_runtime import onboard as onboard_module

    plan = _plan(tmp_path)
    model_root = tmp_path / "models"
    calls: list[tuple[list[dict], Path]] = []

    def download_families(rows, *, root):
        calls.append((list(rows), root))
        for row in rows:
            target = root / row["destination"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            os.truncate(target, row["size_bytes"] or 1)
        return {
            "artifacts": len(rows),
            "artifacts_fetched": len(rows),
            "missing_bytes": 0,
            "failed": {},
            "families_detail": {},
        }

    monkeypatch.setattr(
        onboard_module, "_DOWNLOADER", SimpleNamespace(download_families=download_families)
    )
    result = execute_plan(plan, model_root=model_root)

    assert len(calls) == 1
    rows, root = calls[0]
    assert root == model_root
    assert {row["destination"] for row in rows} == set(plan["download"]["destinations"])
    assert result["ok"] is True
