"""Backend API for the Turbofit Hermes dashboard extension."""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from fastapi import APIRouter, HTTPException
except ImportError as exc:  # the dashboard is an optional extra, the plugin is not
    raise ImportError(
        "the TurboFit dashboard needs fastapi: pip install 'turbofit[dashboard]'. "
        "The plugin itself (and /turbofit) works without it."
    ) from exc


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PLUGIN_ROOT, PLUGIN_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plugin_tools import (  # noqa: E402
    apply_configuration,
    combination_snapshot,
    hardware_tier_snapshot,
    multimodal_snapshot,
    recommendation_snapshot,
    status_snapshot,
)
from product_ops import (  # noqa: E402
    available_models,
    local_model_replacement,
    retire_local_model,
    serve_status,
    serve_tailnet,
    shift_configuration,
    smoke_local_runtime,
    update_products,
)


router = APIRouter()


def _config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    return load_config()


def _profile_rows() -> list[dict[str, Any]]:
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.runtime_cli import run
    from turbofit_runtime.selection import ProfileCatalog

    import io

    catalog = ProfileCatalog.from_paths(sorted((PLUGIN_ROOT / "runtime-profiles").glob("*gb.yaml")))
    output = io.StringIO()
    code = run(
        ["list"],
        catalog=catalog,
        hardware=probe_hardware(),
        selection_path=Path.home() / ".config" / "turbofit" / "selection.json",
        output=output,
    )
    payload = json.loads(output.getvalue())
    if code:
        raise RuntimeError(payload.get("error", "failed to list profiles"))
    return payload


def _backend_status() -> dict[str, Any]:
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.freetoken import FreeTokenClient, probe_freetoken_compatibility
    from turbofit_runtime.runtime_backends import LemonadeClient

    try:
        fingerprint = probe_hardware()
        hardware = asdict(fingerprint)
        local_backend = fingerprint.devices[0].backend if fingerprint.devices else "cpu"
        hardware_error = None
    except Exception as exc:
        hardware = None
        local_backend = "unknown"
        hardware_error = type(exc).__name__
    try:
        lemonade = {"available": True, "health": LemonadeClient().health()}
    except Exception as exc:
        lemonade = {"available": False, "error": type(exc).__name__}
    compatibility = probe_freetoken_compatibility()
    try:
        freetoken = {
            "available": True,
            "status": "candidate",
            "health": FreeTokenClient().health(),
            "compatibility": asdict(compatibility),
            "auto_promote": False,
        }
    except Exception as exc:
        freetoken = {
            "available": False,
            "status": compatibility.status,
            "error": type(exc).__name__,
            "compatibility": asdict(compatibility),
            "auto_promote": False,
        }
    engines = []
    try:
        from dataclasses import asdict as _asdict
        from turbofit_runtime.engine_check import detect_driver_major, detect_wsl, probe_engines
        from turbofit_runtime.hardware import probe_hardware as _probe

        live = _probe()
        engines = [
            {
                "engine_id": item.engine_id,
                "display_name": item.display_name,
                "compatible": item.compatible,
                "installed": item.installed,
                "running": item.running,
                "eligible": item.eligible,
                "reason": item.reason,
                "support_mode": item.support_mode,
            }
            for item in probe_engines(live, driver_major=detect_driver_major(), is_wsl=detect_wsl())
        ]
    except Exception as exc:
        engines = [{"error": type(exc).__name__}]
    return {
        "hardware": hardware,
        "hardware_error": hardware_error,
        "local_backend": local_backend,
        "lemonade": lemonade,
        "freetoken": freetoken,
        "engines": engines,
        "supported": ["cuda", "rocm", "metal", "cpu", "lemonade", "freetoken-candidate", "llama.cpp", "mlx", "sglang", "vllm", "turbohaul-manager"],
    }


def _tournament_rows() -> dict[str, Any]:
    from turbofit_runtime.executor import production_recipe_sha256
    from turbofit_runtime.model_catalog import ModelCatalog
    from turbofit_runtime.recipes import RecipeBook
    from turbofit_runtime.schema import MatrixRow
    from turbofit_runtime.tier_tournament import load_tournaments

    configurations = json.loads((PLUGIN_ROOT / "references/configuration-matrix.json").read_text(encoding="utf-8"))
    tournaments = load_tournaments(PLUGIN_ROOT / "references/hardware-tier-tournaments.json", configurations)
    catalog = ModelCatalog.load(PLUGIN_ROOT / "references/model-catalog.json")
    recipes = RecipeBook.load(PLUGIN_ROOT / "references/model-recipes.json")
    by_model = {item.id: item for item in catalog.models}
    display_ids = {}
    current_recipe = {}
    for item in configurations["rows"]:
        main = by_model[item["main"]]
        aux_id = item["auxiliary"]
        aux_name = "auto" if aux_id == "auto" else by_model[aux_id].name
        display_id = MatrixRow.make_id(main.name, aux_name, int(item["context"]))
        display_ids[item["id"]] = display_id
        catalog_item = dict(item)
        catalog_item["id"] = display_id
        current_recipe[item["id"]] = production_recipe_sha256(
            recipes.resolve_catalog_configuration(catalog_item), catalog_item,
        )
    state_path = PLUGIN_ROOT / "references/catalog-campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"rows": {}}
    campaign_rows = state.get("rows") or {}
    report = hardware_tier_snapshot()
    report_by_capacity = {item["capacity_gb"]: item for item in report["tiers"]}
    tiers = []
    for tier in tournaments["tiers"]:
        candidates = []
        for item in tier["candidates"]:
            record = campaign_rows.get(display_ids[item], {})
            current = record.get("recipe_sha256") == current_recipe[item]
            candidates.append({
                "configuration": item,
                "status": record.get("status", "pending") if current else "pending",
                "current_recipe": current,
            })
        reported = report_by_capacity[tier["vram_gb"]]
        measured = reported["recommendations"]["measured_winner"]
        winner = None if measured is None else {
            "configuration": measured["configuration_id"],
            "evidence": measured["fit"]["evidence"],
            "measured_tps": measured["measured_tps"],
            "intelligence_score": measured["intelligence_score"],
        }
        tiers.append({
            **tier,
            "candidate_ranking_winner": tier["winner"],
            "winner": winner,
            "status": reported["status"],
            "evidence_policy": "current-recipe-and-exact-topology-only",
            "candidates": candidates,
        })
    return {"ok": True, "ranking": tournaments["ranking"], "tiers": tiers}


def _auxiliary_tiers() -> dict[str, Any]:
    payload = json.loads(
        (PLUGIN_ROOT / "references/auxiliary-tier-recommendations.json").read_text(encoding="utf-8")
    )
    return {"ok": True, **payload}


@router.get("/status")
async def get_status() -> dict[str, Any]:
    return await asyncio.to_thread(status_snapshot, _config())


@router.get("/profiles")
async def get_profiles() -> dict[str, Any]:
    try:
        return {"ok": True, "profiles": await asyncio.to_thread(_profile_rows)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/combinations")
async def get_combinations() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(combination_snapshot)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/backends")
async def get_backends() -> dict[str, Any]:
    return await asyncio.to_thread(_backend_status)


@router.get("/tournaments")
async def get_tournaments() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_tournament_rows)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/hardware-tiers")
async def get_hardware_tiers() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(hardware_tier_snapshot)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _leaderboard_rows() -> dict[str, Any]:
    from turbofit_runtime.leaderboard import build_leaderboard, load_engine_matrix, load_tier_report

    report = load_tier_report(PLUGIN_ROOT / "references" / "hardware-tier-report.json")
    matrix = load_engine_matrix(PLUGIN_ROOT / "references" / "engine-serve-matrix.json")
    return build_leaderboard(report, matrix)


@router.get("/leaderboard")
async def get_leaderboard() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_leaderboard_rows)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


ONBOARD_URL = "/api/plugins/turbofit/onboard"
ONBOARD_DOWNLOAD_URL = "/api/plugins/turbofit/onboard/download"


def _fit_rows(tier_gb: int | None = None) -> dict[str, Any]:
    """What to run on this machine, and what is missing to run it."""
    from turbofit_runtime.fit_view import load_fit_view

    payload = load_fit_view(PLUGIN_ROOT, tier_gb=tier_gb)
    # The fit card's job is to say "this is what you will download, N GiB" and
    # let the user start it, so the first-run plan travels with the answer.
    try:
        payload["onboard"] = _onboard_summary(tier_gb=tier_gb)
    except Exception as exc:  # a plan failure must not hide the fit answer
        payload["onboard"] = {
            "ok": False,
            "status": "unavailable",
            "download_bytes": 0,
            "message": f"the first-run download plan could not be built ({type(exc).__name__}: {exc}).",
            "url": ONBOARD_URL,
            "download_url": ONBOARD_DOWNLOAD_URL,
        }
    return payload


def _onboard_plan(include_fallback: str | int = "none", tier_gb: int | None = None) -> dict[str, Any]:
    from turbofit_runtime.onboard import build_plan

    return build_plan(PLUGIN_ROOT, include_fallback=include_fallback, tier_gb=tier_gb)


def _onboard_summary(*, tier_gb: int | None = None) -> dict[str, Any]:
    """The compact form of the plan the fit card renders."""
    plan = _onboard_plan("none", tier_gb)
    recommendation = plan.get("recommendation") or {}
    fallback = plan.get("fallback") or {}
    return {
        "ok": bool(plan.get("ok")),
        "status": plan.get("status"),
        "tier_gb": (plan.get("tier") or {}).get("hardware_tier_gb"),
        "measured": (plan.get("tier") or {}).get("measured"),
        "download_bytes": int((plan.get("total") or {}).get("download_bytes") or 0),
        "download_gib": float((plan.get("total") or {}).get("download_gib") or 0.0),
        "artifacts_to_download": int((plan.get("total") or {}).get("artifacts_to_download") or 0),
        "artifacts_present": int((plan.get("total") or {}).get("artifacts_present") or 0),
        "main": recommendation.get("main"),
        "main_family": recommendation.get("main_family"),
        "placement": recommendation.get("placement"),
        "rungs": [
            {
                "index": rung.get("index"),
                "model": rung.get("model"),
                "download_bytes": int(rung.get("incremental_bytes") or 0),
                "evidence": rung.get("evidence"),
                "selectable": bool(rung.get("selectable")),
            }
            for rung in fallback.get("rungs") or []
            if rung.get("selectable")
        ],
        "message": plan.get("message"),
        "next": plan.get("next") or [],
        "url": ONBOARD_URL,
        "download_url": ONBOARD_DOWNLOAD_URL,
    }


@router.get("/fit")
async def get_fit(tier: int | None = None) -> dict[str, Any]:
    if tier is not None and not 1 <= tier <= 4096:
        raise HTTPException(status_code=422, detail="tier must be a GB value from 1 to 4096")
    try:
        return await asyncio.to_thread(_fit_rows, tier)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/onboard")
async def get_onboard(include_fallback: str = "none", tier: int | None = None) -> dict[str, Any]:
    """The first-run plan: scan, recommend, and the exact bytes it would fetch."""
    from turbofit_runtime.onboard import FallbackPolicyError, parse_fallback_policy

    if tier is not None and not 1 <= tier <= 4096:
        raise HTTPException(status_code=422, detail="tier must be a GB value from 1 to 4096")
    try:
        parse_fallback_policy(include_fallback)
    except FallbackPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return await asyncio.to_thread(_onboard_plan, include_fallback, tier)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/onboard/download")
async def post_onboard_download(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch exactly the plan the user was shown -- nothing else."""
    from turbofit_runtime.onboard import FallbackPolicyError, execute_plan, parse_fallback_policy

    payload = body or {}
    include_fallback = payload.get("include_fallback", "none")
    tier = payload.get("tier")
    if tier is not None and (isinstance(tier, bool) or not isinstance(tier, int) or not 1 <= tier <= 4096):
        raise HTTPException(status_code=422, detail="tier must be a GB value from 1 to 4096")
    if not isinstance(include_fallback, (str, int)) or isinstance(include_fallback, bool):
        raise HTTPException(status_code=422, detail="include_fallback must be none, all, or a rung number")
    try:
        parse_fallback_policy(include_fallback)
    except FallbackPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        plan = await asyncio.to_thread(_onboard_plan, include_fallback, tier)
        return await asyncio.to_thread(execute_plan, plan)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/auxiliary-tiers")
async def get_auxiliary_tiers() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_auxiliary_tiers)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/recommendations")
async def get_recommendations(preference: str | None = None) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(recommendation_snapshot, preference)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/multimodal")
async def get_multimodal() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = await asyncio.to_thread(load_config)
        return await asyncio.to_thread(multimodal_snapshot, config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/configure")
async def configure(body: dict[str, Any]) -> dict[str, Any]:
    primary = body.get("primary", False)
    fallback = body.get("fallback") if "fallback" in body else None
    fallback_chain = body.get("fallback_chain")
    multimodal = body.get("multimodal")
    profile = body.get("profile")
    base_url = body.get("base_url")
    publish_tailnet_routes = body.get("publish_tailnet", True)
    install_turbosouth = body.get("install_turbosouth", body.get("install_turbosouth", False))
    install_turbosouth = body.get("install_turbosouth", install_turbosouth)
    install_desktop = body.get("install_desktop", False)
    install_lemonade = body.get("install_lemonade", False)
    install_native = body.get("install_native", False)
    install_freetoken = body.get("install_freetoken", False)
    port_defaults = {
        "dashboard_local_port": 9127,
        "provider_local_port": 8091,
        "dashboard_https_port": 9444,
        "provider_https_port": 9443,
    }
    ports = {name: body.get(name, default) for name, default in port_defaults.items()}
    if (
        not isinstance(primary, bool)
        or (fallback is not None and not isinstance(fallback, bool))
        or not isinstance(publish_tailnet_routes, bool)
        or not isinstance(install_turbosouth, bool)
        or not isinstance(install_desktop, bool)
        or not isinstance(install_lemonade, bool)
        or not isinstance(install_native, bool)
        or not isinstance(install_freetoken, bool)
    ):
        raise HTTPException(status_code=422, detail="setup switches must be booleans")
    if profile is not None and not isinstance(profile, str):
        raise HTTPException(status_code=422, detail="profile must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise HTTPException(status_code=422, detail="base_url must be a string")
    if fallback_chain is not None and not isinstance(fallback_chain, list):
        raise HTTPException(status_code=422, detail="fallback_chain must be a list")
    if multimodal is not None and not isinstance(multimodal, dict):
        raise HTTPException(status_code=422, detail="multimodal must be an object")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535 for value in ports.values()):
        raise HTTPException(status_code=422, detail="Tailscale ports must be integers from 1 to 65535")
    try:
        return await asyncio.to_thread(
            apply_configuration,
            primary=primary,
            fallback=fallback,
            fallback_chain=fallback_chain,
            multimodal=multimodal,
            profile=profile,
            base_url=base_url,
            publish_tailnet_routes=publish_tailnet_routes,
            install_turbosouth=install_turbosouth,
            install_desktop=install_desktop,
            install_lemonade=install_lemonade,
            install_native=install_native,
            install_freetoken=install_freetoken,
            **ports,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/update")
async def post_update() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(update_products)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/smoke")
async def post_smoke(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = body or {}
    timeout_seconds = payload.get("timeout_seconds", 300.0)
    try:
        return await asyncio.to_thread(smoke_local_runtime, timeout_seconds=timeout_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=503, detail="local smoke failed; inspect Turbofit logs") from None


@router.get("/audition")
async def get_audition(main: str = "maple-preview-tq2", aux: str = "maple-preview-tq2", context: int = 65536) -> dict[str, Any]:
    try:
        from turbofit_runtime.engine_check import audition_pair, detect_driver_major, detect_wsl
        from turbofit_runtime.hardware import probe_hardware

        rows = await asyncio.to_thread(
            audition_pair,
            probe_hardware(),
            main_alias=main,
            aux_alias=aux,
            context=context,
        )
        return {"ok": True, "main": main, "aux": aux, "context": context, "engines": rows, "driver_major": detect_driver_major(), "wsl": detect_wsl()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/local-models")
@router.get("/available-models")
async def get_available_models() -> dict[str, Any]:
    """Every catalog model the user can pick, default-first, with presence."""
    try:
        return await asyncio.to_thread(available_models)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models")
async def get_local_models() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(local_model_replacement)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/retire-model")
async def post_retire_model(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = body or {}
    family = payload.get("family") or ""
    action = payload.get("action") or ""
    if not isinstance(family, str) or not isinstance(action, str):
        raise HTTPException(status_code=422, detail="family and action must be strings")
    try:
        return await asyncio.to_thread(retire_local_model, family, action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/shift")
async def post_shift(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = body or {}
    target = payload.get("target") or payload.get("shift") or ""
    if not isinstance(target, str):
        raise HTTPException(status_code=422, detail="target must be a string")
    try:
        return await asyncio.to_thread(shift_configuration, target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/serve")
async def post_serve(body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = body or {}
    try:
        return await asyncio.to_thread(
            serve_tailnet,
            dashboard_local_port=int(payload.get("dashboard_local_port") or 9127),
            provider_local_port=int(payload.get("provider_local_port") or 8091),
            dashboard_https_port=int(payload.get("dashboard_https_port") or 9444),
            provider_https_port=int(payload.get("provider_https_port") or 9443),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/serve")
async def get_serve() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(serve_status)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
