"""Hermes-facing configuration and status helpers for the Turbofit plugin."""
from __future__ import annotations

import copy
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import urlopen

# Hermes loads directory plugins as packages without installing their Python
# projects.  Make the repository's src-layout importable from that real load
# path instead of relying on a developer-set PYTHONPATH.
PLUGIN_ROOT = Path(__file__).resolve().parent
MULTIMODAL_CATALOG = PLUGIN_ROOT / "references" / "multimodal-models.json"
SRC_ROOT = PLUGIN_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from turbofit_runtime.tailnet import publish_tailnet, tailnet_status


DEFAULT_BASE_URL = "http://127.0.0.1:8091/v1"
NOUS_FREE_FALLBACK_CHAIN = [
    {"provider": "nous", "model": "upstage/solar-pro4:free"},
    {"provider": "nous", "model": "meituan/longcat-2.0:free"},
    {"provider": "nous", "model": "tencent/hy3:free"},
    {"provider": "nous", "model": "poolside/laguna-s-2.1:free"},
    {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
]
SELECTION_PATH = Path(
    os.getenv(
        "TURBOFIT_SELECTION_STATE",
        Path.home() / ".config" / "turbofit" / "selection.json",
    )
)
RUNTIME_STATE_PATH = Path(
    os.getenv(
        "TURBOFIT_RUNTIME_STATE",
        Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "turbofit"
        / "runtime-state.json",
    )
)
_CONFIG_LOCK = threading.RLock()


def install_sirvir_profile(*, hermes_home: Path | None = None) -> dict[str, Any]:
    """Install or update Sirvir while preserving profile-owned user state."""
    source = PLUGIN_ROOT / "profiles" / "sirvir"
    if not source.is_dir():
        raise FileNotFoundError(f"missing bundled Sirvir profile: {source}")
    root = Path(hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    target = root / "profiles" / "sirvir"
    updated = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "SOUL.md", "INSTRUCTIONS.md", "config.yaml", "distribution.yaml"):
        source_file = source / name
        if not source_file.is_file():
            raise FileNotFoundError(f"incomplete bundled Sirvir profile: {source_file}")
        shutil.copyfile(source_file, target / name)
    return {
        "installed": True,
        "updated": updated,
        "profile": "sirvir",
        "path": str(target),
    }


def install_desktop_plugin(*, hermes_home: Path | None = None) -> dict[str, Any]:
    """Install or update Turbofit's native Hermes Desktop surface."""
    source = PLUGIN_ROOT / "desktop" / "plugin.js"
    if not source.is_file():
        raise FileNotFoundError(f"missing bundled desktop plugin: {source}")
    root = Path(hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    target = root / "desktop-plugins" / "turbofit" / "plugin.js"
    updated = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {
        "installed": True,
        "updated": updated,
        "plugin": "turbofit",
        "path": str(target),
    }


def launch_setup_screen() -> dict[str, Any]:
    """Launch Hermes Dashboard so the Turbofit setup page can be opened."""
    executable = shutil.which("hermes")
    if not executable:
        raise FileNotFoundError("hermes executable is not available")
    process = subprocess.Popen(
        [executable, "dashboard"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return {
        "launched": True,
        "pid": process.pid,
        "url": "http://127.0.0.1:9119/",
        "page": "Turbofit",
    }


def install_lemonade_runtime() -> dict[str, Any]:
    script = PLUGIN_ROOT / "scripts" / "install-lemonade-runtime"
    if not script.is_file():
        raise FileNotFoundError(f"missing Lemonade installer: {script}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC_ROOT), environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, str(script), "install"],
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
        env=environment,
    )
    try:
        payload = json.loads(result.stdout or result.stderr)
    except ValueError as exc:
        raise RuntimeError((result.stdout or result.stderr).strip() or "Lemonade install failed") from exc
    if result.returncode:
        raise RuntimeError(payload.get("error") or "Lemonade install failed")
    return payload


def install_native_runtime(backend: str = "auto") -> dict[str, Any]:
    script = PLUGIN_ROOT / "scripts" / "install-dspark-runtime"
    if not script.is_file():
        raise FileNotFoundError(f"missing native runtime installer: {script}")
    result = subprocess.run(
        [sys.executable, str(script), "install", "--backend", backend],
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    try:
        payload = json.loads(result.stdout or result.stderr)
    except ValueError as exc:
        raise RuntimeError((result.stdout or result.stderr).strip() or "native runtime install failed") from exc
    if result.returncode:
        raise RuntimeError(payload.get("error") or "native runtime install failed")
    return payload


def _is_local_or_tailnet(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".ts.net"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    tailnet = ipaddress.ip_network("100.64.0.0/10")
    return address.is_loopback or address in tailnet


def _validated_base_url(value: str | None) -> str:
    base_url = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    if parsed.scheme == "http" and not _is_local_or_tailnet(parsed.hostname):
        raise ValueError(
            "plain HTTP base_url must use loopback or a Tailscale address; use HTTPS otherwise"
        )
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def provider_definition(base_url: str | None = None) -> dict[str, Any]:
    """Return the canonical named custom-provider definition."""
    return {
        "name": "TurboFit",
        "api": _validated_base_url(base_url),
        "api_key": "not-needed",
        "transport": "chat_completions",
        "default_model": "auto",
        "models": {"auto": {}, "active:main": {}, "active:aux": {}},
    }


def configure_hermes(
    config: Mapping[str, Any],
    *,
    primary: bool = False,
    fallback: bool | None = None,
    fallback_chain: list[Mapping[str, Any]] | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return a copied Hermes config with Turbofit registered idempotently."""
    updated = copy.deepcopy(dict(config))
    provider = provider_definition(base_url)

    raw_providers = updated.get("providers")
    providers = copy.deepcopy(dict(raw_providers)) if isinstance(raw_providers, Mapping) else {}
    providers["turbofit"] = provider
    updated["providers"] = providers

    # Remove only the obsolete duplicate if an older installation wrote one.
    existing = updated.get("custom_providers")
    legacy = list(existing) if isinstance(existing, list) else []
    legacy = [
        item
        for item in legacy
        if not (isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit")
    ]
    if legacy:
        updated["custom_providers"] = legacy
    else:
        updated.pop("custom_providers", None)

    if primary:
        raw_model = updated.get("model")
        model: dict[str, Any] = copy.deepcopy(dict(raw_model)) if isinstance(raw_model, Mapping) else {}
        model["provider"] = "custom:turbofit"
        model["default"] = "auto"
        updated["model"] = model

    if fallback_chain is not None:
        normalized_chain: list[dict[str, str]] = []
        if not isinstance(fallback_chain, list):
            raise ValueError("fallback_chain must be a list")
        for index, item in enumerate(fallback_chain):
            if not isinstance(item, Mapping) or set(item) != {"provider", "model"}:
                raise ValueError(
                    f"fallback_chain[{index}] must contain only provider and model"
                )
            provider_name = item.get("provider")
            model_name = item.get("model")
            if (
                not isinstance(provider_name, str)
                or not provider_name.strip()
                or not isinstance(model_name, str)
                or not model_name.strip()
            ):
                raise ValueError(
                    f"fallback_chain[{index}] provider and model must be non-empty strings"
                )
            normalized_chain.append({
                "provider": provider_name.strip(),
                "model": model_name.strip(),
            })
        updated["fallback_providers"] = normalized_chain

    if fallback is not None:
        raw_chain = updated.get("fallback_providers")
        chain = list(raw_chain) if isinstance(raw_chain, list) else []
        chain = [
            item
            for item in chain
            if not (
                isinstance(item, Mapping)
                and str(item.get("provider", "")).lower() in {"custom:turbofit", "turbofit"}
            )
        ]
        if fallback:
            chain.append(
                {
                    "provider": "custom:turbofit",
                    "model": "auto",
                }
            )
        updated["fallback_providers"] = chain

    return updated


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _provider_flags(config: Mapping[str, Any]) -> tuple[bool, bool, str]:
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    primary = str(model.get("provider", "")).lower() == "custom:turbofit"
    fallback = any(
        isinstance(item, Mapping)
        and str(item.get("provider", "")).lower() in {"custom:turbofit", "turbofit"}
        for item in (config.get("fallback_providers") or [])
    )
    named = config.get("providers") if isinstance(config.get("providers"), Mapping) else {}
    modern = named.get("turbofit") if isinstance(named.get("turbofit"), Mapping) else None
    legacy = config.get("custom_providers") or []
    legacy_match = next((
        item for item in legacy
        if isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit"
    ), None)
    modern_map: Mapping[str, Any] = modern or {}
    legacy_map: Mapping[str, Any] = legacy_match or {}
    endpoint = str(
        modern_map.get("api")
        or modern_map.get("base_url")
        or legacy_map.get("base_url")
        or DEFAULT_BASE_URL
    )
    return primary, fallback, endpoint


def gateway_health(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.5) -> dict[str, Any]:
    endpoint = f"{_validated_base_url(base_url)}/models"
    try:
        with urlopen(endpoint, timeout=timeout) as response:  # noqa: S310 - user-configured local/Tailscale endpoint
            decoded = json.loads(response.read().decode("utf-8"))
        payload = decoded if isinstance(decoded, dict) else {}
        data = payload.get("data")
        ids = [item.get("id") for item in data if isinstance(item, dict)] if isinstance(data, list) else []
        return {"reachable": True, "endpoint": endpoint, "models": ids}
    except Exception as exc:
        return {"reachable": False, "endpoint": endpoint, "error": str(exc)}


def status_snapshot(config: Mapping[str, Any], *, probe: bool = True) -> dict[str, Any]:
    primary, fallback, endpoint = _provider_flags(config)
    selection = _load_json(SELECTION_PATH)
    runtime = _load_json(RUNTIME_STATE_PATH)
    named_providers = config.get("providers")
    provider_map = named_providers if isinstance(named_providers, Mapping) else {}
    fallback_chain = [
        {"provider": str(item["provider"]), "model": str(item["model"])}
        for item in (config.get("fallback_providers") or [])
        if isinstance(item, Mapping)
        and isinstance(item.get("provider"), str)
        and item.get("provider")
        and isinstance(item.get("model"), str)
        and item.get("model")
    ]
    return {
        "ok": True,
        "provider": {
            "registered": "turbofit" in provider_map or any(
                isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit"
                for item in (config.get("custom_providers") or [])
            ),
            "primary": primary,
            "fallback": fallback,
            "base_url": endpoint,
            "fallback_chain": fallback_chain,
        },
        "selection": selection,
        "runtime": runtime,
        "gateway": gateway_health(endpoint) if probe else {"reachable": None, "endpoint": endpoint},
        "tailnet": tailnet_status(),
        "platform": {"os": os.name, "sys_platform": sys.platform},
    }


def select_profile(profile: str) -> dict[str, Any]:
    requested = str(profile or "auto").strip()
    combination_id = None
    if requested != "auto" and not requested.startswith("hardware-") and not requested.startswith("manual-"):
        combination_id = requested
        requested = prepare_manual_profile(requested)
    script = PLUGIN_ROOT / "scripts" / "turbofit-runtime"
    if not script.is_file():
        raise FileNotFoundError(f"missing runtime selector: {script}")
    result = subprocess.run(
        [sys.executable, str(script), "set", requested],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    try:
        parsed = json.loads(result.stdout or result.stderr)
        payload: dict[str, Any] = parsed if isinstance(parsed, dict) else {"output": parsed}
    except ValueError:
        payload = {"output": (result.stdout or result.stderr).strip()}
    if result.returncode:
        raise RuntimeError(payload.get("error") or payload.get("output") or "profile selection failed")
    if combination_id is not None:
        payload["combination_id"] = combination_id
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", "turbofit-controller.service"],
            check=False,
            timeout=15,
        ).returncode == 0
        if active:
            restarted = subprocess.run(
                ["systemctl", "--user", "restart", "turbofit-controller.service"],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
            if restarted.returncode:
                raise RuntimeError((restarted.stderr or "failed to reload Turbofit controller").strip())
        payload["controller_restarted"] = active
    return payload


def combination_snapshot() -> dict[str, Any]:
    """Return every current exact configuration with fit or incompatibility evidence."""
    script = PLUGIN_ROOT / "scripts" / "turbofit-runtime-recommend"
    result = subprocess.run(
        [sys.executable, str(script), "--json", "--limit", "10000", "--prefer", "balanced"],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
    )
    try:
        rows = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError((result.stderr or "failed to enumerate exact combinations").strip()) from exc
    if result.returncode or not isinstance(rows, list):
        raise RuntimeError((result.stderr or "failed to enumerate exact combinations").strip())
    from turbofit_runtime.catalog_campaign import build_configuration_index
    from turbofit_runtime.model_catalog import ModelCatalog

    catalog = ModelCatalog.load(PLUGIN_ROOT / "references/model-catalog.json")
    configurations = json.loads(
        (PLUGIN_ROOT / "references/configuration-matrix.json").read_text(encoding="utf-8")
    )
    index = build_configuration_index(configurations, catalog)
    models = {item.id: item for item in catalog.models}
    for row in rows:
        configuration = index.get(str(row.get("profile")))
        if configuration is None:
            continue
        main = models[str(configuration["main"])]
        raw_aux = str(configuration["auxiliary"])
        auxiliary = main if raw_aux == "auto" else models[raw_aux]
        row.update({
            "main_catalog_id": main.id,
            "main_name": main.name,
            "main_quantization": main.quantization,
            "main_runtime_features": list(main.runtime_features),
            "aux_catalog_id": raw_aux,
            "aux_name": auxiliary.name if raw_aux != "auto" else f"Shared main ({main.name})",
            "aux_quantization": auxiliary.quantization,
            "aux_runtime_features": list(auxiliary.runtime_features),
        })
    return {
        "ok": True,
        "combinations": rows,
        "compatible": sum(bool(row.get("fit")) for row in rows if isinstance(row, Mapping)),
        "total": len(rows),
        "policy": "current-recipe-evidence-and-physical-fit",
    }


def prepare_manual_profile(profile_id: str) -> str:
    """Materialize one exact validated combination for the adaptive controller."""
    from turbofit_runtime.catalog_campaign import build_configuration_index
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.manual_profiles import write_manual_profile
    from turbofit_runtime.model_catalog import ModelCatalog
    from turbofit_runtime.recipes import RecipeBook

    choices = combination_snapshot()["combinations"]
    selected = next((item for item in choices if item.get("profile") == profile_id), None)
    if selected is None:
        raise ValueError(f"unknown or stale exact combination: {profile_id}")
    if not selected.get("fit"):
        raise ValueError(str(selected.get("fit_reason") or f"combination does not fit: {profile_id}"))
    profiles = json.loads(
        (PLUGIN_ROOT / "references/successful-runtime-profiles.json").read_text(encoding="utf-8")
    ).get("profiles") or {}
    entry = profiles.get(profile_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"missing exact runtime profile: {profile_id}")
    configurations = json.loads(
        (PLUGIN_ROOT / "references/configuration-matrix.json").read_text(encoding="utf-8")
    )
    catalog = ModelCatalog.load(PLUGIN_ROOT / "references/model-catalog.json")
    configuration = build_configuration_index(configurations, catalog).get(profile_id)
    if configuration is None:
        raise ValueError(f"exact combination is not canonical: {profile_id}")
    recipe = RecipeBook.load(PLUGIN_ROOT / "references/model-recipes.json").resolve_catalog_configuration(configuration)
    state = _load_json(PLUGIN_ROOT / "references/catalog-campaign-state.json") or {"rows": {}}
    record = (state.get("rows") or {}).get(profile_id) or {}
    enriched = {**entry, "production_recipe_sha256": record.get("recipe_sha256")}
    generated_id = f"manual-{profile_id}"
    write_manual_profile(
        Path.home() / ".config/turbofit",
        profile_id=generated_id,
        profile_entry=enriched,
        recipe=recipe,
        hardware=probe_hardware(),
    )
    return generated_id


def recommendation_snapshot(
    preference: str | None = None,
    *,
    limit: int = 3,
) -> dict[str, Any]:
    """Rescan physical hardware and return evidence-backed user choices."""
    from turbofit_runtime.hardware import probe_hardware

    requested = str(preference or "").strip().lower() or None
    if requested == "quality":
        requested = "intelligence"
    elif requested == "context":
        requested = "balanced"
    if requested not in {None, "intelligence", "balanced", "speed"}:
        raise ValueError("preference must be intelligence, balanced, or speed")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer from 1 to 20")

    hardware = probe_hardware()
    script = PLUGIN_ROOT / "scripts" / "turbofit-runtime-recommend"
    if not script.is_file():
        raise FileNotFoundError(f"missing recommendation engine: {script}")
    preferences = (requested,) if requested else ("intelligence", "balanced", "speed")
    recommendations: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for name in preferences:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--json",
                "--fit-only",
                "--limit",
                str(limit),
                "--prefer",
                name,
            ],
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
        )
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            payload = []
        recommendations[name] = payload if isinstance(payload, list) else []
        if result.returncode and not recommendations[name]:
            errors[name] = (result.stderr or "no evidence-backed configuration fits").strip()
    return {
        "ok": any(recommendations.values()),
        "hardware": {
            **asdict(hardware),
            "topology_key": hardware.topology_key,
            "total_vram_mb": hardware.total_vram_mb,
            "total_usable_memory_mb": hardware.total_usable_memory_mb,
        },
        "requested_preference": requested,
        "preferences": list(preferences),
        "recommendations": recommendations,
        "errors": errors,
    }


def hardware_tier_snapshot() -> dict[str, Any]:
    """Return all eight tier candidates with explicit evidence and score status."""
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.tier_report import build_tier_report

    return build_tier_report(PLUGIN_ROOT, probe_hardware())


def multimodal_snapshot(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return hardware-fit multimodal options plus the user's selections."""
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.multimodal import MultimodalCatalog, recommend_multimodal

    payload = recommend_multimodal(
        hardware=probe_hardware(),
        catalog=MultimodalCatalog.load(MULTIMODAL_CATALOG),
    )
    selected: dict[str, Any] = {}
    if isinstance(config, Mapping):
        turbofit = config.get("turbofit")
        if isinstance(turbofit, Mapping):
            multimodal = turbofit.get("multimodal")
            if isinstance(multimodal, Mapping) and isinstance(multimodal.get("selected"), Mapping):
                selected = dict(multimodal["selected"])
    payload["selected"] = selected
    return payload


def apply_configuration(
    *,
    primary: bool,
    fallback: bool | None,
    fallback_chain: list[Mapping[str, Any]] | None = None,
    multimodal: Mapping[str, str] | None = None,
    profile: str | None,
    base_url: str | None,
    install_sirvir: bool = False,
    install_desktop: bool = False,
    install_lemonade: bool = False,
    install_native: bool = False,
    publish_tailnet_routes: bool = False,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
) -> dict[str, Any]:
    from hermes_cli.config import load_config, save_config

    with _CONFIG_LOCK:
        sirvir = install_sirvir_profile() if install_sirvir else None
        desktop = install_desktop_plugin() if install_desktop else None
        lemonade = install_lemonade_runtime() if install_lemonade else None
        native = install_native_runtime() if install_native else None
        selected = select_profile(profile) if profile else None
        publication = (
            publish_tailnet(
                dashboard_local_port=dashboard_local_port,
                provider_local_port=provider_local_port,
                dashboard_https_port=dashboard_https_port,
                provider_https_port=provider_https_port,
            )
            if publish_tailnet_routes
            else None
        )
        effective_base_url = publication["provider_base_url"] if publication else base_url
        configured = configure_hermes(
            load_config(),
            primary=primary,
            fallback=fallback,
            fallback_chain=list(NOUS_FREE_FALLBACK_CHAIN) if fallback_chain is None else fallback_chain,
            base_url=effective_base_url,
        )
        if multimodal is not None:
            from turbofit_runtime.multimodal import MultimodalCatalog, configure_multimodal

            configured = configure_multimodal(
                configured,
                selections=multimodal,
                catalog=MultimodalCatalog.load(MULTIMODAL_CATALOG),
            )
        save_config(configured, merge_existing=False)
    return {
        "ok": True,
        "configured": True,
        "selection": selected,
        "tailnet": publication,
        "sirvir": sirvir,
        "desktop_plugin": desktop,
        "lemonade": lemonade,
        "native_runtime": native,
        "restart_required": True,
    }


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    try:
        from hermes_cli.config import load_config

        return json.dumps(status_snapshot(load_config()))
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def handle_configure(args: dict[str, Any], **_: Any) -> str:
    try:
        primary = args.get("primary", False)
        fallback = args.get("fallback") if "fallback" in args else None
        publish_routes = args.get("publish_tailnet", False)
        install_sirvir = args.get("install_sirvir", False)
        install_desktop = args.get("install_desktop", False)
        install_lemonade = args.get("install_lemonade", False)
        install_native = args.get("install_native", False)
        if (
            not isinstance(primary, bool)
            or (fallback is not None and not isinstance(fallback, bool))
            or not isinstance(publish_routes, bool)
            or not isinstance(install_sirvir, bool)
            or not isinstance(install_desktop, bool)
            or not isinstance(install_lemonade, bool)
            or not isinstance(install_native, bool)
        ):
            raise ValueError("setup switches must be booleans")
        profile = args.get("profile")
        base_url = args.get("base_url")
        fallback_chain = args.get("fallback_chain")
        multimodal = args.get("multimodal")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("profile must be a string")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        if fallback_chain is not None and not isinstance(fallback_chain, list):
            raise ValueError("fallback_chain must be a list")
        if multimodal is not None and not isinstance(multimodal, Mapping):
            raise ValueError("multimodal must be an object")
        ports = {
            name: args.get(name, default)
            for name, default in {
                "dashboard_local_port": 9127,
                "provider_local_port": 8091,
                "dashboard_https_port": 9444,
                "provider_https_port": 9443,
            }.items()
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ports.values()):
            raise ValueError("Tailscale ports must be integers")
        result = apply_configuration(
            primary=primary,
            fallback=fallback,
            fallback_chain=fallback_chain,
            multimodal=multimodal,
            profile=profile,
            base_url=base_url,
            publish_tailnet_routes=publish_routes,
            install_sirvir=install_sirvir,
            install_desktop=install_desktop,
            install_lemonade=install_lemonade,
            install_native=install_native,
            **ports,
        )
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})
