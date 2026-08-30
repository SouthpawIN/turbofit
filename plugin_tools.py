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

import yaml

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
# Looked up from Hermes-Agent, not guessed:
# - stealth/ox-alpha is the official Nous curated "Ox Alpha" model
#   ($0/$0 on the Portal; same weights as OpenCode Zen x-preview-f-free).
# - The other four are live Portal freeRecommendedModels as of 2026-08-22
#   that Hermes also documents as the free Portal tail (stepfun + hy3 +
#   both Laguna :free slugs). No Hermes-branded models. No NVIDIA NIM.
NOUS_FREE_LABELS = {
    "stealth/ox-alpha": "Ox Alpha",
    "stepfun/step-3.7-flash:free": "Step 3.7 Flash",
    "tencent/hy3:free": "Tencent HY3",
    "poolside/laguna-s-2.1:free": "Laguna S 2.1",
    "poolside/laguna-xs-2.1:free": "Laguna XS 2.1",
}
SUBSCRIPTION_LABELS = {
    "nous": "Nous (subscription)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "openrouter": "OpenRouter",
    "xai": "xAI",
    "groq": "Groq",
    "google": "Google",
    "gemini": "Gemini",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "together": "Together",
    "fireworks": "Fireworks",
}
NOUS_FREE_FALLBACK_CHAIN = [
    {"provider": "nous", "model": "stealth/ox-alpha"},
    {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
    {"provider": "nous", "model": "tencent/hy3:free"},
    {"provider": "nous", "model": "poolside/laguna-s-2.1:free"},
    {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"},
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
    """Install or update Sirvir from its canonical GitHub distribution."""
    executable = shutil.which("hermes")
    if not executable:
        raise FileNotFoundError("hermes executable is not available")
    root = Path(hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    target = root / "profiles" / "sirvir"
    updated = (target / "distribution.yaml").is_file()
    command = (
        [executable, "profile", "update", "sirvir", "--yes"]
        if updated else
        [
            executable,
            "profile",
            "install",
            "https://github.com/SouthpawIN/sirvir.git",
            "--name",
            "sirvir",
            "--yes",
        ]
    )
    environment = {**os.environ, "HERMES_HOME": str(root)}
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "Sirvir profile installation failed").strip())
    manifest_path = target / "distribution.yaml"
    if not manifest_path.is_file():
        raise RuntimeError("Hermes reported success but the Sirvir profile was not installed")
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        raise RuntimeError("installed Sirvir distribution manifest is invalid") from exc
    return {
        "installed": True,
        "updated": updated,
        "profile": "sirvir",
        "path": str(target),
        "source": "https://github.com/SouthpawIN/sirvir.git",
        "version": str(manifest.get("version") or "unknown"),
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
    activation = activate_slash_commands(hermes_home=root)
    return {
        "installed": True,
        "updated": updated,
        "plugin": "turbofit",
        "path": str(target),
        "slash_commands": activation,
    }


def _default_hermes_home(hermes_home: Path) -> Path:
    if hermes_home.parent.name == "profiles":
        return hermes_home.parent.parent
    return hermes_home


def hermes_homes(hermes_home: Path | None = None) -> list[Path]:
    """Return the default Hermes home plus every named profile home."""
    current = Path(hermes_home or os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    default = _default_hermes_home(current)
    homes = [default]
    profiles = default / "profiles"
    if profiles.is_dir():
        homes.extend(sorted(path for path in profiles.iterdir() if path.is_dir()))
    if current not in homes:
        homes.append(current)
    return homes


def _enable_plugin(config_path: Path) -> bool:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    plugins = payload.get("plugins")
    if not isinstance(plugins, dict):
        plugins = {}
        payload["plugins"] = plugins
    enabled = plugins.get("enabled")
    if not isinstance(enabled, list):
        enabled = []
        plugins["enabled"] = enabled
    names = {str(item).strip().lower() for item in enabled}
    if "turbofit" in names or "user/turbofit" in names:
        return False
    enabled.append("turbofit")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return True


def _publish_path(source: Path, destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        return "exists"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
        return "symlink"
    except OSError:
        pass
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return "junction"
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return "copy"


def activate_slash_commands(
    *,
    hermes_home: Path | None = None,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    """Enable /turbofit in every Hermes home, including Sirvir.

    Desktop profile sessions only scan that profile's plugins/ and
    plugins.enabled. Installing into ~/.hermes alone leaves /turbofit
    unknown in Sirvir with 'not a quick/plugin/bundle/skill command'.
    """
    source = Path(plugin_root or PLUGIN_ROOT).resolve()
    skill = PLUGIN_ROOT / "skills" / "turbofit"
    homes = hermes_homes(hermes_home)
    published = []
    for home in homes:
        plugin_dest = home / "plugins" / "turbofit"
        skill_dest = home / "skills" / "turbofit"
        published.append({
            "home": str(home),
            "enabled": _enable_plugin(home / "config.yaml"),
            "plugin": _publish_path(source, plugin_dest) if plugin_dest.resolve() != source else "self",
            "skill": _publish_path(skill, skill_dest) if skill.is_dir() else "missing-skill",
        })
    return {"ok": True, "homes": len(homes), "published": published}


def launch_setup_screen() -> dict[str, Any]:
    """Refresh the Hermes Desktop Turbofit page. Dashboard is deprecated."""
    desktop = install_desktop_plugin()
    models = ensure_recommended_models()
    slash = activate_slash_commands()
    return {
        "launched": True,
        "surface": "desktop",
        "path": "/turbofit",
        "page": "Turbofit",
        "desktop": desktop,
        "models": models,
        "slash_commands": slash,
        "message": "Recommended models are downloading or verified. Open Hermes Desktop → Turbofit, or ask Sirvir to finish setup.",
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


def recommended_artifact_families(usable_memory_mb: int | None = None) -> list[str]:
    """Auto-chain families for this machine, plus default Ornith auxiliary."""
    if usable_memory_mb is None:
        from turbofit_runtime.hardware import probe_hardware

        usable_memory_mb = int(probe_hardware().total_usable_memory_mb)
    if usable_memory_mb < 16 * 1024:
        main = "maple-preview-tq2"
    elif usable_memory_mb < 24 * 1024:
        main = "qwen3-8-27b-unleashed-ud-iq3-xxs"
    elif usable_memory_mb < 96 * 1024:
        main = "qwen3-8-27b-unleashed-ud-q3-k-xl"
    else:
        main = "qwen3-8-27b-bf16"
    return [main, "ornith-1-5-35a3b"]


def ensure_recommended_models(
    *,
    families: list[str] | None = None,
    usable_memory_mb: int | None = None,
    download_fn=None,
) -> dict[str, Any]:
    """Download SHA-pinned recommended artifacts if they are missing."""
    from importlib.machinery import SourceFileLoader
    import importlib.util

    script = PLUGIN_ROOT / "scripts" / "download-artifacts"
    loader = SourceFileLoader("turbofit_download_artifacts", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"missing artifact downloader: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wanted = set(families or recommended_artifact_families(usable_memory_mb))
    payload = json.loads((PLUGIN_ROOT / "references" / "artifact-manifest.json").read_text(encoding="utf-8"))
    rows = module.selected_artifacts(payload, wanted)
    kwargs = {"root": module.model_root()}
    if download_fn is not None:
        kwargs["download_fn"] = download_fn
    results = [module.install_artifact(item, **kwargs) for item in rows]
    return {
        "ok": True,
        "families": sorted(wanted),
        "downloaded": sum(1 for item in results if item.get("downloaded")),
        "verified": sum(1 for item in results if item.get("verified")),
        "artifacts": results,
    }


def install_freetoken_runtime() -> dict[str, Any]:
    """Install the pinned FreeToken candidate only on its supported hardware contract."""
    script = PLUGIN_ROOT / "scripts" / "install-freetoken-runtime"
    if not script.is_file():
        raise FileNotFoundError(f"missing FreeToken installer: {script}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SRC_ROOT), environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, str(script), "--json"],
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
        env=environment,
    )
    try:
        payload = json.loads(result.stdout or result.stderr)
    except ValueError as exc:
        raise RuntimeError((result.stdout or result.stderr).strip() or "FreeToken install failed") from exc
    if result.returncode:
        blockers = payload.get("blockers") or []
        detail = "; ".join(str(item) for item in blockers) or payload.get("error") or "FreeToken install failed"
        raise RuntimeError(detail)
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
    return address.is_loopback or address.is_private or address in tailnet


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


def _model_entry(context_length: int | None = None) -> dict[str, Any]:
    if context_length is None:
        return {}
    parsed = int(context_length)
    if parsed <= 0:
        raise ValueError("context_length must be a positive integer")
    return {"context_length": parsed}


def provider_definition(
    base_url: str | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Return the canonical named custom-provider definition."""
    entry = _model_entry(context_length)
    return {
        "name": "TurboFit",
        "api": _validated_base_url(base_url),
        "api_key": "not-needed",
        "transport": "chat_completions",
        "default_model": "auto",
        "models": {
            "auto": dict(entry),
            "active:main": dict(entry),
            "active:aux": dict(entry),
        },
    }


def _is_turbofit_target(provider: Any, model: Any) -> bool:
    provider_name = str(provider or "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider_name in {"custom:turbofit", "turbofit"} or model_name.startswith("active:")


def _align_compression_context(
    config: dict[str, Any],
    *,
    primary: bool,
    context_length: int | None,
) -> None:
    """Keep auxiliary.compression on the same context window as main."""
    raw_aux = config.get("auxiliary")
    auxiliary: dict[str, Any] = copy.deepcopy(dict(raw_aux)) if isinstance(raw_aux, Mapping) else {}
    raw_compression = auxiliary.get("compression")
    compression: dict[str, Any] = (
        copy.deepcopy(dict(raw_compression)) if isinstance(raw_compression, Mapping) else {}
    )
    raw_model = config.get("model")
    model_cfg = raw_model if isinstance(raw_model, Mapping) else {}

    if primary:
        compression["provider"] = "custom:turbofit"
        compression["model"] = "active:aux"
        if context_length is not None:
            compression["context_length"] = int(context_length)
        auxiliary["compression"] = compression
        config["auxiliary"] = auxiliary
        return

    if not compression:
        return
    if not _is_turbofit_target(compression.get("provider"), compression.get("model")):
        if context_length is not None and compression.get("context_length") is None:
            compression["context_length"] = int(context_length)
            auxiliary["compression"] = compression
            config["auxiliary"] = auxiliary
        return

    main_provider = str(model_cfg.get("provider") or "").strip()
    main_model = str(model_cfg.get("default") or "").strip()
    if not main_provider or not main_model:
        return
    compression["provider"] = main_provider
    compression["model"] = main_model
    compression.pop("base_url", None)
    main_context = model_cfg.get("context_length")
    if main_context is not None:
        compression["context_length"] = int(main_context)
    else:
        compression.pop("context_length", None)
    auxiliary["compression"] = compression
    config["auxiliary"] = auxiliary


def configure_hermes(
    config: Mapping[str, Any],
    *,
    primary: bool = False,
    fallback: bool | None = None,
    fallback_chain: list[Mapping[str, Any]] | None = None,
    base_url: str | None = None,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Return a copied Hermes config with Turbofit registered idempotently.

    Main and aux always receive the same context_length. When Turbofit is
    primary, compression is pinned to active:aux at that same window. When
    it is not primary, a leftover Turbofit compression target is rebound
    to the configured main model so the two limits cannot drift.
    """
    updated = copy.deepcopy(dict(config))
    provider = provider_definition(base_url, context_length=context_length)

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
        if context_length is not None:
            model["context_length"] = int(context_length)
        updated["model"] = model

    _align_compression_context(updated, primary=primary, context_length=context_length)

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


def _live_usage() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    try:
        from turbofit_runtime.gpu import probe_gpus

        gpus = [sample.to_dict() for sample in probe_gpus()]
    except Exception:
        pass
    host: dict[str, Any] = {}
    try:
        from turbofit_runtime.hardware import probe_hardware

        hardware = probe_hardware()
        host = {
            "system_ram_mb": hardware.system_ram_mb,
            "host_usable_memory_mb": hardware.host_usable_memory_mb,
            "total_vram_mb": hardware.total_vram_mb,
            "topology_key": hardware.topology_key,
        }
    except Exception:
        pass
    runtime = _load_json(RUNTIME_STATE_PATH) or {}
    tps = None
    for key in ("main_tps", "tps", "predicted_per_second"):
        value = runtime.get(key)
        if isinstance(value, (int, float)) and value > 0:
            tps = float(value)
            break
    return {"gpus": gpus, "host": host, "tps": tps, "runtime_keys": sorted(runtime)}


def _provider_catalog(config: Mapping[str, Any]) -> dict[str, Any]:
    named = config.get("providers") if isinstance(config.get("providers"), Mapping) else {}
    subscriptions = []
    for name, spec in named.items():
        token = str(name).strip()
        if not token or token.lower() == "turbofit":
            continue
        body = spec if isinstance(spec, Mapping) else {}
        configured = any(body.get(key) for key in ("api_key", "api", "base_url", "oauth"))
        subscriptions.append({
            "id": token,
            "label": SUBSCRIPTION_LABELS.get(token.lower(), token.replace("_", " ").title()),
            "kind": "nous-subscription" if token.lower() == "nous" else "subscription",
            "configured": bool(configured),
        })
    nous_free = [
        {
            "provider": "nous",
            "model": item["model"],
            "label": NOUS_FREE_LABELS.get(item["model"], item["model"]),
            "kind": "nous-free",
        }
        for item in NOUS_FREE_FALLBACK_CHAIN
    ]
    return {"subscriptions": subscriptions, "nous_free": nous_free}


def _local_scale_ladder(selection: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    requested = ""
    if isinstance(selection, Mapping):
        requested = str(selection.get("profile") or selection.get("requested") or "")
    stem = requested.replace("hardware-", "") if requested.startswith("hardware-") else "8gb"
    path = PLUGIN_ROOT / "runtime-profiles" / f"{stem}.yaml"
    steps: list[dict[str, Any]] = []
    if path.is_file():
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for rung in payload.get("rungs") or []:
                if not isinstance(rung, Mapping):
                    continue
                ident = str(rung.get("id") or "")
                steps.append({
                    "id": ident,
                    "label": ident.replace("local-", "").replace("-", " "),
                    "context": rung.get("context"),
                    "aux_mode": rung.get("aux_mode"),
                    "kind": "local",
                })
        except Exception:
            pass
    if not steps:
        steps = [
            {"id": "local-maple-131072", "label": "maple 131072", "context": 131072, "aux_mode": "shared-main", "kind": "local"},
            {"id": "local-maple-65536", "label": "maple 65536", "context": 65536, "aux_mode": "shared-main", "kind": "local"},
        ]
    steps.append({"id": "nous-free", "label": "Nous keyless free", "context": None, "aux_mode": "api", "kind": "api"})
    return steps


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
        "usage": _live_usage() if probe else {"gpus": [], "host": {}, "tps": None},
        "catalog": _provider_catalog(config),
        "scale_ladder": _local_scale_ladder(selection),
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
        controller_available = os.name != "nt" and bool(shutil.which("systemctl"))
        active = False
        if controller_available:
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
        [
            sys.executable,
            str(script),
            "--json",
            "--limit",
            "10000",
            "--prefer",
            "balanced",
            "--evidence-scope",
            "portable-fit",
        ],
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
    no_match = result.returncode == 1 and rows == []
    if (result.returncode and not no_match) or not isinstance(rows, list):
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
    """Run TurboFit Check: scan physical hardware and return evidence-backed choices."""
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

    portable_preference = requested or "balanced"
    portable = subprocess.run(
        [
            sys.executable,
            str(script),
            "--json",
            "--fit-only",
            "--limit",
            str(limit),
            "--prefer",
            portable_preference,
            "--evidence-scope",
            "portable-fit",
        ],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
        env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
    )
    try:
        portable_payload = json.loads(portable.stdout)
    except ValueError:
        portable_payload = []
    exact_profiles = {
        str(item.get("profile"))
        for rows in recommendations.values()
        for item in rows
        if isinstance(item, Mapping)
    }
    compatible_lanes = [
        item
        for item in (portable_payload if isinstance(portable_payload, list) else [])
        if isinstance(item, Mapping) and str(item.get("profile")) not in exact_profiles
    ]
    if portable.returncode and not compatible_lanes:
        errors["portable_fit"] = (portable.stderr or "no portable local lane fits").strip()
    return {
        "ok": any(recommendations.values()) or bool(compatible_lanes),
        "process": "TurboFit Check",
        "hardware": {
            **asdict(hardware),
            "topology_key": hardware.topology_key,
            "total_vram_mb": hardware.total_vram_mb,
            "total_usable_memory_mb": hardware.total_usable_memory_mb,
        },
        "requested_preference": requested,
        "preferences": list(preferences),
        "recommendations": recommendations,
        "compatible_lanes": compatible_lanes,
        "compatible_lane_policy": "only Fit List mains for this topology; dense 27B is not offered on 8 GB VRAM",
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


def _save_configuration_to_all_homes(
    configured: Mapping[str, Any],
    *,
    base_url: str | None = None,
    primary: bool | None = None,
    fallback: bool | None = None,
    fallback_chain: list[Mapping[str, Any]] | None = None,
    multimodal: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Persist Turbofit provider registration to every Hermes home (default + profiles).

    Fresh macOS profiles (e.g. turbosovth, sirvir) keep their own
    config.yaml under ~/.hermes/profiles/<name>/.  Updating only the
    default ~/.hermes/config.yaml leaves those profiles with an
    unresolvable turbofit provider — the exact Unknown provider
    'turbofit' seen on fresh MacBooks.  This helper replays the same
    configure_hermes result into every home so the provider is resolvable
    from any session.
    """
    from hermes_cli.config import load_config as _load_config, save_config as _save_config

    canonical_base = base_url
    if canonical_base is None:
        try:
            turbo = configured.get("providers", {}).get("turbofit", {})  # type: ignore[union-attr]
            if isinstance(turbo, Mapping):
                canonical_base = str(turbo.get("api") or turbo.get("base_url") or "").strip() or None
        except Exception:
            canonical_base = None

    saved: list[dict[str, Any]] = []
    for home in hermes_homes():
        try:
            previous = os.environ.get("HERMES_HOME")
            os.environ["HERMES_HOME"] = str(home)
            try:
                home_config = _load_config()
                home_configured = configure_hermes(
                    home_config,
                    primary=bool(primary) if primary is not None else False,
                    fallback=fallback,
                    fallback_chain=fallback_chain,
                    base_url=canonical_base,
                )
                if multimodal is not None:
                    from turbofit_runtime.multimodal import MultimodalCatalog, configure_multimodal

                    home_configured = configure_multimodal(
                        home_configured,
                        selections=multimodal,
                        catalog=MultimodalCatalog.load(MULTIMODAL_CATALOG),
                    )
                _save_config(home_configured, merge_existing=False)
                saved.append({"home": str(home), "ok": True})
            finally:
                if previous is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = previous
        except Exception as exc:
            saved.append({"home": str(home), "ok": False, "error": str(exc)})
    return saved


def ensure_provider_registered() -> list[dict[str, Any]]:
    """Ensure ``providers.turbofit`` exists in every Hermes home.

    This is the fresh-install heal: a newly created profile (or a default
    install that never ran ``/turbofit setup``) has no turbofit entry at
    all, so ``custom:turbofit``/``turbofit`` is unresolvable and the agent
    fails with ``Unknown provider 'turbofit'``.  Healing writes a minimal
    provider registration (no primary switch) to every home that is missing
    it.  Safe to call on every ``register()`` and ``status_snapshot``.
    """
    healed: list[dict[str, Any]] = []
    for home in hermes_homes():
        try:
            previous = os.environ.get("HERMES_HOME")
            os.environ["HERMES_HOME"] = str(home)
            try:
                from hermes_cli.config import load_config as _lc, save_config as _sc
                cfg = _lc()
                providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
                if "turbofit" in providers:
                    healed.append({"home": str(home), "healed": False})
                    continue
                # Register without switching primary — just make it resolvable.
                healed_cfg = configure_hermes(cfg, primary=False, fallback=None, base_url=None)
                _sc(healed_cfg, merge_existing=False)
                healed.append({"home": str(home), "healed": True})
            finally:
                if previous is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = previous
        except Exception as exc:
            healed.append({"home": str(home), "healed": False, "error": str(exc)})
    return healed


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
    install_freetoken: bool = False,
    publish_tailnet_routes: bool = False,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
) -> dict[str, Any]:
    from hermes_cli.config import load_config

    with _CONFIG_LOCK:
        sirvir = install_sirvir_profile() if install_sirvir else None
        desktop = install_desktop_plugin() if install_desktop else None
        lemonade = install_lemonade_runtime() if install_lemonade else None
        native = install_native_runtime() if install_native else None
        freetoken = install_freetoken_runtime() if install_freetoken else None
        models = ensure_recommended_models()
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
        canonical = configure_hermes(
            load_config(),
            primary=primary,
            fallback=fallback,
            fallback_chain=list(NOUS_FREE_FALLBACK_CHAIN) if fallback_chain is None else fallback_chain,
            base_url=effective_base_url,
        )
        if multimodal is not None:
            from turbofit_runtime.multimodal import MultimodalCatalog, configure_multimodal

            canonical = configure_multimodal(
                canonical,
                selections=multimodal,
                catalog=MultimodalCatalog.load(MULTIMODAL_CATALOG),
            )
        homes_saved = _save_configuration_to_all_homes(
            canonical,
            base_url=effective_base_url,
            primary=primary,
            fallback=fallback,
            fallback_chain=list(NOUS_FREE_FALLBACK_CHAIN) if fallback_chain is None else fallback_chain,
            multimodal=multimodal,
        )
    return {
        "ok": True,
        "configured": True,
        "homes": homes_saved,
        "selection": selected,
        "tailnet": publication,
        "sirvir": sirvir,
        "desktop_plugin": desktop,
        "lemonade": lemonade,
        "native_runtime": native,
        "freetoken_runtime": freetoken,
        "models": models,
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
        install_freetoken = args.get("install_freetoken", False)
        if (
            not isinstance(primary, bool)
            or (fallback is not None and not isinstance(fallback, bool))
            or not isinstance(publish_routes, bool)
            or not isinstance(install_sirvir, bool)
            or not isinstance(install_desktop, bool)
            or not isinstance(install_lemonade, bool)
            or not isinstance(install_native, bool)
            or not isinstance(install_freetoken, bool)
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
            install_freetoken=install_freetoken,
            **ports,
        )
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})
