#!/usr/bin/env python3
"""Scan Hermes for available free models and write a validated fallback list.

Reads the Hermes endpoint metadata cache, cross-references against known-free
model IDs, and writes a Turbofit-consumable fallback manifest.
Safe to run repeatedly — stale entries are dropped, new ones are appended.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

HOME = os.path.expanduser("~")
CATALOG_PATH = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
FALLBACK_PATH = os.environ.get(
    "TURBOFIT_FALLBACK_MANIFEST",
    f"{HOME}/.config/turbofit/fallback-models.yaml",
)
METADATA_CACHE = f"{HOME}/.hermes/cache/endpoint_model_metadata.json"
PROVIDER_MODELS_CACHE = f"{HOME}/.hermes/provider_models_cache.json"
FREE_RE = re.compile(r"free", re.IGNORECASE)

# Hard exclusion list — models known to be dead or unsuitable for aux fallback
BLOCKLIST = {
    "stealth/ox-alpha",  # retired
    "tencent/hy3:free",  # 404
}


def _load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_yaml_safe(path):
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _provider_name_from_endpoint(endpoint: str) -> str:
    """Derive a short provider name from an endpoint URL."""
    url = endpoint.lower().replace("https://", "").replace("http://", "")
    host = url.split("/")[0]
    # strip port
    host = host.split(":")[0]
    # strip tailscale domains
    host = host.split(".")[0]
    return host or "remote"


def discover_free_models(metadata_cache):
    """Extract free models from every Hermes endpoint metadata cache entry."""
    free = {}
    for endpoint, meta in metadata_cache.items():
        if not isinstance(meta, dict):
            continue
        provider = _provider_name_from_endpoint(endpoint)
        for model_id, model_meta in (meta.get("models") or {}).items():
            if not isinstance(model_id, str):
                continue
            if model_id in BLOCKLIST:
                continue
            if not FREE_RE.search(model_id):
                continue
            free[model_id] = {
                "context_length": (model_meta or {}).get("context_length", 0) or 0,
                "description": ((model_meta or {}).get("description") or "").strip()[:120],
                "provider": provider,
                "source": "endpoint-scan",
            }
    return free


def discover_free_models_from_provider_cache(provider_cache):
    """Extract free models from the Hermes provider models cache."""
    free = {}
    for provider, entry in provider_cache.items():
        if not isinstance(entry, dict):
            continue
        provider_name = str(provider).split(":")[0]  # strip custom: prefix / URL fragments
        for model_id in entry.get("models", []) or []:
            if not isinstance(model_id, str):
                continue
            if model_id in BLOCKLIST:
                continue
            if not FREE_RE.search(model_id):
                continue
            free[model_id] = {
                "context_length": 0,
                "description": "",
                "provider": provider_name,
                "source": "provider-cache",
            }
    return free


def main():
    metadata = _load_json_safe(METADATA_CACHE)
    provider_cache = _load_json_safe(PROVIDER_MODELS_CACHE)

    free_models = discover_free_models(metadata)
    free_models.update(discover_free_models_from_provider_cache(provider_cache))
    if not free_models:
        print("FAIL: No free models found in Hermes caches", file=sys.stderr)
        sys.exit(1)

    # Build manifest
    manifest = {
        "schema": "turbofit.fallback-manifest/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {
            mid: {
                "context_length": info["context_length"],
                "description": info["description"],
                "role": "aux",  # free models are fallback aux, never main
                "source": info["source"],
                "provider": info["provider"],
            }
            for mid, info in sorted(free_models.items())
        },
    }

    # Write
    Path(FALLBACK_PATH).parent.mkdir(parents=True, exist_ok=True)
    import yaml

    with open(FALLBACK_PATH, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    # Print summary
    count = len(manifest["models"])
    print(f"OK: {count} free models → {FALLBACK_PATH}")
    for mid in list(manifest["models"].keys())[:5]:
        ctx = manifest["models"][mid]["context_length"]
        prov = manifest["models"][mid]["provider"]
        print(f"  {mid} (ctx={ctx}, provider={prov})")
    if count > 5:
        print(f"  ... and {count - 5} more")

    return manifest


if __name__ == "__main__":
    main()
