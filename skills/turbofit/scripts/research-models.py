#!/usr/bin/env python3
"""
Turbofit Model Database — Daily Research Script v2

Fetches REAL data from the OpenRouter API (/api/v1/models) for all watched model
families. Extracts pricing (including cache rates), context windows, vision
capability, and computes monthly cost projections for typical Hermes Agent usage.

No speculation — only data that exists in the API response.

Usage:
    python3 research-models.py [--dry-run] [--json]

Output:
    - references/research-report.md  (human-readable report)
    - references/model-pricing.json   (machine-readable, for serve cost)
"""

import json
import sys
import os
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(SCRIPT_DIR, "..", "references")
REPORT_PATH = os.path.join(REF_DIR, "research-report.md")
JSON_PATH = os.path.join(REF_DIR, "model-pricing.json")
DB_PATH = os.path.join(REF_DIR, "model-database.yaml")

OPENROUTER_API = "https://openrouter.ai/api/v1/models"

# Models we actively track (by OpenRouter slug substring)
# These are the model families relevant to Hermes Agent users
WATCH_SLUGS = [
    "glm-5.2", "glm-5.1",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash",
    "qwen3.5-flash",
    "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v3.2",
    "minimax-m3",
    "kimi-k2.7", "kimi-k2.6",
    "mimo-v2.5-pro", "mimo-v2.5",
    "nemotron",
    "agentworld",
]

# Monthly usage profiles for cost projection
# Based on Hermes Agent aux offset: 40-85% of tokens go to aux
USAGE_PROFILES = {
    "light": {
        "name": "Light user (hobbyist)",
        "sessions_per_day": 3,
        "tokens_per_session_input": 15000,
        "tokens_per_session_output": 3000,
        "cache_hit_rate": 0.30,
        "aux_offset": 0.50,
    },
    "moderate": {
        "name": "Moderate user (daily agent work)",
        "sessions_per_day": 10,
        "tokens_per_session_input": 30000,
        "tokens_per_session_output": 8000,
        "cache_hit_rate": 0.40,
        "aux_offset": 0.60,
    },
    "heavy": {
        "name": "Heavy user (multi-agent, long sessions)",
        "sessions_per_day": 25,
        "tokens_per_session_input": 60000,
        "tokens_per_session_output": 15000,
        "cache_hit_rate": 0.50,
        "aux_offset": 0.70,
    },
    "extreme": {
        "name": "Extreme (24/7 agent fleet)",
        "sessions_per_day": 60,
        "tokens_per_session_input": 100000,
        "tokens_per_session_output": 25000,
        "cache_hit_rate": 0.55,
        "aux_offset": 0.75,
    },
}

def fetch_openrouter_models():
    """Fetch the full OpenRouter model list."""
    req = Request(OPENROUTER_API, headers={
        "User-Agent": "turbofit-research/2.0",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, Exception) as e:
        print(f"ERROR fetching OpenRouter: {e}", file=sys.stderr)
        return None

def extract_watched_models(api_data):
    """Extract watched models from OpenRouter API response."""
    models = []
    for m in api_data.get("data", []):
        slug = m.get("id", "")
        slug_lower = slug.lower()

        matched = any(w in slug_lower for w in WATCH_SLUGS)
        if not matched:
            continue

        # Skip variants we don't want (instruct, uncensored, etc. unless they're the main one)
        if ":free" in slug_lower:
            slug_base = slug_lower.replace(":free", "")
            # Keep free variants but mark them
        else:
            slug_base = slug_lower

        pricing = m.get("pricing", {})
        arch = m.get("architecture", {})

        # Extract pricing (per-token, convert to per-M)
        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000
        cache_read_price = float(pricing.get("input_cache_read", "0") or "0") * 1_000_000
        cache_write_price = float(pricing.get("input_cache_write", "0") or "0") * 1_000_000

        # Determine vision from architecture
        input_modalities = arch.get("input_modalities", [])
        has_vision = "image" in input_modalities or "video" in input_modalities

        # Context length
        ctx = m.get("context_length", 0) or arch.get("context_length", 0)

        # Top provider context (sometimes different from model-level)
        top_provider = m.get("top_provider", {})
        provider_ctx = top_provider.get("context_length", ctx)

        is_free = prompt_price == 0 and completion_price == 0

        models.append({
            "slug": slug,
            "input_per_m": round(prompt_price, 6),
            "output_per_m": round(completion_price, 6),
            "cache_read_per_m": round(cache_read_price, 6),
            "cache_write_per_m": round(cache_write_price, 6),
            "has_cache_read": cache_read_price > 0,
            "has_cache_write": cache_write_price > 0,
            "context_length": ctx,
            "provider_context_length": provider_ctx,
            "has_vision": has_vision,
            "input_modalities": input_modalities,
            "modality": arch.get("modality", ""),
            "is_free": is_free,
            "max_completion_tokens": top_provider.get("max_completion_tokens", 0),
        })

    return sorted(models, key=lambda x: x["slug"])

def compute_monthly_cost(model, profile):
    """
    Compute projected monthly cost for a model given a usage profile.

    Cost = (input_tokens * cache_hit_rate * cache_read_price
          + input_tokens * (1 - cache_hit_rate) * input_price
          + output_tokens * output_price) * sessions_per_day * 30
    """
    p = profile
    daily_input = p["sessions_per_day"] * p["tokens_per_session_input"]
    daily_output = p["sessions_per_day"] * p["tokens_per_session_output"]

    # Split input into cache hit vs miss
    cache_hit_input = daily_input * p["cache_hit_rate"]
    cache_miss_input = daily_input * (1 - p["cache_hit_rate"])

    # Daily cost in dollars (prices are per-M tokens, so divide by 1M)
    if model["has_cache_read"] and model["cache_read_per_m"] > 0:
        daily_cost = (
            (cache_hit_input / 1_000_000) * model["cache_read_per_m"]
            + (cache_miss_input / 1_000_000) * model["input_per_m"]
            + (daily_output / 1_000_000) * model["output_per_m"]
        )
    else:
        # No cache pricing — all input at full price
        daily_cost = (
            (daily_input / 1_000_000) * model["input_per_m"]
            + (daily_output / 1_000_000) * model["output_per_m"]
        )

    monthly_cost = daily_cost * 30
    return round(monthly_cost, 2)

def compute_pairing_cost(main_model, aux_model, profile):
    """
    Compute projected monthly cost for a main+aux pairing.

    Aux offset: profile["aux_offset"] fraction of tokens go to aux.
    Main handles: (1 - aux_offset) of input + output
    Aux handles: aux_offset of input + output
    """
    p = profile
    daily_input = p["sessions_per_day"] * p["tokens_per_session_input"]
    daily_output = p["sessions_per_day"] * p["tokens_per_session_output"]
    cache_rate = p["cache_hit_rate"]

    def model_daily_cost(model, input_tokens, output_tokens):
        cache_hit = input_tokens * cache_rate
        cache_miss = input_tokens * (1 - cache_rate)
        if model["has_cache_read"] and model["cache_read_per_m"] > 0:
            return (
                (cache_hit / 1_000_000) * model["cache_read_per_m"]
                + (cache_miss / 1_000_000) * model["input_per_m"]
                + (output_tokens / 1_000_000) * model["output_per_m"]
            )
        else:
            return (
                (input_tokens / 1_000_000) * model["input_per_m"]
                + (output_tokens / 1_000_000) * model["output_per_m"]
            )

    aux_offset = p["aux_offset"]
    main_input = daily_input * (1 - aux_offset)
    main_output = daily_output * (1 - aux_offset)
    aux_input = daily_input * aux_offset
    aux_output = daily_output * aux_offset

    main_daily = model_daily_cost(main_model, main_input, main_output)
    aux_daily = model_daily_cost(aux_model, aux_input, aux_output)

    monthly = (main_daily + aux_daily) * 30
    return round(monthly, 2)

def generate_report(models, profiles):
    """Generate markdown report with pricing, cache rates, and monthly projections."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# Turbofit Model Research Report — {today}",
        "",
        "## Summary",
        f"- Models tracked: {len(models)}",
        f"- Data source: OpenRouter API (`/api/v1/models`)",
        f"- Free models: {len([m for m in models if m['is_free']])}",
        f"- Models with cache pricing: {len([m for m in models if m['has_cache_read']])}",
        f"- Models with vision: {len([m for m in models if m['has_vision']])}",
        "",
        "## Model Pricing (live from OpenRouter API)",
        "",
        "| Slug | Vision | Input $/M | Cache Read $/M | Output $/M | Context | Free |",
        "|------|--------|----------|---------------|-----------|---------|------|",
    ]

    for m in models:
        vis = "YES" if m["has_vision"] else "no"
        cache = f"${m['cache_read_per_m']:.4f}" if m["has_cache_read"] else "-"
        free = "FREE" if m["is_free"] else ""
        ctx_k = m["context_length"] // 1000 if m["context_length"] else 0
        lines.append(
            f"| {m['slug']} | {vis} | ${m['input_per_m']:.4f} | {cache} | ${m['output_per_m']:.4f} | {ctx_k}K | {free} |"
        )

    lines.append("")

    # Monthly cost projections per model
    lines.append("## Monthly Cost Projections (single model, no aux)")
    lines.append("")
    lines.append("Assumes all tokens go to this model (no aux offset). Real cost will be lower if using a cheaper aux.")
    lines.append("")
    header = "| Slug |"
    sep = "|------|"
    for pkey, pval in profiles.items():
        header += f" {pval['name']} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for m in models:
        row = f"| {m['slug']} |"
        for pkey, pval in profiles.items():
            cost = compute_monthly_cost(m, pval)
            if cost == 0:
                row += " FREE |"
            else:
                row += f" ${cost:.2f} |"
        lines.append(row)

    lines.append("")

    # Pairing cost projections for recommended pairings
    lines.append("## Pairing Cost Projections (main + aux)")
    lines.append("")
    lines.append("Based on Hermes Agent aux offset (40-85% of tokens route to aux).")
    lines.append("")

    # Define recommended pairings from the pairing matrix
    pairings = []
    # Find models by slug
    def find(slug_part):
        for m in models:
            if slug_part in m["slug"]:
                return m
        return None

    mains = {
        "GLM 5.2": find("glm-5.2"),
        "Qwen 3.7 MAX": find("qwen3.7-max"),
        "DeepSeek V4 Pro": find("deepseek-v4-pro"),
        "DeepSeek V4 Flash": find("deepseek-v4-flash"),
        "Mimo V2.5 Pro": find("mimo-v2.5-pro"),
    }
    auxs = {
        "Qwen 3.5 Flash": find("qwen3.5-flash"),
        "Mimo V2.5": find("mimo-v2.5"),
        "MiniMax M3": find("minimax-m3"),
        "Kimi K2.7 Code": find("kimi-k2.7"),
    }

    for main_name, main_model in mains.items():
        if not main_model:
            continue
        for aux_name, aux_model in auxs.items():
            if not aux_model:
                continue
            # Only show text-only mains with vision auxs, or vision mains with any aux
            if not main_model["has_vision"] and not aux_model["has_vision"]:
                continue  # Skip text-only + text-only (no vision in the pair)
            pairings.append((main_name, aux_name, main_model, aux_model))

    if pairings:
        header = "| Pairing |"
        sep = "|---------|"
        for pkey, pval in profiles.items():
            header += f" {pval['name']} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)

        for main_name, aux_name, main_model, aux_model in pairings:
            pair_label = f"{main_name} + {aux_name}"
            row = f"| {pair_label} |"
            for pkey, pval in profiles.items():
                cost = compute_pairing_cost(main_model, aux_model, pval)
                if cost == 0:
                    row += " FREE |"
                else:
                    row += f" ${cost:.2f} |"
            lines.append(row)

        lines.append("")

    # Free models highlight
    free_models = [m for m in models if m["is_free"]]
    if free_models:
        lines.append("## Free Models (zero cost)")
        lines.append("")
        for m in free_models:
            ctx_k = m["context_length"] // 1000 if m["context_length"] else 0
            vis = "vision" if m["has_vision"] else "text-only"
            lines.append(f"- **{m['slug']}** — {ctx_k}K ctx, {vis}")
        lines.append("")

    # Cache analysis
    lines.append("## Cache Pricing Analysis")
    lines.append("")
    lines.append("Models with cache read pricing offer significant savings for repeated context (common in Hermes Agent long sessions):")
    lines.append("")
    for m in models:
        if m["has_cache_read"] and m["cache_read_per_m"] < m["input_per_m"]:
            savings = ((m["input_per_m"] - m["cache_read_per_m"]) / m["input_per_m"] * 100) if m["input_per_m"] > 0 else 0
            lines.append(f"- **{m['slug']}**: cache read ${m['cache_read_per_m']:.4f}/M vs input ${m['input_per_m']:.4f}/M — {savings:.0f}% savings on cache hits")

    lines.append("")
    lines.append("## Action Items")
    lines.append("")
    lines.append("1. Compare prices above with `model-database.yaml` — update any that changed")
    lines.append("2. Add any NEW models not yet in the database")
    lines.append("3. Check if any free models changed status")
    lines.append("4. Verify cache pricing is reflected in the pairing matrix")
    lines.append("5. Run `bash scripts/sync-github.sh` after updates")
    lines.append("")

    return "\n".join(lines)

def main():
    dry_run = "--dry-run" in sys.argv
    output_json = "--json" in sys.argv

    print(f"Turbofit Research v2 — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    print("\n1. Fetching OpenRouter API...")
    api_data = fetch_openrouter_models()
    if not api_data:
        print("   FAILED — cannot reach OpenRouter API")
        sys.exit(1)
    print(f"   Got {len(api_data.get('data', []))} total models from API")

    print("\n2. Extracting watched models...")
    models = extract_watched_models(api_data)
    print(f"   Found {len(models)} watched models")

    print("\n3. Computing cost projections...")
    # Verify cost computation works
    if models:
        test_cost = compute_monthly_cost(models[0], list(USAGE_PROFILES.values())[1])
        print(f"   Test: {models[0]['slug']} moderate profile = ${test_cost:.2f}/mo")

    print("\n4. Generating report...")
    report = generate_report(models, USAGE_PROFILES)

    # Write JSON for machine-readable use (serve cost command)
    json_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "openrouter",
        "models": models,
        "usage_profiles": USAGE_PROFILES,
    }

    if dry_run:
        print("\n--- DRY RUN — Report preview ---\n")
        print(report[:4000])
        if len(report) > 4000:
            print(f"\n...({len(report) - 4000} more chars)")
    else:
        os.makedirs(REF_DIR, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            f.write(report)
        print(f"   Report: {REPORT_PATH}")

        with open(JSON_PATH, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"   JSON:   {JSON_PATH}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(models)} models, {len([m for m in models if m['has_cache_read']])} with cache pricing, {len([m for m in models if m['is_free']])} free")
    print("STATUS: OK")

if __name__ == "__main__":
    main()
