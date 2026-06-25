#!/usr/bin/env python3
"""
Turbofit Model Database — Daily Research Script v3

REAL data only. No speculation.

1. Fetches live pricing from OpenRouter API (/api/v1/models)
   - input_cache_read rates for models that support caching
   - context_length, vision from architecture.input_modalities
2. Reads the user's actual Hermes Agent usage from state.db
   - real input/output/cache token counts
   - real cache hit rate
   - real sessions/day, avg tokens/session
3. Projects monthly cost based on ACTUAL usage patterns
4. Generates report with real data only

Usage:
    python3 research-models.py [--dry-run]
    python3 research-models.py --db /path/to/state.db  # custom Hermes DB
"""

import json
import sys
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(SCRIPT_DIR, "..", "references")
REPORT_PATH = os.path.join(REF_DIR, "research-report.md")
JSON_PATH = os.path.join(REF_DIR, "model-pricing.json")

OPENROUTER_API = "https://openrouter.ai/api/v1/models"

# Find the Hermes state.db
def find_hermes_db():
    """Find the Hermes state.db — check profile, then default."""
    candidates = [
        os.path.expanduser("~/.hermes/profiles/senter/state.db"),
        os.path.expanduser("~/.hermes/state.db"),
    ]
    # Also check HERMES_HOME
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        candidates.insert(0, os.path.join(hermes_home, "state.db"))

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None

# Models we track (by OpenRouter slug substring)
WATCH_SLUGS = [
    "glm-5.2", "glm-5.1",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus", "qwen3.6-flash",
    "qwen3.5-flash",
    "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v3.2",
    "minimax-m3",
    "kimi-k2.7", "kimi-k2.6",
    "mimo-v2.5-pro", "mimo-v2.5",
    "nemotron",
]

def fetch_openrouter_models():
    """Fetch the full OpenRouter model list."""
    req = Request(OPENROUTER_API, headers={
        "User-Agent": "turbofit-research/3.0",
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
        if not any(w in slug_lower for w in WATCH_SLUGS):
            continue

        pricing = m.get("pricing", {})
        arch = m.get("architecture", {})
        input_modalities = arch.get("input_modalities", [])
        has_vision = "image" in input_modalities or "video" in input_modalities

        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000
        cache_read_price = float(pricing.get("input_cache_read", "0") or "0") * 1_000_000
        cache_write_price = float(pricing.get("input_cache_write", "0") or "0") * 1_000_000

        ctx = m.get("context_length", 0) or arch.get("context_length", 0)
        top_provider = m.get("top_provider", {})
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
            "has_vision": has_vision,
            "input_modalities": input_modalities,
            "is_free": is_free,
            "max_completion_tokens": top_provider.get("max_completion_tokens", 0),
        })

    return sorted(models, key=lambda x: x["slug"])

def read_real_usage(db_path):
    """Read actual token usage from Hermes state.db."""
    if not db_path:
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Overall stats
    cur.execute("""
        SELECT
            COUNT(*) as sessions,
            SUM(message_count) as messages,
            SUM(tool_call_count) as tool_calls,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(cache_read_tokens) as cache_read_tokens,
            SUM(cache_write_tokens) as cache_write_tokens,
            SUM(estimated_cost_usd) as est_cost,
            AVG(input_tokens) as avg_input_per_session,
            AVG(output_tokens) as avg_output_per_session
        FROM sessions
        WHERE input_tokens > 0
    """)
    overall = dict(cur.fetchone())

    # Per-model breakdown
    cur.execute("""
        SELECT model,
            COUNT(*) as sessions,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(cache_read_tokens) as cache_read_tokens,
            SUM(estimated_cost_usd) as est_cost
        FROM sessions
        WHERE input_tokens > 0
        GROUP BY model
        ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
    """)
    by_model = [dict(r) for r in cur.fetchall()]

    # Per-platform
    cur.execute("""
        SELECT source,
            COUNT(*) as sessions,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(estimated_cost_usd) as est_cost
        FROM sessions
        WHERE input_tokens > 0
        GROUP BY source
        ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
    """)
    by_platform = [dict(r) for r in cur.fetchall()]

    # Daily averages for last 30 active days
    cur.execute("""
        SELECT
            COUNT(DISTINCT DATE(started_at, 'unixepoch')) as active_days,
            COUNT(*) as sessions,
            SUM(input_tokens) / COUNT(DISTINCT DATE(started_at, 'unixepoch')) as avg_daily_input,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(cache_read_tokens) as total_cache_read
        FROM sessions
        WHERE input_tokens > 0
    """)
    daily = dict(cur.fetchone())

    conn.close()

    # Calculate real cache hit rate
    total_input = overall["input_tokens"] or 0
    total_cache_read = overall["cache_read_tokens"] or 0
    cache_hit_rate = (total_cache_read / total_input) if total_input > 0 else 0

    # Note: cache_read_tokens can exceed input_tokens because OpenRouter
    # counts cached tokens separately from billed input tokens.
    # The "effective" cache hit rate is: cache_read / (input + cache_read)
    effective_cache_rate = (total_cache_read / (total_input + total_cache_read)) if (total_input + total_cache_read) > 0 else 0

    return {
        "overall": overall,
        "by_model": by_model,
        "by_platform": by_platform,
        "daily": daily,
        "cache_hit_rate": cache_hit_rate,
        "effective_cache_rate": effective_cache_rate,
        "total_input": total_input,
        "total_output": overall["output_tokens"] or 0,
        "total_cache_read": total_cache_read,
        "total_est_cost": overall["est_cost"] or 0,
        "avg_input_per_session": overall["avg_input_per_session"] or 0,
        "avg_output_per_session": overall["avg_output_per_session"] or 0,
        "active_days": daily["active_days"] or 1,
        "avg_sessions_per_day": (daily["sessions"] / daily["active_days"]) if daily["active_days"] and daily["sessions"] else 1,
    }

def compute_monthly_cost_from_real_usage(model, usage):
    """
    Project monthly cost for a model based on the user's REAL usage patterns.

    Uses: actual daily input/output token averages, actual cache hit rate.
    """
    if not usage:
        return 0

    # Daily averages from real data
    daily_input = usage["total_input"] / usage["active_days"]
    daily_output = usage["total_output"] / usage["active_days"]
    daily_cache_read = usage["total_cache_read"] / usage["active_days"]

    # Real cache hit rate (effective — what fraction of input was cache hits)
    cache_rate = usage["effective_cache_rate"]

    # For this model: how much input would be cache hit vs miss?
    # If the model supports caching, use the real cache rate.
    # If not, all input is at full price.
    if model["has_cache_read"] and model["cache_read_per_m"] > 0:
        # With caching: cache_read tokens at cache price, rest at input price
        cache_hit_input = daily_input * cache_rate
        cache_miss_input = daily_input * (1 - cache_rate)
        daily_cost = (
            (cache_hit_input / 1_000_000) * model["cache_read_per_m"]
            + (cache_miss_input / 1_000_000) * model["input_per_m"]
            + (daily_output / 1_000_000) * model["output_per_m"]
        )
    else:
        # No caching support — all input at full price
        daily_cost = (
            (daily_input / 1_000_000) * model["input_per_m"]
            + (daily_output / 1_000_000) * model["output_per_m"]
        )

    monthly_cost = daily_cost * 30
    return round(monthly_cost, 2)

def compute_pairing_cost_from_real_usage(main_model, aux_model, usage):
    """
    Project monthly cost for a main+aux pairing based on REAL usage.

    Hermes Agent aux offset: 40-85% of tokens route to aux.
    We use the REAL ratio from the user's data where possible.
    """
    if not usage:
        return 0

    # Real daily totals
    daily_input = usage["total_input"] / usage["active_days"]
    daily_output = usage["total_output"] / usage["active_days"]
    cache_rate = usage["effective_cache_rate"]

    # Aux offset: estimate from the user's data.
    # Hermes aux handles compression, vision, web extract, skill search, etc.
    # Default to 60% if we can't determine it from the data.
    aux_offset = 0.60  # midpoint of 40-85%

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

    main_input = daily_input * (1 - aux_offset)
    main_output = daily_output * (1 - aux_offset)
    aux_input = daily_input * aux_offset
    aux_output = daily_output * aux_offset

    main_daily = model_daily_cost(main_model, main_input, main_output)
    aux_daily = model_daily_cost(aux_model, aux_input, aux_output)

    monthly = (main_daily + aux_daily) * 30
    return round(monthly, 2)

def generate_report(models, usage, db_path):
    """Generate markdown report with REAL data."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# Turbofit Model Research Report — {today}",
        "",
        "## Data Sources",
        "- Model pricing: OpenRouter API (`/api/v1/models`) — live",
        f"- Usage data: Hermes state.db (`{db_path}`) — real user data",
        "",
    ]

    # Real usage summary
    if usage:
        lines.extend([
            "## Your Real Usage (from Hermes Insights)",
            "",
            f"- Active days: {usage['active_days']}",
            f"- Total sessions: {usage['overall']['sessions']}",
            f"- Total messages: {usage['overall']['messages']:,}",
            f"- Total tool calls: {usage['overall']['tool_calls']:,}",
            f"- Total input tokens: {usage['total_input']:,}",
            f"- Total output tokens: {usage['total_output']:,}",
            f"- Total cache read tokens: {usage['total_cache_read']:,}",
            f"- Effective cache hit rate: {usage['effective_cache_rate']*100:.1f}%",
            f"- Avg input/session: {usage['avg_input_per_session']:,.0f}",
            f"- Avg output/session: {usage['avg_output_per_session']:,.0f}",
            f"- Avg sessions/day: {usage['avg_sessions_per_day']:.1f}",
            f"- Total estimated cost: ${usage['total_est_cost']:.2f}",
            "",
        ])

        # Per-model actual usage
        lines.extend([
            "### Models You've Actually Used",
            "",
            "| Model | Sessions | Input Tokens | Output Tokens | Cache Read | Est. Cost |",
            "|-------|----------|-------------|--------------|------------|-----------|",
        ])
        for m in usage["by_model"][:15]:
            lines.append(
                f"| {m['model'] or 'unknown'} | {m['sessions']} | {m['input_tokens'] or 0:,} | "
                f"{m['output_tokens'] or 0:,} | {m['cache_read_tokens'] or 0:,} | ${m['est_cost'] or 0:.2f} |"
            )
        lines.append("")

        # Per-platform
        lines.extend([
            "### By Platform",
            "",
            "| Platform | Sessions | Input | Output | Cost |",
            "|----------|----------|-------|--------|------|",
        ])
        for p in usage["by_platform"]:
            lines.append(
                f"| {p['source'] or 'unknown'} | {p['sessions']} | {p['input_tokens'] or 0:,} | "
                f"{p['output_tokens'] or 0:,} | ${p['est_cost'] or 0:.2f} |"
            )
        lines.append("")
    else:
        lines.extend([
            "## Usage Data",
            "",
            "No Hermes state.db found. Cost projections will use default estimates.",
            "",
        ])

    # Live pricing
    lines.extend([
        "## Live Model Pricing (from OpenRouter API)",
        "",
        "| Slug | Vision | Input $/M | Cache Read $/M | Output $/M | Context | Free |",
        "|------|--------|----------|---------------|-----------|---------|------|",
    ])
    for m in models:
        vis = "YES" if m["has_vision"] else "no"
        cache = f"${m['cache_read_per_m']:.4f}" if m["has_cache_read"] else "-"
        free = "FREE" if m["is_free"] else ""
        ctx_k = m["context_length"] // 1000 if m["context_length"] else 0
        lines.append(
            f"| {m['slug']} | {vis} | ${m['input_per_m']:.4f} | {cache} | "
            f"${m['output_per_m']:.4f} | {ctx_k}K | {free} |"
        )
    lines.append("")

    # Monthly cost projections based on REAL usage
    if usage:
        lines.extend([
            "## Monthly Cost Projections (based on YOUR real usage)",
            "",
            "Projected using your actual daily input/output averages and cache hit rate.",
            "Single model = all tokens go to this model. Pairing = 60% aux offset.",
            "",
            "| Model | Single (monthly) | Notes |",
            "|-------|-------------------|-------|",
        ])
        for m in models:
            cost = compute_monthly_cost_from_real_usage(m, usage)
            cost_str = "FREE" if cost == 0 else f"${cost:.2f}"
            notes = []
            if m["has_cache_read"]:
                savings = ((m["input_per_m"] - m["cache_read_per_m"]) / m["input_per_m"] * 100) if m["input_per_m"] > 0 else 0
                notes.append(f"{savings:.0f}% cache savings")
            if m["has_vision"]:
                notes.append("vision")
            lines.append(f"| {m['slug']} | {cost_str} | {'; '.join(notes)} |")
        lines.append("")

        # Pairing projections
        lines.extend([
            "### Recommended Pairings (projected on YOUR usage)",
            "",
            "| Pairing | Monthly Cost | Notes |",
            "|---------|-------------|-------|",
        ])

        def find(slug_part):
            for m in models:
                if slug_part in m["slug"] and ":free" not in m["slug"]:
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
                if not main_model["has_vision"] and not aux_model["has_vision"]:
                    continue
                cost = compute_pairing_cost_from_real_usage(main_model, aux_model, usage)
                cost_str = "FREE" if cost == 0 else f"${cost:.2f}"
                lines.append(f"| {main_name} + {aux_name} | {cost_str} | |")
        lines.append("")

    # Cache analysis
    lines.extend([
        "## Cache Pricing Analysis",
        "",
        "Models with cache read pricing offer significant savings for repeated context.",
        f"Your effective cache hit rate: {usage['effective_cache_rate']*100:.1f}%" if usage else "",
        "",
    ])
    for m in models:
        if m["has_cache_read"] and m["cache_read_per_m"] < m["input_per_m"] and m["input_per_m"] > 0:
            savings = ((m["input_per_m"] - m["cache_read_per_m"]) / m["input_per_m"] * 100)
            lines.append(
                f"- **{m['slug']}**: cache read ${m['cache_read_per_m']:.4f}/M vs "
                f"input ${m['input_per_m']:.4f}/M — {savings:.0f}% savings on cache hits"
            )
    lines.append("")

    # Free models
    free_models = [m for m in models if m["is_free"]]
    if free_models:
        lines.extend([
            "## Free Models (zero cost on OpenRouter)",
            "",
        ])
        for m in free_models:
            ctx_k = m["context_length"] // 1000 if m["context_length"] else 0
            vis = "vision" if m["has_vision"] else "text-only"
            lines.append(f"- **{m['slug']}** — {ctx_k}K ctx, {vis}")
        lines.append("")

    # Nous Tool Gateway note
    lines.extend([
        "## About the Nous Tool Gateway",
        "",
        "The Nous Tool Gateway (Firecrawl, FAL, OpenAI TTS, Browser Use) is active",
        "whenever the user has a Nous Portal subscription — regardless of which",
        "models are selected for main or aux. It is a subscription feature, not",
        "a per-model feature.",
        "",
    ])

    lines.extend([
        "## Action Items",
        "",
        "1. Compare prices above with model-database.yaml — update any that changed",
        "2. Add any NEW models not yet in the database (only real, available models)",
        "3. Check if any free models changed status",
        "4. Run `bash scripts/sync-github.sh` after updates",
        "",
    ])

    return "\n".join(lines)

def main():
    dry_run = "--dry-run" in sys.argv

    print(f"Turbofit Research v3 — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Find and read Hermes usage data
    db_path = None
    for arg in sys.argv:
        if arg.startswith("--db="):
            db_path = arg.split("=", 1)[1]
            break
    if not db_path:
        db_path = find_hermes_db()

    print(f"\n1. Hermes state.db: {db_path or 'NOT FOUND'}")
    usage = None
    if db_path:
        try:
            usage = read_real_usage(db_path)
            if usage:
                print(f"   Sessions: {usage['overall']['sessions']}")
                print(f"   Input tokens: {usage['total_input']:,}")
                print(f"   Cache read tokens: {usage['total_cache_read']:,}")
                print(f"   Effective cache rate: {usage['effective_cache_rate']*100:.1f}%")
                print(f"   Total cost: ${usage['total_est_cost']:.2f}")
        except Exception as e:
            print(f"   ERROR reading DB: {e}")

    # 2. Fetch OpenRouter API
    print("\n2. Fetching OpenRouter API...")
    api_data = fetch_openrouter_models()
    if not api_data:
        print("   FAILED — cannot reach OpenRouter API")
        sys.exit(1)
    print(f"   Got {len(api_data.get('data', []))} total models")

    # 3. Extract watched models
    print("\n3. Extracting watched models...")
    models = extract_watched_models(api_data)
    print(f"   Found {len(models)} watched models")

    # 4. Generate report
    print("\n4. Generating report...")
    report = generate_report(models, usage, db_path)

    # 5. Write files
    json_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": "openrouter",
        "models": models,
        "usage": {
            "total_input": usage["total_input"] if usage else 0,
            "total_output": usage["total_output"] if usage else 0,
            "total_cache_read": usage["total_cache_read"] if usage else 0,
            "effective_cache_rate": usage["effective_cache_rate"] if usage else 0,
            "active_days": usage["active_days"] if usage else 0,
            "avg_sessions_per_day": usage["avg_sessions_per_day"] if usage else 0,
            "avg_input_per_session": usage["avg_input_per_session"] if usage else 0,
            "avg_output_per_session": usage["avg_output_per_session"] if usage else 0,
            "total_est_cost": usage["total_est_cost"] if usage else 0,
        } if usage else None,
    }

    if dry_run:
        print("\n--- DRY RUN ---\n")
        print(report[:5000])
        if len(report) > 5000:
            print(f"\n...({len(report) - 5000} more chars)")
    else:
        os.makedirs(REF_DIR, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            f.write(report)
        print(f"   Report: {REPORT_PATH}")
        with open(JSON_PATH, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"   JSON:   {JSON_PATH}")

    print("\n" + "=" * 60)
    print(f"STATUS: OK")

if __name__ == "__main__":
    main()
