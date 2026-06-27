#!/usr/bin/env python3
"""Post turbofit benchmark results to Discord #benchmarks channel."""
import json, os, sys
import requests

HOME = os.path.expanduser("~")
RESULTS_PATH = f"{HOME}/.hermes/skills/turbofit/references/benchmark-results.json"
ENV_PATH = os.path.join(HOME, ".hermes", "profiles", "frieza", ".env")
CHANNEL_ID = "1520328040427688068"

def load_token():
    if not os.path.exists(ENV_PATH):
        print(f"ERROR: .env not found: {ENV_PATH}")
        sys.exit(1)
    import re
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if re.match(r"^DISCORD_BOT", line) and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("ERROR: token not in .env")
    sys.exit(1)

def grade_emoji(composite):
    if composite is None: return "—"
    if composite >= 90: return "🧠 A+"
    if composite >= 80: return "🧠 A"
    if composite >= 70: return "🧠 B"
    if composite >= 60: return "🧠 C"
    if composite >= 50: return "🧠 D"
    return "🧠 F"

def format_msg(data):
    date = data.get("date", "?")
    ctx = data.get("ctx", 0)
    gpu = data.get("gpu_name", "?")
    results = data.get("results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    fail = [r for r in results if r.get("status") != "ok"]
    
    # Sort by reasoning composite (intelligence first), speed as tiebreaker
    def rank_key(r):
        rc = r.get("reasoning_composite") or 0
        speed = r.get("tok_s", 0)
        return (rc, speed)
    
    ranked = sorted(ok, key=rank_key, reverse=True)
    
    lines = [
        f"🧠 **turbofit Daily Benchmarks** — {date}",
        f"GPU: {gpu} | Context: {ctx:,} tokens",
        f"✅ {len(ok)} models | ❌ {len(fail)} failed/skipped",
        "", "## Smartest Models (ranked by reasoning)"
    ]
    
    for i, r in enumerate(ranked[:10], 1):
        rc = r.get("reasoning_composite")
        iq_str = f"{rc:.0f}" if rc is not None else "?"
        mtp = "🔥" if r.get("has_mtp") else ""
        vis = "👁️" if r.get("has_vision") else ""
        lines.append(
            f"**{i}. {r['name']}** — IQ {iq_str}/100 | {r['tok_s']:.1f} tok/s "
            f"({r.get('size_gb',0):.1f}G) {r.get('tier','?')}/{r.get('role','?')} {mtp}{vis}"
        )
    
    # Fastest section for speed demons
    speed_ranked = sorted(ok, key=lambda x: x.get("tok_s", 0), reverse=True)
    lines += ["", "## Fastest Models"]
    for i, r in enumerate(speed_ranked[:5], 1):
        rc = r.get("reasoning_composite")
        iq_str = f"IQ {rc:.0f}" if rc is not None else ""
        lines.append(
            f"**{i}. {r['name']}** — {r['tok_s']:.1f} tok/s {iq_str} "
            f"({r.get('size_gb',0):.1f}G)"
        )
    
    if fail:
        lines += ["", "## Failed/Skipped"]
        for r in fail[:5]:
            lines.append(f"- {r.get('name','?')}: {r.get('reason','?')}")
    lines += ["", "🔗 [SouthpawIN/turbofit](https://github.com/SouthpawIN/turbofit)"]
    return "\n".join(lines)

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"No results at {RESULTS_PATH}")
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    token = load_token()
    msg = format_msg(data)
    if len(msg) > 2000:
        msg = msg[:1997] + "..."
    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"content": msg}, timeout=10)
    if resp.status_code in (200, 201):
        print(f"✅ Posted {len(msg)} chars to Discord")
    else:
        print(f"❌ Failed {resp.status_code}: {resp.text[:200]}")

if __name__ == "__main__":
    main()
