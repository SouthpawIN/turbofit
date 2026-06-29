#!/usr/bin/env python3
"""
TurboFit Fleet Benchmark Fetcher — for remote installs.

Other turbofit installs run this to pull the latest fleet benchmark numbers
from SouthpawIN/turbofit on GitHub. These numbers come from a dual RTX 3090
setup and give users a reference point for expected performance.

Usage:
    python3 scripts/fetch-fleet-benchmarks.py
    # or
    serve fleet-benchmarks

The data is also available at:
    https://github.com/SouthpawIN/turbofit/blob/main/skills/turbofit/references/fleet-benchmarks.json
"""
import json
import os
import sys
import urllib.request

FLEET_BENCH_URL = "https://raw.githubusercontent.com/SouthpawIN/turbofit/main/skills/turbofit/references/fleet-benchmarks.json"
LOCAL_CACHE = os.path.expanduser("~/.config/turbofit/fleet-benchmarks-cache.json")

def fetch():
    print("Fetching fleet benchmarks from GitHub...")
    try:
        req = urllib.request.Request(FLEET_BENCH_URL, headers={"User-Agent": "turbofit"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
    except Exception as e:
        print(f"Failed to fetch from GitHub: {e}")
        if os.path.exists(LOCAL_CACHE):
            print("Using cached version...")
            with open(LOCAL_CACHE) as f:
                data = json.load(f)
        else:
            print("No cached version available. Run again later.")
            sys.exit(1)
        return

    # Cache locally
    os.makedirs(os.path.dirname(LOCAL_CACHE), exist_ok=True)
    with open(LOCAL_CACHE, "w") as f:
        json.dump(data, f, indent=2)

    # Display
    latest = data.get("latest", {})
    hardware = data.get("hardware", "Unknown")
    updated = data.get("last_updated", "Unknown")

    print(f"\n{'='*60}")
    print(f"TurboFit Fleet Benchmarks — {hardware}")
    print(f"Last updated: {updated}")
    print(f"{'='*60}")
    print(f"{'Model':<28} {'Tier':>4} {'Size':>6} {'tok/s':>8} {'1K tok/s':>10} {'8K ctx':>8}")
    print(f"{'-'*28} {'-'*4} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")

    for alias, info in sorted(latest.items(), key=lambda x: x[1].get("speed_small_tok_s") or 0, reverse=True):
        tier = info.get("tier", "?")
        size = f"{info.get('size_gb', '?')}GB"
        speed = info.get("speed_small_tok_s")
        speed_large = info.get("speed_large_tok_s")
        ctx8k = "yes" if info.get("context_8k_ok") else "no"

        speed_str = f"{speed:.1f}" if speed else "N/A"
        large_str = f"{speed_large:.1f}" if speed_large else "N/A"

        print(f"{alias:<28} {tier:>4} {size:>6} {speed_str:>8} {large_str:>10} {ctx8k:>8}")

    print(f"\n{'='*60}")
    print(f"Full data: {LOCAL_CACHE}")
    print(f"Source: https://github.com/SouthpawIN/turbofit")

if __name__ == "__main__":
    fetch()
