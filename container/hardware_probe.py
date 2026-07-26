#!/usr/bin/env python3
"""Cross-platform, best-effort hardware normalization for Turbofit."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys


def run(args: list[str], timeout: int = 5) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return ""


def ram_gb() -> float:
    system = platform.system()
    if system == "Linux":
        text = run(["/bin/sh", "-c", "awk '/MemTotal/ {print $2}' /proc/meminfo"])
        return round(int(text.strip()) / 1024 / 1024, 2) if text.strip().isdigit() else 0
    if system == "Darwin":
        text = run(["sysctl", "-n", "hw.memsize"])
        return round(int(text.strip()) / 1024**3, 2) if text.strip().isdigit() else 0
    if system == "Windows":
        text = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        return round(int(text.strip()) / 1024**3, 2) if text.strip().isdigit() else 0
    return 0


def nvidia() -> list[dict]:
    query = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    text = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    rows = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != 6:
            continue
        try:
            rows.append({"index": int(f[0]), "name": f[1], "vendor": "nvidia", "backend": "cuda", "memory_total_mb": int(f[2]), "memory_used_mb": int(f[3]), "memory_free_mb": int(f[4]), "utilization_pct": int(f[5])})
        except ValueError:
            pass
    return rows


def amd() -> list[dict]:
    text = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if not text:
        text = run(["rocminfo"])
    names = re.findall(r"(?:Card series|Name|Marketing Name)\s*[:|]\s*([^,|\n]+)", text, re.I)
    total = re.findall(r"(?:Total Memory|VRAM Total)\D+(\d+)", text, re.I)
    rows = []
    for i, name in enumerate(names or ["AMD GPU"]):
        mb = int(total[i]) / 1024 / 1024 if i < len(total) else 0
        rows.append({"index": i, "name": name.strip(), "vendor": "amd", "backend": "rocm", "memory_total_mb": round(mb), "memory_used_mb": 0, "memory_free_mb": round(mb), "utilization_pct": 0})
    return rows


def apple() -> list[dict]:
    if platform.system() != "Darwin":
        return []
    text = run(["system_profiler", "SPDisplaysDataType", "-json"])
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    cards = data.get("SPDisplaysDataType", [])
    return [{"index": i, "name": c.get("sppci_model", "Apple GPU"), "vendor": "apple", "backend": "metal", "memory_total_mb": 0, "memory_used_mb": 0, "memory_free_mb": 0, "utilization_pct": 0} for i, c in enumerate(cards)]


def probe() -> dict:
    gpus = nvidia() or amd() or apple()
    total_vram = sum(x["memory_total_mb"] for x in gpus) / 1024
    free_vram = sum(x["memory_free_mb"] for x in gpus) / 1024
    return {"os": platform.system().lower(), "arch": platform.machine(), "system_ram_gb": ram_gb(), "gpus": gpus, "gpu_count": len(gpus), "total_vram_gb": round(total_vram, 2), "free_vram_gb": round(free_vram, 2), "backends": sorted({x["backend"] for x in gpus})}


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
