#!/usr/bin/env python3
"""Run a physical-host auxiliary quality/tool/performance tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from turbofit_runtime.backend import CampaignBackend
from turbofit_runtime.recipes import RecipeBook

QUALITY = (
    ("capital", "What is the capital of France? Answer briefly.", lambda s: "paris" in s.lower()),
    ("science", "What is the chemical symbol for gold? Answer with only the symbol.", lambda s: s.strip().lower() == "au"),
    ("arithmetic", "What is 234 * 567? Answer with only the integer.", lambda s: re.search(r"\b132678\b", s) is not None),
    ("reasoning", "Alice has 5 apples, buys 12 more, then gives 8 away. How many remain? Answer with only the integer.", lambda s: s.strip() == "9"),
    ("instruction", "Answer with exactly one word: What is 2 + 2?", lambda s: s.strip() == "4"),
)
TOOLS = (
    ("weather", "What is the weather in Tokyo?", "get_weather", {"location": "Tokyo"}),
    ("calendar", "Schedule a meeting called Review tomorrow at 3pm.", "create_calendar_event", {"title": "Review"}),
    ("multiply", "Calculate 15 * 23 using the available tool.", "multiply", {"a": 15, "b": 23}),
)
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "datetime": {"type": "string"}}, "required": ["title", "datetime"]}}},
    {"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}},
]


def post(port: int, alias: str, payload: dict, timeout: float = 300) -> dict:
    payload = {"model": alias, "temperature": 0, **payload}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    body["wall_seconds"] = time.monotonic() - started
    return body


def tool_score(message: dict, expected: str, expected_args: dict) -> tuple[bool, dict]:
    calls = message.get("tool_calls") or []
    if not calls:
        return False, {}
    function = calls[0].get("function") or {}
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, {}
    if function.get("name") != expected:
        return False, arguments
    for key, value in expected_args.items():
        actual = arguments.get(key)
        if isinstance(value, str):
            if str(actual).lower() != value.lower():
                return False, arguments
        elif actual != value:
            return False, arguments
    return True, arguments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_candidate(family: str, *, context: int, gpu: str, results_dir: Path) -> dict:
    recipes = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cuda")
    component = recipes.resolve_component(
        family, role="aux", gpu=gpu, port=11700,
        context=context, alias=f"aux-bench-{family}",
    )
    backend = CampaignBackend(
        gateway_script=ROOT / "scripts/turbofit-gateway.py",
        gateway_port=11701,
        result_dir=results_dir,
        runtime_state=results_dir / "runtime.json",
        production_gateway_service="turbofit-controller.service",
        accelerator_backend="cuda",
    )
    handle = None
    result = {
        "family": family,
        "context": context,
        "gpu": gpu,
        "method": component.method,
        "model_path": component.model_path,
        "model_sha256": sha256_file(Path(component.model_path)),
        "command": list(component.command),
    }
    try:
        handle = backend.start(component)
        result["health"] = backend.wait_ready(component, handle)
        quality = []
        for name, prompt, scorer in QUALITY:
            body = post(component.port, component.alias, {
                "messages": [{"role": "user", "content": prompt}], "max_tokens": 512,
            })
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            quality.append({"name": name, "pass": bool(scorer(content)), "content": content, "wall_seconds": body["wall_seconds"], "timings": body.get("timings", {})})
        tools = []
        for name, prompt, expected, args in TOOLS:
            body = post(component.port, component.alias, {
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOL_SCHEMAS, "tool_choice": "auto", "max_tokens": 512,
            })
            message = body["choices"][0]["message"]
            passed, parsed = tool_score(message, expected, args)
            tools.append({"name": name, "pass": passed, "expected": expected, "arguments": parsed, "message": message, "wall_seconds": body["wall_seconds"]})
        throughput = []
        for _ in range(3):
            body = post(component.port, component.alias, {
                "messages": [{"role": "user", "content": "Write a concise Python merge sort implementation with type hints."}],
                "max_tokens": 256,
            })
            throughput.append(float((body.get("timings") or {}).get("predicted_per_second", 0)))
        result.update({
            "status": "completed",
            "quality": quality,
            "quality_passes": sum(item["pass"] for item in quality),
            "quality_total": len(quality),
            "tools": tools,
            "tool_passes": sum(item["pass"] for item in tools),
            "tool_total": len(tools),
            "decode_tps_samples": throughput,
            "decode_tps_median": statistics.median(throughput),
            "gpu_peak_mb": backend.peak_gpu_mb(),
        })
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc)})
    finally:
        if handle is not None:
            backend.stop(component, handle)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "references/results/auxiliary-tournament-48gb.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "turbofit.auxiliary-tournament/v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "physical_tier": "hardware-48gb",
        "candidates": [benchmark_candidate(name, context=args.context, gpu=args.gpu, results_dir=args.output.parent / "auxiliary-tournament-logs") for name in args.candidate],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": [{k: item.get(k) for k in ("family", "status", "quality_passes", "tool_passes", "decode_tps_median", "error")} for item in payload["candidates"]]}, indent=2))
    return 0 if all(item["status"] == "completed" for item in payload["candidates"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
