#!/usr/bin/env python3
"""Exercise one Turbofit Main:Aux checklist row and file linked evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turbofit_runtime.benchmark_schema import (  # noqa: E402
    PromotionRejected,
    load_record,
    load_suite,
    require_promotion,
)


def request(url: str, payload: dict | None = None, timeout: int = 180) -> tuple[int, dict]:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}
    except Exception as e:
        return 0, {"error": str(e)}


def gpu_snapshot() -> list[dict]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5)
    except Exception:
        return []
    result = []
    for line in raw.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 5:
            result.append(dict(zip(("index", "name", "memory_used_mb", "memory_free_mb", "utilization_pct"), fields)))
    return result


def mark_row(checklist: Path, anchor: str, evidence_link: str) -> None:
    text = checklist.read_text()
    marker = f'<a id="{anchor}"></a>\n- [ ]'
    if marker not in text:
        raise SystemExit(f"anchor not found or row already marked: {anchor}")
    text = text.replace(marker, f'<a id="{anchor}"></a>\n- [x]', 1)
    row_start = text.index(f'<a id="{anchor}"></a>')
    next_newline = text.index("\n", row_start)
    insert_at = text.index("\n", next_newline + 1)
    text = text[:insert_at] + f"\n  - Evidence: {evidence_link}" + text[insert_at:]
    checklist.write_text(text)


def promotion_allowed(record_path: str | None, suite_path: str) -> tuple[bool, str | None]:
    if not record_path:
        return False, "--promotion-record is required with --mark-success"
    try:
        require_promotion(load_record(record_path), load_suite(suite_path))
    except (OSError, ValueError, RuntimeError, PromotionRejected) as exc:
        return False, str(exc)
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--aux", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--label", required=True)
    wiki_topic = Path.home() / ".hermes" / "wiki" / "topics" / "turbofit"
    parser.add_argument("--checklist", default=str(wiki_topic / "main-aux-inference-checklist.md"))
    parser.add_argument("--evidence-dir", default=str(wiki_topic / "evidence"))
    parser.add_argument("--mark-success", action="store_true")
    parser.add_argument("--promotion-record")
    parser.add_argument("--suite", default=str(ROOT / "benchmarks" / "suite.yaml"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    health_status, health = request(base + "/health", timeout=10)
    started = time.monotonic()
    smoke_status, smoke = request(base + "/v1/chat/completions", {
        "model": args.main,
        "messages": [{"role": "user", "content": "Reply exactly: matrix smoke passed."}],
        "max_tokens": 32,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    elapsed = round(time.monotonic() - started, 3)
    content = ((smoke.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    timings = smoke.get("timings", {})
    passed = health_status == 200 and smoke_status == 200 and bool(content)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = evidence_dir / f"{args.anchor}.md"
    evidence.write_text(f"""---
title: Matrix evidence - {args.label}
created: {stamp[:10]}
updated: {stamp[:10]}
type: benchmark
tags: [turbofit, benchmark, inference]
---

# Matrix evidence: `{args.label}`

- Checklist: [[../main-aux-inference-checklist#{args.anchor}|exact checklist row]]
- Timestamp: `{stamp}`
- Main: `{args.main}`
- Auxiliary: `{args.aux}`
- Requested context: `{args.context}`
- Health status: `{health_status}`
- Smoke status: `{smoke_status}`
- Smoke response: `{content}`
- Wall time: `{elapsed}s`
- Prompt tok/s: `{timings.get('prompt_per_second', 'n/a')}`
- Generation tok/s: `{timings.get('predicted_per_second', 'n/a')}`
- GPU snapshot: `{json.dumps(gpu_snapshot())}`
- Result: `{'exercised' if passed else 'failed'}`
""")
    link = f"[matrix evidence](evidence/{evidence.name})"
    promotion_gate = False
    promotion_error = None
    if args.mark_success:
        promotion_gate, promotion_error = promotion_allowed(args.promotion_record, args.suite)
    if passed and args.mark_success and promotion_gate:
        mark_row(Path(args.checklist), args.anchor, link)
    marked = bool(passed and args.mark_success and promotion_gate)
    print(json.dumps({"passed": passed, "health_status": health_status, "smoke_status": smoke_status, "evidence": str(evidence), "promotion_gate": promotion_gate, "promotion_error": promotion_error, "marked_success": marked, "timings": timings}, indent=2))
    return 0 if passed and (not args.mark_success or promotion_gate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
