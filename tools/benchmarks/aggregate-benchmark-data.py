#!/usr/bin/env python3
"""Aggregate benchmark history for training dataset selection."""

import json
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

HOME = os.path.expanduser("~")
SKILL_DIR = f"{HOME}/.hermes/skills/turbofit"
RESULTS_DIR = f"{SKILL_DIR}/references"


def find_benchmark_files(days=30):
    cutoff = datetime.now() - timedelta(days=days)
    files = []
    for path in Path(RESULTS_DIR).glob("benchmark-results*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                date_str = data.get("date", "")
                if date_str:
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date >= cutoff:
                        files.append((file_date, data, path))
        except Exception:
            continue
    return sorted(files, key=lambda x: x[0])


def aggregate_performance(files):
    model_history = defaultdict(list)
    for date, data, path in files:
        for result in data.get("results", []):
            if result.get("status") == "ok":
                name = result.get("name", "unknown")
                tok_s = result.get("tok_s", 0)
                if tok_s > 0:
                    model_history[name].append({
                        "date": date.strftime("%Y-%m-%d"),
                        "tok_s": tok_s,
                        "size_gb": result.get("size_gb", 0),
                        "tier": result.get("tier", "?"),
                        "has_mtp": result.get("has_mtp", False),
                        "has_vision": result.get("has_vision", False)
                    })
    return model_history


def analyze_trends(model_history):
    analysis = []
    for model, runs in model_history.items():
        if len(runs) < 2:
            continue
        runs = sorted(runs, key=lambda x: x["date"])
        values = [r["tok_s"] for r in runs]
        avg = sum(values) / len(values)
        mn, mx = min(values), max(values)
        latest = values[-1]
        variance = ((mx - mn) / avg * 100) if avg > 0 else 0
        
        trend = "stable"
        if len(runs) >= 3:
            recent = sum(values[-3:]) / 3
            early = sum(values[:3]) / 3
            if recent > early * 1.05: trend = "improving"
            elif recent < early * 0.95: trend = "declining"
        
        analysis.append({
            "model": model, "runs": len(runs), "avg_tok_s": avg,
            "min_tok_s": mn, "max_tok_s": mx, "latest_tok_s": latest,
            "variance": variance, "trend": trend,
            "has_mtp": runs[-1].get("has_mtp", False),
            "has_vision": runs[-1].get("has_vision", False),
            "tier": runs[-1].get("tier", "?"),
            "size_gb": runs[-1].get("size_gb", 0)
        })
    return sorted(analysis, key=lambda x: x["avg_tok_s"], reverse=True)


def identify_candidates(analysis):
    candidates = []
    for m in analysis:
        score = 0
        if m["variance"] > 15: score += 2
        if m["trend"] == "declining": score += 3
        if m["avg_tok_s"] < 50: score += 1
        if not m["has_mtp"]: score += 2
        if m["size_gb"] > 15 and m["avg_tok_s"] < 60: score += 2
        if score >= 3:
            candidates.append({**m, "score": score})
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def generate_report(analysis, candidates, days):
    lines = [
        f"# turbofit Benchmark Analysis — Last {days} Days",
        "", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
        "## Performance Summary", "",
        "| Model | Runs | Avg tok/s | Min | Max | Variance | Trend | MTP | Vision |",
        "|-------|------|-----------|-----|-----|----------|-------|-----|--------|"
    ]
    for m in analysis[:20]:
        mtp = "✅" if m["has_mtp"] else "❌"
        vis = "✅" if m["has_vision"] else "❌"
        lines.append(
            f"| {m['model']} | {m['runs']} | {m['avg_tok_s']:.1f} | "
            f"{m['min_tok_s']:.1f} | {m['max_tok_s']:.1f} | "
            f"{m['variance']:.1f}% | {m['trend']} | {mtp} | {vis} |"
        )
    
    if candidates:
        lines.extend([
            "", "## Training Candidates", "",
            "Models needing optimization or retraining:", "",
            "| Model | Score | Avg tok/s | Variance | Trend | MTP | Size |",
            "|-------|-------|-----------|----------|-------|-----|------|"
        ])
        for c in candidates[:10]:
            mtp = "✅" if c["has_mtp"] else "❌"
            lines.append(
                f"| {c['model']} | {c['score']} | {c['avg_tok_s']:.1f} | "
                f"{c['variance']:.1f}% | {c['trend']} | {mtp} | {c['size_gb']:.1f}G |"
            )
    
    lines.extend([
        "", "## Connected Workflows", "",
        "- **Daily**: `daily-pipeline.sh` (cron 3am)",
        "- **DeepSpec**: https://github.com/deepseek-ai/DeepSpec for draft model training",
        "- **Training data**: Use this analysis to select fine-tuning datasets"
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    files = find_benchmark_files(args.days)
    if not files:
        print("No benchmark files found")
        return
    
    history = aggregate_performance(files)
    analysis = analyze_trends(history)
    candidates = identify_candidates(analysis)
    report = generate_report(analysis, candidates, args.days)
    
    report_path = f"{RESULTS_DIR}/benchmark-analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Analysis saved: {report_path}")
    print(f"Models tracked: {len(analysis)}")
    print(f"Training candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
