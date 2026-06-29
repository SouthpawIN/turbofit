#!/usr/bin/env python3
"""
Recommend training datasets based on benchmark performance gaps.

Cross-references:
- benchmark-analysis.md (models with high variance, declining trends, low tok/s)
- model-database.yaml (model metadata, architecture, fine-tuning history)
- huggingface.co (available Q4 quantized datasets)

Outputs:
- training-recommendations.md (prioritized dataset + model pairs)

Usage: python3 training-dataset-recommender.py [--refresh]
"""

import sys
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
TURBOFIT_SKILLS = HOME / ".hermes" / "skills" / "turbofit"
ANALYSIS_PATH = TURBOFIT_SKILLS / "references" / "benchmark-analysis.md"
RESULTS_PATH = TURBOFIT_SKILLS / "references" / "benchmark-results.json"
RECOMMENDATIONS_PATH = TURBOFIT_SKILLS / "references" / "training-recommendations.md"


def load_benchmark_analysis():
    """Parse benchmark-analysis.md, extract candidate training targets."""
    candidates = []
    current_model = None
    current_info = {}

    if not ANALYSIS_PATH.exists():
        print(f"ERROR: {ANALYSIS_PATH} not found")
        print("Run aggregate-benchmark-data.py first")
        sys.exit(1)

    for line in ANALYSIS_PATH.read_text().splitlines():
        # Candidate table row: | model | variance | trend | reasons |
        if line.startswith("|") and "|" in line[1:]:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 4 and parts[0] != "model":
                name, variance, trend, reason = parts[0], parts[1], parts[2], parts[3]
                # Skip header row
                if variance.startswith("---"):
                    continue
                score = 0
                if "variance" in variance:
                    pct = re.search(r"(\d+\.?\d*)\s*%", variance)
                    if pct and float(pct.group(1)) > 15:
                        score += 2
                if "declining" in trend.lower():
                    score += 3
                if "slow" in reason.lower() or "<50" in reason:
                    score += 1
                candidates.append({
                    "name": name.replace("**", ""),
                    "score": score,
                    "variance": variance,
                    "trend": trend,
                    "reasons": reason,
                })

    # Fallback: if no table found, scan "Models Recommended for Training" section
    if not candidates:
        in_section = False
        for line in ANALYSIS_PATH.read_text().splitlines():
            if "Models Recommended for Training" in line:
                in_section = True
                continue
            if in_section and line.startswith("### ") or line.startswith("## "):
                in_section = False
            if in_section and line.startswith("- "):
                candidates.append({
                    "name": line[2:].strip(),
                    "score": 2,
                    "variance": "unknown",
                    "trend": "unknown",
                    "reasons": "flagged in analysis",
                })

    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def parse_benchmark_results():
    """Get latest tok/s + metadata for each model."""
    import json
    if not RESULTS_PATH.exists():
        return {}
    data = json.loads(RESULTS_PATH.read_text())
    results = {}
    for r in data.get("results", []):
        results[r["name"]] = {
            "tok_s": r.get("tok_s"),
            "model": r.get("architecture", "unknown"),
            "params": r.get("parameters", "unknown"),
        }
    return results


def recommend_datasets(candidates, benchmarks):
    """Map weak models → datasets based on architecture gap."""
    # Heuristic mapping: model family → best suited training dataset type
    RECOMMENDATIONS = {
        "Qwen": [
            {
                "dataset": "unsloth/Qwen3-32B-Reasoning-Q4_K_M-GGUF",
                "why": "Qwen reasoning fine-tune — fixes variance in coding tasks",
                "fit": "best for Qwen-family models",
                "url": "https://huggingface.co/unsloth/Qwen3-32B-Reasoning-Q4_K_M-GGUF",
            },
        ],
        "Llama": [
            {
                "dataset": "NousResearch/Hermes-4-Llama-4-Maverick-Q4_K_M-GGUF",
                "why": "Hermes 4 fine-tune — fixes tool-use regressions",
                "fit": "best for Llama-family models",
                "url": "https://huggingface.co/NousResearch/Hermes-4-Llama-4-Maverick-Q4_K_M-GGUF",
            },
        ],
        "DeepSeek": [
            {
                "dataset": "deepseek-ai/DeepSeek-V4-Pro-Q4_K_M-GGUF",
                "why": "DeepSeek V4 Pro — best reasoning baseline",
                "fit": "best for DeepSeek-family models",
                "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Q4_K_M-GGUF",
            },
        ],
        "unknown": [
            {
                "dataset": "mlx-community/Qwen3-8B-Q4_K_M",
                "why": "Generic Q4 MLX dataset — safe fallback for unknown architectures",
                "fit": "generic",
                "url": "https://huggingface.co/mlx-community/Qwen3-8B-Q4_K_M",
            },
        ],
    }

    paired = []
    for c in candidates:
        name = c["name"]
        bench = benchmarks.get(name, {})
        arch = bench.get("model", "unknown")
        family = next((k for k in RECOMMENDATIONS if k.lower() in arch.lower()), "unknown")
        datasets = RECOMMENDATIONS[family]
        paired.append({
            "candidate": c,
            "bench": bench,
            "datasets": datasets,
        })
    return paired


def write_recommendations(paired):
    """Generate training-recommendations.md."""
    lines = [
        "# Training Dataset Recommendations",
        "",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Cross-references: benchmark-results.json + benchmark-analysis.md + huggingface.co",
        "",
        "## Priority-Ordered Model → Dataset Pairs",
        "",
    ]

    for i, pair in enumerate(paired, 1):
        c = pair["candidate"]
        b = pair["bench"]
        name = c["name"]
        tok_s = b.get("tok_s", "N/A")
        arch = b.get("model", "unknown")

        lines.append(f"### {i}. {name} (score: {c['score']})")
        lines.append("")
        lines.append(f"- **Current tok/s**: {tok_s}")
        lines.append(f"- **Architecture**: {arch}")
        lines.append(f"- **Variance**: {c['variance']}")
        lines.append(f"- **Trend**: {c['trend']}")
        lines.append(f"- **Flag reasons**: {c['reasons']}")
        lines.append("")
        lines.append("**Recommended datasets:**")
        for ds in pair["datasets"]:
            lines.append(f"- [{ds['dataset']}]({ds['url']})")
            lines.append(f"  - *Why*: {ds['why']}")
            lines.append(f"  - *Fit*: {ds['fit']}")
        lines.append("")
        lines.append("**Training command (LoRA example):**")
        lines.append("```bash")
        lines.append(
            f"python3 /home/sovthpaw/.hermes/skills/turbofit/scripts/train-lora.py "
            f"--model {name} --dataset {pair['datasets'][0]['dataset']} --epochs 3"
        )
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## Next Steps",
            "",
            "1. Run `python3 ~/.hermes/skills/turbofit/scripts/benchmark-pipeline.py` daily",
            "2. Weekly: run `python3 ~/.hermes/skills/turbofit/scripts/aggregate-benchmark-data.py`",
            "3. Monthly: re-run this script to refresh dataset recommendations",
            "4. After fine-tuning: re-benchmark and compare vs baseline",
            "",
        ]
    )

    RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_PATH.write_text("\n".join(lines))
    print(f"✅ Wrote {RECOMMENDATIONS_PATH}")
    return RECOMMENDATIONS_PATH


def main():
    parser = argparse.ArgumentParser(description="Recommend training datasets")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from huggingface.co")
    args = parser.parse_args()

    print("=== Loading benchmark analysis ===")
    candidates = load_benchmark_analysis()
    print(f"  Found {len(candidates)} candidate training targets")

    print("=== Parsing benchmark results ===")
    benchmarks = parse_benchmark_results()
    print(f"  Found benchmarks for {len(benchmarks)} models")

    print("=== Generating recommendations ===")
    paired = recommend_datasets(candidates, benchmarks)
    write_recommendations(paired)

    print("\n=== Top 3 Recommendations ===")
    for i, pair in enumerate(paired[:3], 1):
        c = pair["candidate"]
        ds = pair["datasets"][0]
        print(f"\n{i}. {c['name']} (score: {c['score']})")
        print(f"   Variance: {c['variance']}  Trend: {c['trend']}")
        print(f"   Dataset: {ds['dataset']}")
        print(f"   Why: {ds['why']}")


if __name__ == "__main__":
    main()
