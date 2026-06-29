#!/bin/bash
# Daily benchmark pipeline — benchmarks all catalog models and pushes to GitHub
# Runs at 2am via cron

set -e

LOG=~/.local/share/turbofit/logs/daily-benchmark.log
mkdir -p "$(dirname "$LOG")"

echo "=== $(date) ===" >> "$LOG"
echo "Starting daily benchmark pipeline..." >> "$LOG"

# Run the benchmark pipeline
python3.12 ~/.hermes/skills/turbofit/scripts/benchmark-pipeline.py \
  --push \
  --gpu 0 \
  --ctx 65536 \
  --tasks mmlu,gsm8k,gpqa,humaneval \
  --limit 100 \
  >> "$LOG" 2>&1

echo "Benchmark pipeline complete." >> "$LOG"
echo "=== End ===" >> "$LOG"
