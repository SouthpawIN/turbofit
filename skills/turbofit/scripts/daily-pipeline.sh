#!/usr/bin/env bash
# turbofit Daily Pipeline — runs at 3am via cron
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SKILL_DIR/references"
LOG_FILE="$LOG_DIR/daily-pipeline.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== turbofit Daily Pipeline Start ==="

# Step 1: Benchmark
log "Step 1/4: Benchmarking models..."
cd "$SCRIPT_DIR"
python3 benchmark-pipeline.py --push 2>&1 | tee -a "$LOG_FILE" || log "Benchmarks had errors (non-fatal)"

# Step 2: Post to Discord
log "Step 2/4: Posting to Discord..."
python3 post-benchmarks-discord.py 2>&1 | tee -a "$LOG_FILE" || log "Discord post failed (non-fatal)"

# Step 3: Aggregate
log "Step 3/4: Aggregating history..."
python3 aggregate-benchmark-data.py --days 30 2>&1 | tee -a "$LOG_FILE"

# Step 4: GitHub sync
log "Step 4/4: GitHub sync..."
bash sync-github.sh 2>&1 | tee -a "$LOG_FILE" || log "GitHub sync failed (non-fatal)"

log "=== Pipeline Complete ==="
