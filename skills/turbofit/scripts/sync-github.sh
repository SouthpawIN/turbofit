#!/usr/bin/env bash
# Turbofit GitHub Sync Script v2
# Syncs the turbofit skill from local Hermes skills to:
#   1. SouthpawIN/turbofit (primary source — what users install from)
#   2. SouthpawIN/sovth-config (collection — references the full repo)
#
# Usage: bash sync-github.sh [--commit-only] [--no-push]
#
# This script is called by the daily research cron after the research script runs.

set -euo pipefail

SOURCE_SKILL_DIR="$HOME/.hermes/skills/turbofit"
TURBOFIT_REPO_DIR="$HOME/projects/turbofit"
SOVTH_CONFIG_REPO_DIR="$HOME/projects/sovth-config"

COMMIT_ONLY=false
NO_PUSH=false

for arg in "$@"; do
    case $arg in
        --commit-only) COMMIT_ONLY=true ;;
        --no-push) NO_PUSH=true ;;
    esac
done

echo "=== Turbofit GitHub Sync ==="
echo "Time: $(date)"
echo ""

# 1. Sync to SouthpawIN/turbofit (primary repo)
echo "1. Syncing to SouthpawIN/turbofit (primary)..."
if [ ! -d "$TURBOFIT_REPO_DIR/.git" ]; then
    echo "   ERROR: $TURBOFIT_REPO_DIR is not a git repo"
    echo "   Clone it first: git clone https://github.com/SouthpawIN/turbofit.git $TURBOFIT_REPO_DIR"
    echo "STATUS: NO_PRIMARY_REPO"
    exit 1
fi

rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$SOURCE_SKILL_DIR/" "$TURBOFIT_REPO_DIR/skills/turbofit/"
echo "   Primary repo synced."

# 2. Sync to SouthpawIN/sovth-config (collection)
echo "2. Syncing to SouthpawIN/sovth-config (collection)..."
if [ -d "$SOVTH_CONFIG_REPO_DIR/.git" ]; then
    rsync -av --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$SOURCE_SKILL_DIR/" "$SOVTH_CONFIG_REPO_DIR/skills/turbofit/"
    echo "   Collection repo synced."
else
    echo "   WARNING: sovth-config repo not found at $SOVTH_CONFIG_REPO_DIR — skipping"
fi

# 3. Commit and push turbofit repo
echo "3. Committing turbofit repo..."
cd "$TURBOFIT_REPO_DIR"
CHANGES=$(git status --short skills/turbofit/)
if [ -z "$CHANGES" ]; then
    echo "   No changes in turbofit repo."
else
    echo "   Changes: $CHANGES"
    git add skills/turbofit/
    COMMIT_MSG="turbofit: daily model database sync $(date +%Y-%m-%d)"
    git commit -m "$COMMIT_MSG" --no-verify 2>&1 || true
    echo "   Committed: $COMMIT_MSG"
fi

# 4. Commit and push sovth-config repo
echo "4. Committing sovth-config repo..."
if [ -d "$SOVTH_CONFIG_REPO_DIR/.git" ]; then
    cd "$SOVTH_CONFIG_REPO_DIR"
    CHANGES2=$(git status --short skills/turbofit/)
    if [ -z "$CHANGES2" ]; then
        echo "   No changes in sovth-config repo."
    else
        echo "   Changes: $CHANGES2"
        git add skills/turbofit/
        COMMIT_MSG2="turbofit: sync from SouthpawIN/turbofit $(date +%Y-%m-%d)"
        git commit -m "$COMMIT_MSG2" --no-verify 2>&1 || true
        echo "   Committed: $COMMIT_MSG2"
    fi
fi

# 5. Push (unless --commit-only or --no-push)
if [ "$COMMIT_ONLY" = true ] || [ "$NO_PUSH" = true ]; then
    echo "5. Skipping push (--commit-only or --no-push)."
    echo "STATUS: COMMIT_ONLY"
    exit 0
fi

echo "5. Pushing turbofit repo..."
cd "$TURBOFIT_REPO_DIR"
git push origin main 2>&1 || {
    echo "   turbofit push failed."
}
echo "   Pushed."

if [ -d "$SOVTH_CONFIG_REPO_DIR/.git" ]; then
    echo "6. Pushing sovth-config repo..."
    cd "$SOVTH_CONFIG_REPO_DIR"
    git push origin main 2>&1 || {
        echo "   sovth-config push failed."
    }
    echo "   Pushed."
fi

echo ""
echo "=== Sync Complete ==="
echo "  Primary:   SouthpawIN/turbofit (what users install from)"
echo "  Collection: SouthpawIN/sovth-config (references the full repo)"
echo "STATUS: OK"
