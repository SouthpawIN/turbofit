#!/usr/bin/env bash
# Turbofit GitHub Sync Script
# Syncs the turbofit skill from local Hermes skills to the sovth-config repo,
# then commits and pushes to GitHub.
#
# Usage: bash sync-github.sh [--commit-only] [--no-push]
#
# This script is called by the daily research cron after the research script runs.

set -euo pipefail

SOURCE_SKILL_DIR="$HOME/.hermes/skills/turbofit"
REPO_SKILL_DIR="$HOME/projects/sovth-config/skills/turbofit"
REPO_DIR="$HOME/projects/sovth-config"

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

# 1. Sync files from Hermes skills to repo
echo "1. Syncing files..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$SOURCE_SKILL_DIR/" "$REPO_SKILL_DIR/"
echo "   Files synced."

# 2. Check for changes
cd "$REPO_DIR"
CHANGES=$(git status --short skills/turbofit/)

if [ -z "$CHANGES" ]; then
    echo "2. No changes detected. Nothing to commit."
    echo "STATUS: NO_CHANGES"
    exit 0
fi

echo "2. Changes detected:"
echo "$CHANGES"
echo ""

# 3. Stage and commit
echo "3. Committing..."
git add skills/turbofit/

COMMIT_MSG="turbofit: daily model database sync $(date +%Y-%m-%d)"

git commit -m "$COMMIT_MSG" --no-verify 2>&1 || {
    echo "   Commit failed (maybe nothing to commit after staging)."
    echo "STATUS: COMMIT_FAILED"
    exit 1
}
echo "   Committed: $COMMIT_MSG"

# 4. Push (unless --commit-only or --no-push)
if [ "$COMMIT_ONLY" = true ] || [ "$NO_PUSH" = true ]; then
    echo "4. Skipping push (--commit-only or --no-push)."
    echo "STATUS: COMMIT_ONLY"
    exit 0
fi

echo "4. Pushing to GitHub..."
git push origin main 2>&1 || {
    echo "   Push failed. Check network or auth."
    echo "STATUS: PUSH_FAILED"
    exit 1
}
echo "   Pushed to origin/main."

echo ""
echo "=== Sync Complete ==="
echo "STATUS: OK"
