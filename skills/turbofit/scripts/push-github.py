#!/usr/bin/env python3
"""
Push turbofit benchmark results and training analysis to GitHub.

Commits and pushes benchmark-results.json and benchmark-analysis.md
to SouthpawIN/turbofit so that every turbofit install pulls the latest
live benchmark data from our daily runs.
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

REPO_DIR = os.path.expanduser("~/projects/turbofit")
RESULTS_PATH = os.path.expanduser(
    "~/.hermes/skills/turbofit/references/benchmark-results.json"
)
ANALYSIS_PATH = os.path.expanduser(
    "~/.hermes/skills/turbofit/references/benchmark-analysis.md"
)
TARGET_RESULTS = "references/benchmark-results.json"
TARGET_ANALYSIS = "references/benchmark-analysis.md"


def run_cmd(cmd, cwd=None):
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=60
        )
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)


def main():
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print(f"ERROR: Not a git repo: {REPO_DIR}")
        sys.exit(1)

    print("=== Turbofit GitHub Sync ===")

    if os.path.exists(RESULTS_PATH):
        dst = os.path.join(REPO_DIR, TARGET_RESULTS)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(RESULTS_PATH, dst)
        print(f"  Copied benchmark-results.json")
    else:
        print(f"  WARNING: benchmark-results.json not found at {RESULTS_PATH}")

    if os.path.exists(ANALYSIS_PATH):
        dst = os.path.join(REPO_DIR, TARGET_ANALYSIS)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(ANALYSIS_PATH, dst)
        print(f"  Copied benchmark-analysis.md")

    ok, status = run_cmd("git status --porcelain", cwd=REPO_DIR)
    if not ok:
        print(f"ERROR: git status failed\n{status}")
        sys.exit(1)

    if not status.strip():
        print("  No changes to commit.")
        sys.exit(0)

    print(f"\nChanges detected, committing...")
    run_cmd("git add references/benchmark-results.json references/benchmark-analysis.md", cwd=REPO_DIR)

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = f"bench: daily results and training analysis {date_str}"
    ok, out = run_cmd(f'git commit -m "{msg}"', cwd=REPO_DIR)
    if not ok:
        print(f"  ERROR: commit failed\n{out}")
        sys.exit(1)
    print(f"  Committed: {msg}")

    print("  Pushing...")
    ok, out = run_cmd("git push origin main", cwd=REPO_DIR)
    if ok:
        print("  ✅ Pushed to SouthpawIN/turbofit")
    else:
        print(f"  ❌ Push failed: {out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
