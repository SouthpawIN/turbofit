# TurboFit — Workflows, Checkpoints & Versioning

One source of truth, two repos.

| Repo | Role |
|------|------|
| `SouthpawIN/agent-wiki` → `wiki-topics/turbofit/` | The **wiki** — the human-facing source of truth (overview + curated references). |
| `SouthpawIN/turbofit` (this repo) | The **product** — the adaptive runtime + canonical machine-readable data (`fit-lists.json`, `model-catalog.json`). |

## The README mirror

- Canonical doc: the wiki hub (`agent-wiki` → `wiki-topics/turbofit/README.md`).
- This repo's `README.md` is a **generated mirror** of that hub, so the public GitHub README always matches the wiki.
- To update: edit the wiki hub → run `tools/sync-wiki-readme.sh` → commit both repos.
- A literal cross-repo symlink is deliberately *not* used: a symlink to another repo renders as a broken link on the public GitHub README, so a synced copy is the GitHub-safe way to keep them identical.

## Worktree + checkpoint workflow (code changes)

Work in a worktree so `main` stays clean and every step is a reviewable checkpoint:

    cd ~/turbofit
    git worktree add -b <branch> ~/turbofit-wt-<branch>
    # ... make + verify the change in the worktree ...
    cd ~/turbofit-wt-<branch>
    git add -A && git commit -m "<what + why>"            # <- checkpoint
    cd ~/turbofit
    git merge --no-ff <branch>                           # <- merge checkpoint
    git branch -d <branch> && git worktree remove ~/turbofit-wt-<branch>
    git push origin main

- Every `git commit` is a **checkpoint**; each branch is an isolated experiment.
- Merge back to `main` only after the change is verified (build / tests / a `serve` smoke).

## Versioning

- **Wiki checkpoints** — commits on `agent-wiki` main (the flywheel commits topic changes).
- **Repo checkpoints** — commits on this repo's main (worktree branches merged back).
- **Tags** mark released rungs and provider versions.
