# turbofit v1.1 — opinionated unified LLM backend

Hardware-fit checker + multi-launcher orchestrator for Hermes-Agent. Uses `llmfit` to verify model fit, generates accurate launch strings for **llama.cpp / Ollama / vLLM / SGlang**, launches detached, wires Hermes config, and adapts to live VRAM pressure via a scaling ladder. Dynamic model database auto-updates daily with real pricing from the OpenRouter API.

End-user UX: `serve auto main` → done.

## Install

```bash
hermes skills install SouthpawIN/turbofit/skills/turbofit
```

## What's new in v1.1

- **Dynamic model database** (`references/model-database.yaml`) — auto-updated daily via research cron
- **API pairing matrix** — optimal main+aux pairings across 4 context tiers and 5 price tiers
- **Scaling ladders for all hardware tiers** — Beefy (7-step), Modest (5-step), Thin (4-step)
- **Cache pricing** — `input_cache_read` rates from OpenRouter API for all models that support it
- **Monthly cost projections** — for light/moderate/heavy/extreme usage profiles
- **Daily research pipeline** — `scripts/research-models.py` fetches live data from OpenRouter API
- **GitHub sync** — `scripts/sync-github.sh` pushes updates automatically

## Structure

```
turbofit/
├── README.md
└── skills/
    └── turbofit/
        ├── SKILL.md
        ├── distribution.yaml
        ├── references/
        │   ├── model-database.yaml      # Dynamic source of truth
        │   ├── model-pricing.json        # Machine-readable live pricing
        │   ├── research-report.md        # Latest research report
        │   ├── api-pairing-matrix.md     # Pairing recommendations
        │   ├── scaling-ladder.md         # All-tier scaling ladders
        │   ├── curated-lineup.md         # Model archetypes + pairing rules
        │   ├── api-model-rankings.md     # Individual model rankings
        │   ├── api-tier-rankings.md      # Quick-reference tier rankings
        │   └── binary-selection.md       # Atomic fork vs stock decision tree
        └── scripts/
            ├── serve                     # Main bash script (2100+ lines)
            ├── research-models.py        # Daily research script
            ├── sync-github.sh            # GitHub sync script
            ├── install.sh                # Shell function installer
            └── turbofit.sharco           # Shell shim
```

## See also

- [sovth-config](https://github.com/SouthpawIN/sovth-config) — overarching config collection (profiles, plugins, skills)
