---
name: turbofit
description: "Compatibility entry for the repository's canonical Turbofit adaptive-runtime skill. Use for portable Turbofiles, Turbohaul residency, stable Hermes routes, pressure adaptation, and evidence-backed release gates."
version: 2.0.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, turbohaul, adaptive-runtime]
---

# Turbofit

The canonical skill documentation is [`../../SKILL.md`](../../SKILL.md). Load and follow that file; this compatibility entry intentionally contains no second operating procedure.

Core invariants:

- Work from the Git repository, not the installed skill directory.
- Turbohaul Manager v0.7 is the sole local residency authority.
- Use stable `auto`, `active:main`, and `active:aux` routes.
- External GPU consumers have absolute priority and are never signaled.
- A real release requires `scripts/release-check --real`; simulation alone is not acceptance.
