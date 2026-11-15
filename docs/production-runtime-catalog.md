# Production runtime catalog authority

The selectable catalog is `runtime-profiles/*gb.yaml`. Its native targets and
pressure requirements are `runtime-resolutions.json` and
`rung-requirements.json` in that directory. Validate these stores together:

```bash
.venv/bin/python -m pytest tests/integration/test_shipped_runtime_catalog.py -q
```

## Current authority boundary

- **8 GB:** Maple's existing user-authorized native 64K/128K promotion is
  unchanged. Authority: `references/results/maple-preview-small-device-evidence.json`.
  This does not imply the binary or artifacts are installed on this machine.
- **48 GB, 2x24:** only the measured Unleashed UD-Q3_K_XL **baseline** 262K
  winner is selectable locally, with API safety as its floor. Authority:
  `references/results/hardware-tier-winner-48gb-qwen38-unleashed-20260823.json`.
  The resolution uses the baseline family (not the separately updated DFlash2
  variant), pinned GPU 1 / port 11605, and measured 19391 MiB residency.
  `main_manifest` is the promotion's recipe digest, not its hardware fingerprint.
- **Other shipped tiers, including Apple:** API-only until exact topology,
  recipe, and promotion authority is available. A 2x24 promotion does not
  certify 1x24, 2x32, Apple, or larger-card/multi-card defaults. An unproven
  canonical tier outranks a smaller tier's local promotion in Auto selection.

API-only evidence addresses hash actual `class-status/*.json` records declaring
`local_status: unproven`; they are **not** fabricated benchmark results or model
manifests. Native rung evidence remains bound to existing promotion records.
Strict profile and positive-residency validation have not been relaxed.

`research/runtime-catalog-proposals.json` preserves the displaced catalog,
resolutions, and requirements from `f4e439d` as a research-only archive. Pending
Tiel Coder proposals, zero-hash Unleashed/Apple proposals, wrong-topology reuse,
and retired Bonsai rungs in that archive are not production runtime authority.
The existing `research/candidates.json` queue is unchanged.

## Remaining independent release blockers

Repairing the adaptive catalog does not certify the whole release. In particular,
`references/model-catalog.json` still contains `qwen3-8-flash-next-strata` with a
non-Hugging-Face source, rejected by the strict model catalog loader. Several
other tests already fail at `f4e439d` (engine registry, matrix/documentation,
plugin interfaces, residency, and platform fixtures). The native installer
Visual Studio fixture mutates global `os.name` and crashes pytest on Linux.
Do not suppress these failures or claim a successful release gate.

No installed runtime configuration or model process is changed by this repair.
