# External speed evidence — RTX PRO 6000 (96 GB), Basalt 0.5.0, Qwen3.8-Flash-Next IQ3_S

Discovered 2026-11-19. Single RTX PRO 6000 Blackwell (SM120), Basalt 0.5.0,
served as `qwen3.8-flash-next`. Evidence payload: `research/basalt-tuned-combo-rtx-pro-6000-20261119.json`.

## Tuned combo (spec 6 + relaxed-drafting 0.9 + reasoning-effort none)

| Metric | Value |
|---|---|
| C1 decode (median, 3 runs) | 424.9 tok/s |
| C8 aggregate (median, 3 runs) | 583.2 tok/s |
| Fresh 29K-token TTFT | 2767 ms |
| Cached-turn TTFT at 29K | 35.8 ms |
| Tool calls | 10/10 |
| Text loops | 0 of 16 |

Candidate ranking from the tuning sweep (see research/): combo > effort-none (424.6/567.7) > base (321.4/499.2) > spec6 (317.9/516.9). Spec 12 was the known cliff (201 tok/s).

## Verified external run

localmaxxing.com run `cmwohxfy20849mi018zbrlyqm` — Basalt 0.5.0, canonical reasoning-v1 prompt, verified: **473.0 tok/s** decode, 1614 tok/s prefill, 187 ms TTFT, 73.9% draft accept.

## Notes

- Basalt is a Strata fork (same engine lineage, tuned for Blackwell sm_120 + CUDA 13); `lmx` runs it with the `strata` engine flag.
- 262K context fits on one 96 GB card alongside a 1.9 GiB reserve; `--vram-elastic` shrinks VRAM to free GPU for image/video renders alongside the LLM (H3 clip rendered while Basalt at 1 GiB expert cache: 145 s, byte-identical output).
- Discovery evidence only; no profile promotion implied.
