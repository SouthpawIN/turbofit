# Local video promo (verified)

First clip Turbofit has actually produced on its own hardware. Two 16-frame
AnimateDiff shots rendered locally on one RTX 3090 and assembled with ffmpeg.

| Item | Value |
|---|---|
| Runtime | `animatediff` (`turbofit_runtime.animatediff_worker`) |
| Base | `Meina/MeinaMix_V11` @ `2db85a2712983313dfe3e251f81e0023d775ee0a` |
| Motion adapter | `guoyww/animatediff-motion-adapter-v1-5-3` @ `2e8139b1d1269fd8a21deb96ad19455e187692eb` |
| Prompt manifest | [`prompts.json`](prompts.json) |
| Output | [`turbofit-garage-promo.mp4`](turbofit-garage-promo.mp4) (3.5 s, 640x384, h264, 28 frames @ 8 fps, 618 KiB) |
| Evidence | [`generation-evidence.json`](generation-evidence.json), [`ffprobe.json`](ffprobe.json), per-shot sidecars |

## Command

```bash
PYTHONPATH=src /root/tfenv-media/bin/python scripts/generate-local-video \
  --manifest promo/local-video/prompts.json \
  --output-dir /root/turbofit-promo/local-video \
  --python /root/tfenv-media/bin/python
```

Engine built with `uv venv --python 3.12` plus `torch==2.6.0+cu124`,
`diffusers==0.43.0`, `transformers==5.23.0`. Weights (~7 GB) fetched with
`hf download` at the pinned revisions above.

## Measured on omarchy (1x RTX 3090 24 GB)

| Metric | shot 1 | shot 2 |
|---|---|---|
| wall clock | 56.8 s | 51.6 s |
| pipeline load | 17.2 s | 18.5 s |
| generation (25 steps, 16 frames) | 24.2 s | 26.9 s |
| peak VRAM | 5.1 GB | 5.6 GB |

Total run: 1 min 50 s. `ffprobe`: h264 High, 640x384, yuv420p, 8/1 fps,
28 frames, 3.500 s, 632895 bytes, 1.45 Mb/s.

## What this proves, and what it does not

- Proves: Turbofit's declared runtime now maps to a real invocation, produces a
  playable mp4 from a text prompt on local weights, and reports its own status.
- Does not prove: 90 fps-style smoothness (native output is 8 fps limited
  animation), or that a 32-frame single generation works. Past one 16-frame
  motion context the model degrades to mush, so longer pieces must be built as
  separate shots — that is why this promo is two shots with an xfade, not one
  four-second generation.
