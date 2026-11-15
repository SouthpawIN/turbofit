# Multimodal model manager

## Multimodal model manager

Turbofit scans total usable memory and platform support, then labels each option as ready, candidate, unsupported, or too large. Candidate integrations are never marked ready until their adapter exists.

| Modality | Managed options |
|---|---|
| Image | Hermes configured image-generation provider |
| Video | Hermes configured video provider; AnimateDiff SD1.5 motion adapter (verified locally); MiniMax H3 research candidate |
| Music | Hermes music generation; MiniMax Music 3; ACE-Step 1.5 2B and 4B local candidates |
| Speech-to-text | Hermes local transcription; Parakeet TDT 0.6B v3; Nemotron 3.5 ASR 0.6B |
| Text-to-speech | Edge TTS; Soprano TTS; Darwin TTS 1.7B Cross |

### New multimodal candidates in 2.3

- **MiniMax Music 3** is pinned to the official [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) commit `fbdf52fbaaca799592917417eb05f1899f1255ec`. Its model card describes complete songs up to five minutes, an 8B global LLM plus 0.6B local LLM, 32 kHz 16-bit stereo output, a full-precision route under 24 GB VRAM, and streamed CPU offload down to 8 GB VRAM.
- **NVIDIA Parakeet TDT 0.6B v3** is pinned to [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) commit `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` as the new local speech-to-text candidate.
- **Soprano TTS** is pinned to [`MurrayMacdonald/soprano-tts`](https://huggingface.co/MurrayMacdonald/soprano-tts) commit `55651f7b114c8c2a4d98d612e9f13bfa3b6a8123` as the new local text-to-speech candidate.

All three remain explicitly labeled `candidate` until their Turbofit adapter and physical generation/transcription receipt pass. A pinned entry is reproducible metadata, not a false claim of completed integration.

![Turbofit 2.3 multimodal manager: MiniMax Music 3, Parakeet TDT 0.6B v3, Soprano TTS, and local MiniMax H3 video](assets/turbofit-2.3-multimodal.png)

The MiniMax H3 repository is a roughly 498 GB BF16 audio/video model release, not a small image model. Its official full-precision workflow recommends four GPUs, and Turbofit carries a streamed host-offload script at `scripts/generate-h3-promo-clips` for the 24 GB class. **That route has not produced a clip on the project hardware**: the weights are not materialized, no `generation-evidence.json` exists under `promo/h3-local*`, and the catalog therefore lists `minimax-h3` as a `candidate` with the `install-adapter` action. It moves to `run-turbofit-local` only after a receipt lands.

Catalog and pinned revisions: [`references/multimodal-models.json`](references/multimodal-models.json).

### Local video runtime (verified)

`animatediff` is the first video runtime with a real receipt. It is an SD1.5 text-to-image base plus a motion adapter, 16 frames per motion context, assembled to mp4 with ffmpeg. `turbofit_runtime.media_pipeline` owns the registry, the readiness probe and the invocation plan; `turbofit_runtime.animatediff_worker` is the only module that imports torch, and it runs as a subprocess so the probe stays importable on a CPU-only host.

```bash
scripts/generate-local-video --manifest promo/local-video/prompts.json --output-dir promo/local-video/out
```

Readiness is reported in the same vocabulary as the rest of the runtime code: `waiting-runtime` (engine module missing, with the `uv pip install` line), `waiting-download` (weights missing, with the pinned `hf download` line), `waiting-config` (a builtin Hermes provider section that is not set), `waiting-memory` (accelerator too small), `ready`. Pass the probe results to `recommend_multimodal(..., runtimes=...)` and an entry can no longer claim `use-now` while its engine is absent — it reports `install-adapter` with the fix command instead.

Measured on omarchy (1x RTX 3090): 640x384, 16 frames, 25 steps, 24-27 s of generation per shot, 5.1-5.6 GB peak VRAM, 1 min 50 s for a two-shot run. Native output is 8 fps, and more than 16 frames in a single call degrades badly — build longer pieces from multiple shots. Evidence: [`promo/local-video/`](promo/local-video/README.md).

---
