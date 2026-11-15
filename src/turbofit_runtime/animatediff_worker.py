"""Local AnimateDiff text-to-video worker (SD1.5 base + motion adapter).

Run through :mod:`turbofit_runtime.media_pipeline`; it owns validation and the
actionable "engine missing / weights missing" errors.  This module only does
GPU work, so torch is imported inside :func:`main` and never at import time.

    python -m turbofit_runtime.animatediff_worker \
        --model-root ~/Models/storage/diffusers/meinamix-v11 \
        --motion-adapter ~/Models/storage/diffusers/animatediff-motion-adapter-v1-5-3 \
        --prompt "..." --output clip.mp4 --frames 16 --width 640 --height 384
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DEFAULT_NEGATIVE = "text, watermark, deformed, extra limbs, photorealistic, blurry, lowres, static"


def _variant(base: Path) -> str | None:
    fp16 = base / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    return "fp16" if fp16.is_file() else None


def _load_adapter(path: Path, dtype, device: str):
    from diffusers import MotionAdapter
    from safetensors.torch import load_file

    if path.is_file():
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(str(path), device=device))
        return adapter
    for name in ("diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.safetensors"):
        if (path / name).is_file() or (path / "config.json").is_file():
            variant = "fp16" if (path / name).name.endswith("fp16.safetensors") else None
            return MotionAdapter.from_pretrained(str(path), torch_dtype=dtype, variant=variant)
    raise FileNotFoundError(
        f"motion adapter {path} is neither a .safetensors file nor a diffusers folder "
        "with config.json"
    )


def build_pipeline(model_root: Path, motion_adapter: Path, device: str):
    import torch
    from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler

    dtype = torch.float16
    adapter = _load_adapter(motion_adapter, dtype, device)
    pipe = AnimateDiffPipeline.from_pretrained(
        str(model_root),
        motion_adapter=adapter,
        torch_dtype=dtype,
        variant=_variant(model_root),
        requires_safety_checker=False,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    return pipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--motion-adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.frames > 16:
        print(
            f"warning: {args.frames} frames spans {args.frames // 16} motion contexts; "
            "quality degrades past one context, prefer multiple shots",
            flush=True,
        )

    import torch
    from diffusers.utils import export_to_video

    torch.manual_seed(args.seed)
    started = time.monotonic()
    pipe = build_pipeline(args.model_root, args.motion_adapter, args.device)
    load_seconds = round(time.monotonic() - started, 3)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    generated = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    )
    generation_seconds = round(time.monotonic() - started - load_seconds, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(generated.frames[0], str(args.output), fps=args.fps)

    peak_vram = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    payload = {
        "schema": "turbofit.local-video-generation/v1",
        "output": str(args.output),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "model_root": str(args.model_root),
        "motion_adapter": str(args.motion_adapter),
        "frames": len(generated.frames[0]),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_vram_mb": peak_vram,
        "bytes": args.output.stat().st_size,
    }
    evidence_path = args.output.with_suffix(".json")
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if os.environ.get("TURBOFIT_VIDEO_QUIET") != "1":
        print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
