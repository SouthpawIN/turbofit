"""Pinned MiniMax H3 local recipe.

This is the only supported way TurboFit launches H3 on someone else's
machine. Recommend without this recipe is a lie.
"""
from __future__ import annotations

from typing import Any

H3_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
H3_SOURCE = "https://huggingface.co/MiniMaxAI/MiniMax-H3"
H3_MINIMUM_ACCELERATOR_MB = 24576
H3_MINIMUM_HOST_RAM_MB = 98304


def align_vae_decode(module: Any) -> Any:
    """Decode latents must match the VAE parameter device and dtype.

    Observed failures on this host:
    - VAE left float32 / parked on CPU while decode emitted CUDA Half
    - VAE forced to float16 while remaining bias/conv weights stayed float32
    """
    original = module.decode

    def decode(latents, *args, **kwargs):
        target = next(module.parameters())
        aligned = latents.to(device=target.device, dtype=target.dtype)
        return original(aligned, *args, **kwargs)

    module.decode = decode
    return module


def place_h3_vaes(pipe: Any, device: str) -> Any:
    """Keep both VAEs float32 on the generation device, then align decode."""
    import torch

    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.audio_vae.to(device=device, dtype=torch.float32)
    align_vae_decode(pipe.vae)
    align_vae_decode(pipe.audio_vae)
    return pipe


def h3_launch_recipe(*, device: str = "cuda:0") -> dict[str, object]:
    return {
        "schema": "turbofit.h3-launch/v1",
        "source": H3_SOURCE,
        "revision": H3_REVISION,
        "workflow": "t2va",
        "device": device,
        "quantization": "torchao-int8-weight-only-v2",
        "transformer_offload": "block_level",
        "text_encoder_offload": "leaf_level",
        "vae_dtype": "float32",
        "vae_device": device,
        "verify": ["scripts/verify-h3-live", "--smoke", "--device", device],
        "minimum_accelerator_memory_mb": H3_MINIMUM_ACCELERATOR_MB,
        "minimum_memory_mb": H3_MINIMUM_HOST_RAM_MB,
    }
