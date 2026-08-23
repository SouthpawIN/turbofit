from types import SimpleNamespace

import torch

from turbofit_runtime.h3_runtime import align_vae_decode, h3_launch_recipe, place_h3_vaes


class FakeVae(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.seen = None

    def decode(self, latents, *args, **kwargs):
        self.seen = (latents.device.type, str(latents.dtype))
        if latents.dtype != self.bias.dtype:
            raise RuntimeError(f"Input type ({latents.dtype}) and bias type ({self.bias.dtype}) should be the same")
        return latents + self.bias


def test_align_vae_decode_casts_half_latents_onto_float32_weights() -> None:
    vae = FakeVae()
    align_vae_decode(vae)
    out = vae.decode(torch.ones(2, dtype=torch.float16))
    assert vae.seen == ("cpu", "torch.float32")
    assert out.dtype == torch.float32


def test_place_h3_vaes_keeps_float32_recipe() -> None:
    pipe = SimpleNamespace(vae=FakeVae(), audio_vae=FakeVae())
    place_h3_vaes(pipe, "cpu")
    assert pipe.vae.bias.dtype == torch.float32
    pipe.vae.decode(torch.ones(2, dtype=torch.float16))
    assert pipe.vae.seen[1] == "torch.float32"


def test_h3_launch_recipe_is_the_portable_verify_command() -> None:
    recipe = h3_launch_recipe(device="cuda:1")
    assert recipe["revision"] == "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
    assert recipe["vae_dtype"] == "float32"
    assert recipe["verify"] == ["scripts/verify-h3-live", "--smoke", "--device", "cuda:1"]
