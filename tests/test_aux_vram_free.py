"""The auxiliary keeps VRAM free for the main model.

40-85% of Hermes tokens go through the auxiliary, so it has to be cheap: its
weights stay memory-mapped on disk and the active experts stream into host RAM
(or straight off the SSD below 32 GB). That only pays off if the auxiliary does
not also claim a card, so the composed command runs with ``-ngl 0``.

An explicit ``gpu_layers`` override still wins — a user who wants the auxiliary
on the GPU can ask for it, and the ladder can promote it when VRAM is idle.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.recipes import RecipeBook


def _book(*, ram_mb: int, overrides: dict | None = None) -> RecipeBook:
    entry: dict = {
        "auto_cpu_moe": True,
        "methods": {"65536": "baseline", "8192": "baseline"},
    }
    if overrides:
        entry["context_overrides"] = overrides
    data = {
        "schema_version": 1,
        "atomic_binary": "/bin/true",
        "models": {
            "Test Aux": {
                "alias": "test-aux",
                "kind": "process",
                "model": "aux.gguf",
                "port": 11605,
                **entry,
            }
        },
        "variants": {},
    }
    fingerprint = HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=ram_mb,
        devices=(
            AcceleratorDevice(
                index=0, uuid="GPU-0", name="Test", vendor="nvidia", backend="cuda",
                memory_total_mb=24576, compute_capability="8.6", bus_id=None,
            ),
        ),
    )
    return RecipeBook(
        data, backend_name="cuda", platform_name="linux", hardware=fingerprint,
    )


def _aux(book: RecipeBook, *, context: int = 65_536):
    return book.resolve_component(
        "Test Aux", role="aux", gpu="0", port=11605, context=context,
    )


def _ngl(command: tuple[str, ...] | list[str]) -> str:
    for flag in ("-ngl", "--n-gpu-layers", "--gpu-layers"):
        if flag in command:
            return str(command[list(command).index(flag) + 1])
    raise AssertionError(f"no gpu-layer flag in {command}")


def test_aux_streams_from_ram_and_claims_no_vram() -> None:
    aux = _aux(_book(ram_mb=131_072))
    assert _ngl(aux.command) == "0"
    assert "--no-mmap" not in aux.command


def test_aux_on_a_small_host_still_claims_no_vram() -> None:
    aux = _aux(_book(ram_mb=16_384))
    assert _ngl(aux.command) == "0"
    assert "--no-mmap" not in aux.command


def test_explicit_gpu_layers_override_is_honoured() -> None:
    book = _book(ram_mb=131_072, overrides={"65536": {"gpu_layers": 12}})
    aux = _aux(book)
    assert _ngl(aux.command) == "12"


def test_main_role_is_not_forced_off_the_gpu() -> None:
    book = _book(ram_mb=131_072)
    main = book.resolve_component(
        "Test Aux", role="main", gpu="0", port=11606, context=65_536,
    )
    assert _ngl(main.command) != "0"
