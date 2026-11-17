"""What this host's CPU can actually execute.

Strata's documented CPU requirement is **AVX2**, not AVX-512:

* ``docs/AI_SETUP.md`` (Niko1221/Strata): "CPU: x86-64 with AVX2. Without AVX2
  (Xeon E5 v1/v2 and older) setup still installs, as an experimental and slow
  build".
* ``docs/AMD_HIP.md`` reports a measured configuration on "a Ryzen 9 3900X (16
  threads, no AVX-512)".

So AVX2 is the gate. AVX-512 is an *accelerator* (faster prefill), never an
exclusion: treating a missing AVX-512 flag as a deferral hid Strata from every
AVX2-only desktop, which is most of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

#: The gate. Without this flag Strata's kernel is not what its docs support.
STRATA_REQUIRED = ("avx2",)

#: Optional ISA that makes Strata faster (prefill), never a requirement.
STRATA_ACCELERATOR = ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi")


def cpu_flags(cpuinfo: Path | str | None = None) -> frozenset[str] | None:
    """Lowercase CPU feature flags, or None when the host exposes no /proc."""
    path = Path(cpuinfo) if cpuinfo is not None else Path("/proc/cpuinfo")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if line.lower().startswith("flags"):
            _, _, rest = line.partition(":")
            return frozenset(rest.split())
    return None


def missing_flags(required: Iterable[str], flags: frozenset[str] | None) -> tuple[str, ...]:
    if flags is None:
        return ()
    return tuple(sorted(flag for flag in required if flag not in flags))


def strata_supported(flags: frozenset[str] | None) -> bool | None:
    """True/False on Linux, None where the CPU cannot be interrogated."""
    if flags is None:
        return None
    return not missing_flags(STRATA_REQUIRED, flags)


def strata_accelerated(flags: frozenset[str] | None) -> bool | None:
    """True when AVX-512 is present (faster prefill); None when unknown."""
    if flags is None:
        return None
    return not missing_flags(STRATA_ACCELERATOR, flags)
