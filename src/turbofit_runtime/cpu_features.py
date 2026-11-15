"""What this host's CPU can actually execute.

Colibri/Strata needs AVX-512 VNNI and VBMI for its expert kernel and refuses to
decode without them (its scalar fallback is test-only). Discovering that from a
failed load is a bad way to learn it, so the requirement is probed up front.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Colibri is an Intel Ice Lake / AMD Zen 4 class requirement.
STRATA_REQUIRED = ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi")


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
