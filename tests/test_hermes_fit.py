"""Hermetic tests for the built-in-recommendation bridge (hermes_fit).

No real GPU probe: we monkeypatch the built-in ``probe_budget`` with synthetic
``HardwareBudget`` objects and run the REAL built-in catalog + estimator
(deterministic) so the physics is genuine. The allowlist / ban overlay is
exercised via ``monkeypatch`` on ``allowed_lineup``.
"""
from __future__ import annotations

import types

import turbofit_runtime.allowed_lineup as allowed_lineup
import turbofit_runtime.hermes_fit as hermes_fit
from hermes_cli.local_runtime import catalog as real_catalog
from hermes_cli.local_runtime import estimator as real_estimator
from hermes_cli.local_runtime import hardware as real_hardware

_GIB = 1024**3


def _budget(usable_gb, total_gb, host_gb, *, uma, gpu_name, platform, pci):
    return real_hardware.HardwareBudget(
        usable_vram_bytes=usable_gb * _GIB,
        total_device_bytes=total_gb * _GIB,
        ram_available_bytes=host_gb * _GIB,
        uma=uma,
        gpu_name=gpu_name,
        platform=platform,
        gpu_pci_id=pci,
        lazy_reads=True,
        commit_room_bytes=4 * _GIB,
    )


APPLE_32 = _budget(8, 32, 0, uma=True, gpu_name="Apple M1", platform="darwin", pci=None)
NVIDIA_48 = _budget(48, 48, 64, uma=False, gpu_name="NVIDIA RTX 3090", platform="linux", pci=0x1E04)


def _patch_probe(monkeypatch, budget):
    fake_hardware = types.SimpleNamespace(probe_budget=lambda: budget)
    monkeypatch.setattr(
        hermes_fit, "_import_hermes", lambda: (fake_hardware, real_catalog, real_estimator)
    )


def test_apple_uma_compensated(monkeypatch):
    """A 22 GB MoE must fit on a 32 GB unified box once the pool is credited."""
    _patch_probe(monkeypatch, APPLE_32)
    rec = hermes_fit.build_recommendation()
    assert rec.uma is True
    assert rec.ladder == "unified/moe"
    assert rec.compensated is True
    assert rec.pool_kind == "unified"
    assert rec.backend == "metal"
    assert rec.primary is not None
    # The built-in pick is the fast MoE, which only fits after UMA compensation.
    assert rec.primary.model == "qwen3.6-35b-a3b"
    assert rec.primary.banned is False
    # Flash Next (q95) is always surfaced as the stretch / convergence target.
    assert rec.stretch is not None
    assert rec.stretch["model"] == "qwen3.8-flash-next"
    assert rec.stretch["fits"] is False
    assert rec.suggested_profile == "hardware-32gb"


def test_nvidia_dedicated_dense_ladder(monkeypatch):
    _patch_probe(monkeypatch, NVIDIA_48)
    rec = hermes_fit.build_recommendation()
    assert rec.uma is False
    assert rec.ladder == "dedicated/dense"
    assert rec.compensated is False
    assert rec.pool_kind == "dedicated"
    assert rec.backend == "cuda"
    assert rec.primary is not None
    # 48 GB dedicated: the built-in's own pick is the best-quality resident.
    assert rec.primary.model == "qwen3.8-27b"


def test_ban_filter_falls_back_to_turbofit_canonical(monkeypatch):
    """If every built-in fit is banned, primary falls to TurboFit's own pick."""
    _patch_probe(monkeypatch, APPLE_32)
    monkeypatch.setattr(allowed_lineup, "is_banned", lambda alias: True)
    rec = hermes_fit.build_recommendation()
    canon = rec.turbofit_canonical
    assert canon is not None
    assert rec.primary is not None
    # The fallback primary is the allowlist canonical main (not a banned id).
    assert rec.primary.banned is False
    assert rec.primary.model == canon["main"]


def test_stretch_reported_when_no_flash_next_in_catalog(monkeypatch):
    # With the real catalog, flash-next is present and refused on 32 GB.
    _patch_probe(monkeypatch, APPLE_32)
    rec = hermes_fit.build_recommendation()
    stretch_ids = [r["model"] for r in rec.refused]
    assert "qwen3.8-flash-next" in stretch_ids
    assert rec.stretch["fits"] is False
    assert rec.stretch["required_gb"] > 0


def test_fallback_without_hermes(monkeypatch):
    """Standalone host (no Hermes): degrade to allowlist + ladders, never crash."""
    monkeypatch.setattr(hermes_fit, "available", lambda: False)
    fp = types.SimpleNamespace(
        total_usable_memory_mb=48 * 1024,
        system_ram_mb=64 * 1024,
        memory_pool_kind="dedicated",
        backends={"cuda"},
    )
    rec = hermes_fit.build_recommendation(fingerprint=fp)
    assert rec.source == "turbofit-fallback"
    assert rec.ladder == "dedicated/dense"
    assert rec.primary is not None
    assert rec.engine is not None
    # Fallback must still be JSON-serializable end to end.
    import json

    json.dumps(rec.to_dict(), default=str)


def test_recommend_is_json_serializable(monkeypatch):
    _patch_probe(monkeypatch, APPLE_32)
    rec = hermes_fit.build_recommendation()
    import json

    text = json.dumps(rec.to_dict(), indent=2, default=str)
    assert "unified/moe" in text
    assert "qwen3.8-flash-next" in text
