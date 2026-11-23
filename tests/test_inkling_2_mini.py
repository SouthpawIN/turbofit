"""Inkling 2 Mini: a first-class Strata family at the 64K context floor.

The operator's rules, pinned:

* every admitted placement holds the hard 64K floor — a row that can only
  hold 32K is refused, never substituted;
* SQ-20 — the bit-exact *native* SF4+MX8 build — is placed on 24 GB cards,
  fully resident with room for KV, and is not offered below the card it is
  sized for;
* SQ-10 is admitted on small cards through hybrid RAM+VRAM expert tiering,
  and fits entirely in VRAM from 12 GB;
* the lossless pack is a native precision, not a lossy quant: the quant
  policy admits it first-class without a fabricated justification string.

Artifact facts are pinned to the published HF revision (pack card +
SHA256SUMS.txt at strata-engine/Inkling-2-Mini-Strata); the launch commands
are planning resolutions, not a claim of a live load test.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from turbofit_runtime import strata_tiers
from turbofit_runtime.context_floor import MIN_CONTEXT_TOKENS
from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.quant_policy import ALLOWED, classify_entry, default_admits
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.tier_fit import TIERS, tier_budget_bytes

ROOT = Path(__file__).parents[1]
GIB = 1024 ** 3
AVX2 = frozenset({"fpu", "avx", "avx2", "sse4_2"})
REVISION = "04996a1e2c45009b6a04f2af126cb6bdb2ba3587"
REPO = "strata-engine/Inkling-2-Mini-Strata"
PACK_IDS = ("inkling-2-mini-sq10", "inkling-2-mini-sq15", "inkling-2-mini-sq20")
# Real file sizes and LFS sha256 digests from the pinned HF revision,
# cross-checked against the repo's SHA256SUMS.txt.
PACK_BYTES = {
    "Inkling-2-Mini-SQ-10.strata": (
        10_471_526_400,
        "ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934",
    ),
    "Inkling-2-Mini-SQ-15.strata": (
        14_348_316_672,
        "2ea32eed533afba257e2cbee7b073dc2c3da80fe55d1d2d794779b5498e30737",
    ),
    "Inkling-2-Mini-SQ-20.strata": (
        19_346_685_952,
        "0bbfd52344f0dd4779a7807b5f73554a812f24c035e6877b63cd0d499f7e1b97",
    ),
    "Inkling-2-Mini-Vision.strata": (
        636_526_592,
        "6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677",
    ),
}


def inkling_rows() -> tuple[strata_tiers.StrataRow, ...]:
    return tuple(
        row for row in strata_tiers.load_rows() if row.family == "Inkling 2 Mini"
    )


def catalog_entries() -> dict:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    return {item.id: item for item in catalog.models if item.id in PACK_IDS}


def test_catalog_pins_three_packs_at_the_published_revision() -> None:
    entries = catalog_entries()
    assert set(entries) == set(PACK_IDS)
    for item in entries.values():
        assert item.family == "Inkling 2 Mini"
        assert item.source == f"https://huggingface.co/{REPO}"
        assert item.revision == REVISION
        assert item.upstream_source == "https://huggingface.co/thinkingmachines/Inkling-2-Mini"
        assert item.moe is True and item.vision is True
        assert item.roles == ("main",)
        assert MIN_CONTEXT_TOKENS in item.supported_contexts
        assert min(item.supported_contexts) >= MIN_CONTEXT_TOKENS
        assert item.artifact in PACK_BYTES


def test_manifest_pins_real_sizes_and_digests() -> None:
    manifest = json.loads((ROOT / "references/artifact-manifest.json").read_text())
    rows = {
        item["path"]: item
        for item in manifest["artifacts"]
        if item["repo_id"] == REPO
    }
    assert set(rows) == set(PACK_BYTES)
    for path, (size, sha) in PACK_BYTES.items():
        row = rows[path]
        assert row["revision"] == REVISION
        assert row["size_bytes"] == size
        assert row["sha256"] == sha
        assert row["destination"] == f"inkling-2-mini/{path}"
    # The vision sidecar serves every pack.
    assert set(rows["Inkling-2-Mini-Vision.strata"]["families"]) == set(PACK_IDS)


def test_footprints_are_the_pinned_pack_plus_vision_bytes() -> None:
    footprints = json.loads((ROOT / "references/model-footprints.json").read_text())["models"]
    vision = PACK_BYTES["Inkling-2-Mini-Vision.strata"][0]
    files = {
        "inkling-2-mini-sq10": "Inkling-2-Mini-SQ-10.strata",
        "inkling-2-mini-sq15": "Inkling-2-Mini-SQ-15.strata",
        "inkling-2-mini-sq20": "Inkling-2-Mini-SQ-20.strata",
    }
    for model_id, pack in files.items():
        record = footprints[model_id]
        assert record["weight_bytes"] == PACK_BYTES[pack][0] + vision
        assert record["repos"] == [REPO]
        assert record["moe"] is True


def test_every_admitted_placement_holds_the_64k_floor() -> None:
    rows = inkling_rows()
    assert len(rows) == 3
    for tier in TIERS:
        admitted, _refused = strata_tiers.place_tier(tier, ram_bytes=64 * GIB, rows=rows)
        for placement in admitted:
            assert placement.context >= MIN_CONTEXT_TOKENS
            assert placement.context in {65_536, 131_072, 262_144, 1_048_576}


def test_a_32k_curve_is_refused_not_substituted() -> None:
    row = replace(inkling_rows()[0], context_curve=((8, 32_768),))
    result = strata_tiers.place_row(row, tier_gb=24, ram_bytes=128 * GIB)
    assert isinstance(result, strata_tiers.StrataRefusal)
    assert "below the 65,536 token (64K) floor" in result.reason


def test_sq20_is_placed_on_24gb_cards_fully_resident() -> None:
    rows = inkling_rows()
    admitted, refused = strata_tiers.place_tier(24, ram_bytes=64 * GIB, rows=rows)
    assert admitted, "SQ-20 must be admitted at 24 GB"
    top = admitted[0]
    assert top.model_id == "inkling-2-mini-sq20"
    assert top.context >= MIN_CONTEXT_TOKENS
    # 18.02 GiB pack + 0.59 GiB vision + KV fits the 24 GB band: no RAM spill.
    assert top.ram_bytes == 0
    assert top.total_bytes <= tier_budget_bytes(24)
    assert not refused
    # Below the card it is sized for, SQ-20 is refused with the documented floor.
    admitted16, refused16 = strata_tiers.place_tier(16, ram_bytes=64 * GIB, rows=rows)
    assert all(item.model_id != "inkling-2-mini-sq20" for item in admitted16)
    sq20 = next(item for item in refused16 if item.model_id == "inkling-2-mini-sq20")
    assert "documented from 24 GB" in sq20.reason


def test_sq10_is_admitted_on_small_cards_via_hybrid_tiering() -> None:
    rows = inkling_rows()
    admitted, refused = strata_tiers.place_tier(8, ram_bytes=64 * GIB, rows=rows)
    assert [item.model_id for item in admitted] == ["inkling-2-mini-sq10"]
    top = admitted[0]
    assert top.context >= MIN_CONTEXT_TOKENS
    # Hybrid: the pack exceeds the card, so expert-tiered RAM carries the rest.
    assert top.vram_bytes <= tier_budget_bytes(8)
    assert top.ram_bytes > 0
    assert top.caveats and "runs slowly" in top.caveats[0]
    # The larger packs are not streamed onto the small card in its place.
    assert {item.model_id for item in refused} == {
        "inkling-2-mini-sq15", "inkling-2-mini-sq20",
    }


def test_sq10_fits_entirely_in_vram_from_12gb() -> None:
    rows = inkling_rows()
    admitted, _refused = strata_tiers.place_tier(12, ram_bytes=64 * GIB, rows=rows)
    sq10 = next(item for item in admitted if item.model_id == "inkling-2-mini-sq10")
    assert sq10.ram_bytes == 0
    assert sq10.context >= MIN_CONTEXT_TOKENS


def test_sq15_leads_the_16gb_band() -> None:
    rows = inkling_rows()
    admitted, _refused = strata_tiers.place_tier(16, ram_bytes=64 * GIB, rows=rows)
    assert [item.model_id for item in admitted] == [
        "inkling-2-mini-sq15", "inkling-2-mini-sq10",
    ]
    assert admitted[0].ram_bytes == 0


def test_lossless_sq20_is_a_native_precision_not_a_lossy_quant() -> None:
    entry = catalog_entries()["inkling-2-mini-sq20"]
    assert "lossless" in entry.quantization.lower()
    assert "native" in entry.quantization.lower()
    # No justification string is invented: the entry is not above the ceiling.
    assert entry.quantization_justification is None
    verdict = classify_entry(entry)
    assert verdict.quant_class == ALLOWED
    assert verdict.token is None  # no GGUF quant token: a native build
    admitted, reason = default_admits(entry)
    assert admitted is True
    assert "native" in reason
    # The low packs are first-class too — never ranked below a wider quant.
    for model_id in ("inkling-2-mini-sq10", "inkling-2-mini-sq15"):
        low = catalog_entries()[model_id]
        assert low.quantization_justification is None
        assert classify_entry(low).quant_class == ALLOWED
        assert default_admits(low)[0] is True


def test_strata_entry_documents_the_shape_engine_and_fidelity() -> None:
    recipes = json.loads((ROOT / "references/model-recipes.json").read_text())
    documented = recipes["strata"]["models"]["inkling-2-mini"]
    assert documented["shape_engine"] == "strata-inkling"
    assert documented["engine_version"] == "0.1.65"
    packs = documented["packs"]
    assert packs["SQ-20"]["native_lossless"] is True
    assert packs["SQ-20"]["top1_agreement"] == 1.0
    assert packs["SQ-20"]["kl_divergence_test"] == 0.0
    assert packs["SQ-15"]["top1_agreement"] == 0.9668
    assert packs["SQ-15"]["kl_divergence_test"] == 0.00889
    assert packs["SQ-10"]["hybrid_from_vram_gib"] == 4
    assert documented["vision_pack"]["precision"] == "lossless BF16"
    # The context curve is labelled derived, not vendor-documented.
    assert "derived" in documented["context_tokens_source"].lower()


def test_recipes_resolve_to_strata_serve_with_the_pinned_pack() -> None:
    book = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cuda")
    entries = catalog_entries()
    for model_id, entry in entries.items():
        for context in entry.supported_contexts:
            row = {
                "id": f"{model_id}--maple-preview-tq2--{context // 1024}k",
                "main": model_id,
                "auxiliary": "maple-preview-tq2",
                "context": context,
                "status": "candidate",
            }
            component = book.resolve_catalog_configuration(row).components[-1]
            command = list(component.command)
            assert command[0].endswith("strata")
            assert "--serve" in command
            pack = command[command.index("--pack") + 1]
            assert pack.endswith(entry.artifact)
            assert "${" not in pack  # every root variable was substituted
            assert command[command.index("--max-context") + 1] == str(context)
            vision = command[command.index("--mmproj") + 1]
            assert vision.endswith("Inkling-2-Mini-Vision.strata")
            # llama.cpp vocabulary must never reach the Strata engine.
            for flag in ("--cpu-moe", "--n-cpu-moe", "-ngl", "--jinja", "--fit"):
                assert flag not in command
            assert component.engine == "strata"


def test_avx2_host_admits_inkling_and_missing_avx2_names_the_flag() -> None:
    view = strata_tiers.tier_view(tier_gb=24, ram_bytes=64 * GIB, cpu_flags=AVX2)
    assert any(item.model_id == "inkling-2-mini-sq20" for item in view.admitted)
    deferred = strata_tiers.tier_view(
        tier_gb=24, ram_bytes=64 * GIB, cpu_flags=frozenset({"fpu", "avx"}),
    )
    assert deferred.admitted == ()
    assert "lacks avx2" in deferred.reason
