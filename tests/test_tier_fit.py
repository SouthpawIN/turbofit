from __future__ import annotations

from turbofit_runtime.tier_fit import (
    MIN_CONTEXT,
    fits,
    kv_bytes,
    tier_budget_bytes,
    weight_bytes_for_family,
)

GIB = 1024 ** 3


def test_sharded_family_is_summed_once_even_when_every_shard_is_listed() -> None:
    artifacts = [
        {"repo_id": "org/model-gguf", "path": "Q4/model-00001-of-00002.gguf", "size_bytes": 10},
        {"repo_id": "org/model-gguf", "path": "Q4/model-00002-of-00002.gguf", "size_bytes": 90},
    ]
    total, complete = weight_bytes_for_family(artifacts, {"org/model-gguf": {}})

    assert total == 100
    assert complete is True


def test_missing_shard_is_completed_from_the_pinned_listing() -> None:
    artifacts = [
        {"repo_id": "org/model-gguf", "path": "Q4/model-00001-of-00003.gguf", "size_bytes": 10},
    ]
    listing = {"org/model-gguf": {
        "Q4/model-00001-of-00003.gguf": 10,
        "Q4/model-00002-of-00003.gguf": 20,
        "Q4/model-00003-of-00003.gguf": 30,
    }}

    assert weight_bytes_for_family(artifacts, listing) == (60, True)


def test_missing_shard_without_a_listing_is_reported_incomplete() -> None:
    artifacts = [
        {"repo_id": "org/model-gguf", "path": "Q4/model-00001-of-00002.gguf", "size_bytes": 10},
    ]

    total, complete = weight_bytes_for_family(artifacts, {})
    assert total == 10
    assert complete is False


def test_fit_floor_is_64k_and_never_32k() -> None:
    budget = tier_budget_bytes(8)
    weights = budget - kv_bytes(MIN_CONTEXT) - 1

    assert fits(weights, MIN_CONTEXT, 8)
    assert not fits(0, 32_768, 384)


def test_tier_budget_carries_the_documented_reporting_tolerance() -> None:
    assert tier_budget_bytes(8) == 8 * GIB + 16 * 1024 * 1024
    assert tier_budget_bytes(64) > 64 * GIB
