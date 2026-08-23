from turbofit_runtime.list_promotion import retain_or_replace_winner, select_tier_winner


def test_promotion_preserves_committed_winner_when_local_campaign_state_is_absent() -> None:
    existing = {
        "configuration": "a",
        "evidence": "sha256:" + "a" * 64,
        "hardware_fingerprint": "fp-a",
    }

    assert retain_or_replace_winner(existing, None, ["a", "b"]) == existing


def test_promotion_replaces_existing_winner_only_with_new_eligible_evidence() -> None:
    existing = {"configuration": "a", "evidence": "sha256:" + "a" * 64, "hardware_fingerprint": "fp-a"}
    selected = {"configuration": "b", "evidence": "sha256:" + "b" * 64, "hardware_fingerprint": "fp-b"}

    assert retain_or_replace_winner(existing, selected, ["a", "b"]) == selected


def test_list_winner_requires_physical_and_same_tier_intelligence() -> None:
    candidates = ["a", "b"]
    physical = {
        "a": {"status": "success", "current_recipe": True, "evidence": "sha256:" + "a" * 64, "hardware_fingerprint": "fp-a"},
        "b": {"status": "success", "current_recipe": True, "evidence": "sha256:" + "b" * 64, "hardware_fingerprint": "fp-b"},
    }
    intelligence = {
        "a": {"hardware_tier_gb": 48, "intelligence_score": 70, "throughput_tps": 20, "balanced_score": 31.11},
        "b": {"hardware_tier_gb": 8, "intelligence_score": 99, "throughput_tps": 100, "balanced_score": 99.5},
    }

    winner = select_tier_winner(candidates, physical, intelligence, hardware_tier_gb=48)

    assert winner == {
        "configuration": "a",
        "evidence": "sha256:" + "a" * 64,
        "hardware_fingerprint": "fp-a",
    }


def test_list_winner_uses_highest_balanced_real_score() -> None:
    physical = {
        name: {"status": "success", "current_recipe": True, "evidence": "sha256:" + char * 64, "hardware_fingerprint": f"fp-{name}"}
        for name, char in (("a", "a"), ("b", "b"))
    }
    intelligence = {
        "a": {"hardware_tier_gb": 48, "intelligence_score": 80, "throughput_tps": 10, "balanced_score": 32},
        "b": {"hardware_tier_gb": 48, "intelligence_score": 65, "throughput_tps": 35, "balanced_score": 67},
    }

    winner = select_tier_winner(["a", "b"], physical, intelligence, hardware_tier_gb=48)

    assert winner["configuration"] == "b"
