from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def module() -> dict:
    return runpy.run_path(str(ROOT / "scripts/turbofit-tier-campaign"), run_name="tier_campaign_test")


def test_tier_campaign_has_no_retiring_active_models() -> None:
    loaded = module()
    _, configurations, _, display_ids = loaded["planned"]()
    deferred = loaded["deferred_display_ids"](configurations, display_ids)

    assert deferred == frozenset()


def test_tier_campaign_refuses_to_label_other_topologies_as_physical() -> None:
    restrict = module()["restrict_to_physical_tier"]

    assert restrict(None, "hardware-48gb") == {"hardware-48gb"}
    assert restrict({"hardware-48gb"}, "hardware-48gb") == {"hardware-48gb"}
    with pytest.raises(ValueError, match="requires exact host topology hardware-48gb"):
        restrict({"hardware-8gb"}, "hardware-48gb")
