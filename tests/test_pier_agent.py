from __future__ import annotations

import pytest

from turbofit_runtime.pier_limits import agent_step_limit_flag


def test_pier_agent_step_limit_is_emitted_into_mini_swe_config() -> None:
    assert agent_step_limit_flag(16) == "-c agent.step_limit=16 "


def test_pier_agent_step_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        agent_step_limit_flag(0)
