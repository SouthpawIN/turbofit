"""Pure validation helpers for the PIER mini-swe adapter."""
from __future__ import annotations


def agent_step_limit_flag(step_limit: int) -> str:
    if isinstance(step_limit, bool) or not isinstance(step_limit, int) or step_limit <= 0:
        raise ValueError("agent step limit must be a positive integer")
    return f"-c agent.step_limit={step_limit} "
