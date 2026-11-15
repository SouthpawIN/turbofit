"""Request normalisation for the Turbofit gateway (TF3 + TF4).

TF3 — backend-aware finite reasoning budget normalisation. The semantics are
ported from the validated ``turbohaul-nostream-relay`` repair: an explicit
finite budget (clamped to the backend hard cap) wins, then reasoning-disabled
controls map to a zero budget, then a reasoning effort maps to a finite
budget, and anything else receives the backend's finite default. Policies are
opt-in per backend: without one the payload is forwarded untouched, so no
single model's cap is applied globally.

TF4 — streaming usage request default. Streaming requests without a caller
preference get ``stream_options.include_usage=true`` so the client's context
meter still receives the final usage frame; an explicit ``false`` survives and
non-stream requests are untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def normalise_reasoning_budget(payload: Dict[str, Any], policy: Optional[dict]) -> Dict[str, Any]:
    """Return a copy of ``payload`` with a finite ``thinking_budget_tokens``.

    ``policy`` comes from the resolved backend (recipe/request metadata) and
    may define ``hard_cap`` and ``default_budget``. ``None``/empty policy
    means the backend does not participate: the payload passes through
    unchanged rather than inheriting another model's cap.
    """
    if not policy:
        return dict(payload)

    hard_cap = policy.get("hard_cap")
    if isinstance(hard_cap, bool) or not isinstance(hard_cap, int) or not 0 < hard_cap < 2147483647:
        raise ValueError("reasoning policy requires a positive finite integer hard_cap")
    default_budget = _positive_int(policy.get("default_budget"))
    if default_budget is None:
        default_budget = 1024
    if hard_cap:
        default_budget = min(default_budget, hard_cap)

    out = dict(payload)

    # 1. Explicit budget wins, clamped to the backend cap.
    for key in ("thinking_budget_tokens", "reasoning_budget"):
        if key in payload:
            explicit = _positive_int(payload.get(key))
            if explicit is not None:
                out["thinking_budget_tokens"] = (
                    min(explicit, hard_cap) if hard_cap else explicit
                )
                return out

    reasoning = payload.get("reasoning")
    # 2. Disabled reasoning maps to a zero budget.
    if payload.get("think") is False or (
        isinstance(reasoning, dict) and reasoning.get("enabled") is False
    ):
        out["thinking_budget_tokens"] = 0
        return out

    # 3. Effort levels map to finite budgets.
    effort = None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is None:
        effort = payload.get("reasoning_effort")
    effort_name = str(effort or "").strip().lower()
    effort_budgets = {
        "none": 0,
        "minimal": 0,
        "low": 512,
        "medium": 1024,
        "high": 2048,
        "xhigh": 4096,
    }
    budget = effort_budgets.get(effort_name, default_budget)
    if hard_cap:
        budget = min(budget, hard_cap)
    out["thinking_budget_tokens"] = budget
    return out


def caller_controls_reasoning(payload: Dict[str, Any]) -> bool:
    """True when the caller hard-controlled reasoning at the template layer.

    Gateway defaults must never squash these controls. Mere effort metadata
    (``reasoning.effort`` / ``reasoning_effort``) is not a hard control: TF3
    normalises it into a finite budget on policy-covered backends.
    """
    if "think" in payload or "thinking_budget_tokens" in payload:
        return True
    if "reasoning_budget" in payload:
        return True
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and (
        "enable_thinking" in kwargs or "thinking_mode" in kwargs
    ):
        return True
    return False


def apply_stream_usage_default(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy defaulting ``stream_options.include_usage`` for streams."""
    out = dict(payload)
    if out.get("stream") is not True:
        return out
    options = out.get("stream_options")
    if isinstance(options, dict):
        options = dict(options)
        options.setdefault("include_usage", True)
    else:
        options = {"include_usage": True}
    out["stream_options"] = options
    return out