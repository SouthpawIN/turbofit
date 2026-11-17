"""The context floor: 65,536 tokens (64K). 32K is not a floor.

A model that can only hold 32K is not an admissible recommendation at any tier.
The floor is a named constant so tier placement, the recommendation path and the
fit ladder all refuse the same number for the same reason instead of each
carrying its own idea of "enough context".
"""
from __future__ import annotations

MIN_CONTEXT_TOKENS = 65_536


def meets_context_floor(context: int | None) -> bool:
    """True when ``context`` is at or above the 64K floor."""
    return context is not None and int(context) >= MIN_CONTEXT_TOKENS


def floor_refusal(context: int | None, *, subject: str = "configuration") -> str:
    """The sentence a refusal carries when a configuration is under the floor."""
    if context is None:
        return (
            f"{subject} declares no context length; the {MIN_CONTEXT_TOKENS} token "
            "(64K) floor is a hard gate"
        )
    return (
        f"{subject} holds only {int(context):,} tokens, below the "
        f"{MIN_CONTEXT_TOKENS:,} token (64K) floor"
    )
