"""A crashed endpoint is not a model that scored zero.

The first published Hermes Index record was 0.0 — and the trial behind it had
crashed with litellm's ServiceUnavailableError after llama.cpp answered HTTP 503
to an over-context request. The verifier still returned a reward, so the run was
scored as a genuine failure. That measures our plumbing, not the model, and it
is exactly the kind of number that must never reach a leaderboard.
"""
from __future__ import annotations

import pytest

from turbofit_runtime.hermes_suites import _endpoint_failure


@pytest.mark.parametrize("text", [
    "ServiceUnavailableError: openai - request (67738 tokens) exceeds the available context size (65536)",
    {"exception_message": "litellm.APIConnectionError: Connection refused"},
    "InternalServerError: no local backend available",
    "TimeoutError: timed out",
    "RateLimitError: 429",
    "HTTP 503 from the gateway",
])
def test_transport_failures_are_named(text) -> None:
    assert _endpoint_failure(text) is not None


@pytest.mark.parametrize("text", [
    None,
    "",
    "AssertionError: expected output.txt to contain 42",
    {"exception_message": "the agent wrote the wrong file"},
])
def test_model_failures_are_not_excused(text) -> None:
    assert _endpoint_failure(text) is None


def test_a_genuine_zero_has_no_exception_at_all() -> None:
    # The real signal: a task the model attempted and failed raises nothing.
    assert _endpoint_failure(None) is None
