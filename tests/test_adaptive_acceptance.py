from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/adaptive-acceptance"
LOADER = SourceFileLoader("adaptive_acceptance", str(SCRIPT))
SPEC = importlib.util.spec_from_loader("adaptive_acceptance", LOADER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_success_statuses_return_zero() -> None:
    assert MODULE.exit_code_for_status("simulated-pass") == 0
    assert MODULE.exit_code_for_status("real-pass") == 0


def test_non_success_status_returns_failure() -> None:
    assert MODULE.exit_code_for_status("blocked") == 2
    assert MODULE.exit_code_for_status("failed") == 2
