"""A killed engine must not be reported as a bad model file.

Observed on omarchy: five screening rows failed with
``aux failed to load model maple-preview-tq2 (exit code -9)`` and
``main failed to load model arete-27b-q8 (exit code -9)``.  The weights were
complete and had loaded; the engines were SIGKILLed by an external
``pkill -9 -f llama-server`` (and by a concurrent campaign reclaiming the same
ports).  The message blamed the model, so the operator re-downloads 26.7 GiB
that is already correct.  These tests pin the signal-aware wording and the
route-mismatch diagnostic that names the substituted role.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbofit_runtime import artifacts  # noqa: E402


def test_sigkill_is_reported_as_external_not_a_load_failure(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "kernel_oom_killed", lambda pid, **_: False)

    message = artifacts.describe_load_failure(
        role="aux", model="maple-preview-tq2", file="/models/maple.gguf",
        log_path=None, exit_code=-9, pid=4242, log_tail_lines=["model loaded"],
    )

    assert "SIGKILL" in message and "signal 9" in message
    assert "4242" in message
    assert "not a model load failure" in message
    assert "failed to load model" not in message
    assert "pkill" in message and "concurrent campaign" in message
    assert "model loaded" in message


def test_sigkill_with_kernel_oom_evidence_names_the_oom_killer(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "kernel_oom_killed", lambda pid, **_: True)

    message = artifacts.describe_load_failure(
        role="main", model="arete-27b-q8", file="/models/arete.gguf",
        exit_code=-9, pid=99,
    )

    assert "OOM killer" in message
    assert "system RAM" in message


def test_abort_is_reported_as_a_cuda_or_assertion_abort(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "kernel_oom_killed", lambda pid, **_: False)

    message = artifacts.describe_load_failure(
        role="main", model="arete-27b-q8", file="/models/arete.gguf", exit_code=-6,
    )

    assert "SIGABRT" in message
    assert "out-of-memory" in message


def test_positive_exit_code_keeps_the_size_evidence_but_never_blames_a_kill() -> None:
    message = artifacts.describe_load_failure(
        role="aux", model="maple-preview-tq2", file=None, exit_code=1,
    )
    assert "exit code 1" in message
    assert "SIGKILL" not in message


@pytest.fixture(scope="module")
def campaign():
    loader = importlib.machinery.SourceFileLoader(
        "turbofit_intelligence_campaign_script",
        str(ROOT / "scripts/turbofit-intelligence-campaign"),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_route_mismatch_names_the_substituted_aux(campaign) -> None:
    recipe = campaign.ResolvedRecipe(
        row_id="arete-27b-q5--maple-preview-tq2--64k",
        profile_name="arete-27b-q5--maple-preview-tq2--64k",
        main_alias="arete-27b-q5",
        aux_alias="maple-preview-tq2",
        aux_mode="dedicated",
        components=(),
        main_context=65536,
        aux_context=65536,
    )
    route = {
        "main": "arete-27b-q5",
        "aux": "auto:arete-27b-q5",
        "status": {
            "main": {"alias": "arete-27b-q5", "state": "ready", "port": 11605},
            "aux": {
                "alias": "auto:arete-27b-q5", "state": "ready", "port": 11605,
                "mode": "shared-main",
            },
        },
    }

    message = campaign.route_mismatch_message(recipe, route)

    assert "aux: recipe declares 'maple-preview-tq2'" in message
    assert "shared-main fallback 'auto:arete-27b-q5'" in message
    assert "not answering on its port" in message
    assert "main:" not in message  # main matched, so it is not noise


def test_assert_alive_names_a_killed_engine() -> None:
    from turbofit_runtime.backend import CampaignBackend
    from turbofit_runtime.recipes import ResolvedComponent

    component = ResolvedComponent(
        role="aux", family="maple-preview-tq2", alias="maple-preview-tq2",
        kind="process", method="baseline", gpu="0", port=11610, command=("llama-server",),
    )
    handle = {"pid": 777, "log": None, "process": _DeadProcess(-9)}

    backend = CampaignBackend.__new__(CampaignBackend)
    with pytest.raises(RuntimeError) as error:
        backend.assert_alive(component, handle)

    assert "SIGKILL" in str(error.value)
    assert "already answered readiness" in str(error.value)


class _DeadProcess:
    def __init__(self, code: int) -> None:
        self._code = code

    def poll(self) -> int:
        return self._code
