"""A route must never be read from a gateway this run did not start.

Observed on omarchy: an interrupted attempt left its campaign gateway alive on
18092.  The next attempt's gateway died with ``EADDRINUSE``, but ``/status``
still answered 200 — from the *stale* process — and the runner happily recorded
that process's (cached) route for a pair it had not started.  That is the same
class of defect as the ``auto:<main>`` route: the campaign reporting a pair it
never ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.backend import CampaignBackend
from turbofit_runtime.recipes import ResolvedRecipe


class _ExitedProcess:
    returncode = 1

    def poll(self) -> int:
        return 1


def _backend(tmp_path: Path) -> CampaignBackend:
    return CampaignBackend(
        gateway_script=tmp_path / "turbofit-gateway.py",
        gateway_port=18092,
        result_dir=tmp_path / "runtime",
        runtime_state=tmp_path / "runtime-state.json",
    )


def _recipe() -> ResolvedRecipe:
    return ResolvedRecipe(
        row_id="arete-27b-q5--maple-preview-tq2--64k",
        profile_name="arete-27b-q5--maple-preview-tq2--64k",
        main_alias="arete-27b-q5",
        aux_alias="maple-preview-tq2",
        aux_mode="dedicated",
        components=(),
        main_context=65536,
        aux_context=65536,
    )


def test_route_refuses_a_status_answer_from_a_foreign_gateway(tmp_path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "campaign-gateway.log").write_text(
        "OSError: [Errno 98] Address already in use\n", encoding="utf-8"
    )
    monkeypatch.setattr("turbofit_runtime.backend.subprocess.Popen", lambda *a, **k: _ExitedProcess())
    monkeypatch.setattr(backend, "_port_open", lambda port: False)
    monkeypatch.setattr(
        backend, "_request_json",
        lambda *a, **k: (200, {"main": {"alias": "arete-27b-q5"}, "aux": {"alias": "auto:arete-27b-q5"}}, None),
    )
    monkeypatch.setattr(backend, "_listening_gateway_owner", lambda: (269818, "python3 scripts/turbofit-gateway.py"))

    with pytest.raises(RuntimeError) as error:
        backend.route(_recipe(), {})

    message = str(error.value)
    assert "campaign gateway exited" in message
    assert "269818" in message
    # The stale gateway's route never reached the caller.
    assert "auto:" not in message


def test_reclaim_is_a_noop_when_the_port_is_free(tmp_path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    monkeypatch.setattr(backend, "_port_open", lambda port: False)
    calls: list = []
    monkeypatch.setattr(backend, "_campaign_gateway_candidates", lambda: calls.append(1) or [])

    backend._reclaim_campaign_gateway()

    assert calls == []


def test_reclaim_only_targets_gateways_holding_this_state_file(tmp_path) -> None:
    backend = _backend(tmp_path)
    state = json.dumps({"active": "campaign:x"})
    (tmp_path / "runtime-state.json").write_text(state, encoding="utf-8")

    # No other process can match: the environment must name this exact state
    # file, port, and campaign flag.
    for pid, _command in backend._campaign_gateway_candidates():
        environ = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
        assert f"TURBOFIT_RUNTIME_STATE={tmp_path / 'runtime-state.json'}" in environ
        assert "TURBOFIT_CAMPAIGN_GATEWAY=true" in environ.lower() or "TURBOFIT_CAMPAIGN_GATEWAY=1" in environ
