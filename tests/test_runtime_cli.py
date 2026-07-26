from __future__ import annotations

import io
import json
from pathlib import Path

from test_selection import catalog, hardware
from turbofit_runtime.runtime_cli import run
from turbofit_runtime.selection import load_selection


def test_set_auto_persists_safe_terminal_selection(tmp_path: Path) -> None:
    output = io.StringIO()
    state = tmp_path / "selection.json"

    code = run(
        ["set", "auto"],
        catalog=catalog(),
        hardware=hardware(24576),
        selection_path=state,
        output=output,
    )

    assert code == 0
    payload = json.loads(output.getvalue())
    assert payload["profile_id"] == "hardware-24gb"
    assert payload["effective_rung"] == "api"
    assert payload["controller_pending"] is True
    assert load_selection(state)["mode"] == "auto"


def test_manual_use_alias_refuses_a_profile_that_does_not_fit(tmp_path: Path) -> None:
    output = io.StringIO()

    code = run(
        ["use", "hardware-48gb"],
        catalog=catalog(),
        hardware=hardware(24576),
        selection_path=tmp_path / "selection.json",
        output=output,
    )

    assert code == 2
    assert "does not fit physical hardware" in json.loads(output.getvalue())["error"]
    assert not (tmp_path / "selection.json").exists()


def test_list_marks_manual_profiles_by_safe_compatibility(tmp_path: Path) -> None:
    output = io.StringIO()

    code = run(
        ["list"],
        catalog=catalog(),
        hardware=hardware(24576),
        selection_path=tmp_path / "selection.json",
        output=output,
    )

    assert code == 0
    rows = {row["id"]: row for row in json.loads(output.getvalue())}
    assert rows["hardware-24gb"]["manual_compatible"] is True
    assert rows["hardware-48gb"]["manual_compatible"] is False
    assert rows["hardware-16gb"]["manual_compatible"] is True


def test_status_is_read_only_and_reports_missing_selection(tmp_path: Path) -> None:
    output = io.StringIO()
    code = run(
        ["status"],
        catalog=catalog(),
        hardware=hardware(8192),
        selection_path=tmp_path / "missing.json",
        output=output,
    )
    assert code == 1
    assert json.loads(output.getvalue())["configured"] is False


def test_runtime_entrypoint_has_no_direct_process_or_container_authority() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "turbofit-runtime").read_text()
    for forbidden in ("os.kill", "os.killpg", "SIGKILL", '"docker", "rm"', "subprocess.Popen"):
        assert forbidden not in script
