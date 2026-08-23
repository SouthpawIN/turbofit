from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path

from turbofit_runtime.freetoken import FreeTokenCompatibility


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "install-freetoken-runtime"
    loader = importlib.machinery.SourceFileLoader("install_freetoken_runtime", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_installer_refuses_incompatible_host_before_network(monkeypatch, capsys) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "probe_freetoken_compatibility",
        lambda: FreeTokenCompatibility(
            False,
            "blocked",
            ("CUDA toolkit 13+ required",),
            ("CUDA toolkit 13+ with nvcc",),
        ),
    )
    monkeypatch.setattr(module, "install_runtime", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network touched")))

    code = module.run(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["CUDA toolkit 13+ required"]


def test_check_only_reports_missing_pinned_runtime(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _module()
    monkeypatch.setattr(module, "RUNTIME_HOME", tmp_path)
    monkeypatch.setattr(
        module,
        "probe_freetoken_compatibility",
        lambda: FreeTokenCompatibility(True, "candidate", (), ("Linux x86_64",)),
    )

    code = module.run(["--check-only", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["status"] == "missing"
    assert payload["revision"] == "0ab982f10905fa775962a4eddcb44caa50065251"
    assert payload["version"] == "0.1.2"
