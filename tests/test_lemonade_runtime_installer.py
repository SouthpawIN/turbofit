from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import tarfile
import zipfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/install-lemonade-runtime"


def load_script():
    loader = SourceFileLoader("install_lemonade_runtime", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_assets_are_version_and_sha256_pinned() -> None:
    module = load_script()

    assert module.VERSION == "11.5.1"
    assert set(module.ASSETS) == {
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("darwin", "arm64"),
        ("windows", "amd64"),
    }
    assert all(len(checksum) == 64 for _, checksum in module.ASSETS.values())
    assert all(module.VERSION in asset for asset, _ in module.ASSETS.values())


def test_native_service_is_loopback_only() -> None:
    module = load_script()
    command = module.service_command(Path("/runtime-parent"))

    assert command[-4:] == ["--host", "127.0.0.1", "--port", "13305"]
    assert command[0].endswith("/lemonade-11.5.1/lemond")


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    module = load_script()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside", "bad")

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        module.extract_archive(archive, tmp_path / "extract")


def test_install_files_verifies_and_extracts_native_archive(tmp_path: Path, monkeypatch) -> None:
    module = load_script()
    fixture = tmp_path / "fixture.tar.gz"
    with tarfile.open(fixture, "w:gz") as bundle:
        for name in ("lemond", "lemonade"):
            payload = b"native-binary"
            info = tarfile.TarInfo(f"bundle/{name}")
            info.size = len(payload)
            info.mode = 0o755
            bundle.addfile(info, io.BytesIO(payload))
    checksum = module.sha256_file(fixture)
    monkeypatch.setitem(module.ASSETS, ("linux", "x86_64"), ("fixture.tar.gz", checksum))
    monkeypatch.setattr(module.urllib.request, "urlretrieve", lambda _url, target: Path(target).write_bytes(fixture.read_bytes()))

    result = module.install_files(base=tmp_path / "runtimes", system="linux", machine="x86_64")

    assert result["installed"] is True
    assert Path(result["binary"]).read_bytes() == b"native-binary"
    assert os.access(result["binary"], os.X_OK)


def test_installer_has_no_legacy_runtime_dependency() -> None:
    forbidden = "dock" + "er"
    assert forbidden not in SCRIPT.read_text().lower()


def test_installer_is_directly_invocable_outside_repo_pythonpath(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
