from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from turbofit_runtime.backend import CampaignBackend


def test_process_environment_uses_metal_without_cuda_on_macos() -> None:
    env = CampaignBackend.process_environment(
        ("/opt/llama-server", "-m", "/models/model.gguf"),
        gpu="0",
        base={},
        platform_name="darwin",
    )

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["GGML_METAL"] == "1"


def test_process_environment_adds_loaded_kernel_compatibility_libraries() -> None:
    env = CampaignBackend.process_environment(
        ("/opt/llama-server",),
        gpu="0,1",
        base={"LD_LIBRARY_PATH": "/stock/lib"},
        platform_name="linux",
        compatibility_library_dir=lambda: "/compat/nvidia-580.159.03",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["LD_LIBRARY_PATH"] == "/compat/nvidia-580.159.03:/stock/lib"


def test_process_environment_selects_rocm_without_cuda_compatibility() -> None:
    env = CampaignBackend.process_environment(
        ("/opt/llama-server",),
        gpu="0,1",
        base={"CUDA_VISIBLE_DEVICES": "9"},
        platform_name="linux",
        backend_name="rocm",
        compatibility_library_dir=lambda: (_ for _ in ()).throw(
            AssertionError("ROCm must not inspect NVIDIA libraries")
        ),
    )

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["HIP_VISIBLE_DEVICES"] == "0,1"
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1"


def test_process_environment_prefers_libraries_next_to_binary(tmp_path: Path) -> None:
    binary_dir = tmp_path / "atomic" / "build" / "bin"
    binary_dir.mkdir(parents=True)
    binary = binary_dir / "llama-server"
    binary.write_text("")
    (binary_dir / "libllama.so").write_text("")

    env = CampaignBackend.process_environment(
        (str(binary), "-m", "/models/model.gguf"),
        gpu="1",
        base={"LD_LIBRARY_PATH": "/stock/lib", "KEEP": "yes"},
        platform_name="linux",
        compatibility_library_dir=lambda: None,
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["LD_LIBRARY_PATH"] == f"{binary_dir}:/stock/lib"
    assert env["KEEP"] == "yes"


def test_campaign_lease_stops_and_restores_only_previously_active_services(
    tmp_path: Path, monkeypatch,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:4] == ["systemctl", "--user", "is-active", "--quiet"]:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("turbofit_runtime.backend.subprocess.run", fake_run)
    backend = CampaignBackend(
        gateway_script=tmp_path / "gateway.py", result_dir=tmp_path,
        runtime_state=tmp_path / "runtime.json",
        campaign_lease_path=tmp_path / "campaign-lease.json",
    )

    backend.acquire_campaign_lease()
    assert (tmp_path / "campaign-lease.json").is_file()
    backend.release_campaign_lease()
    assert not (tmp_path / "campaign-lease.json").exists()

    assert ["systemctl", "--user", "stop", "turbofit-controller.service"] in calls
    assert ["systemctl", "--user", "start", "turbofit-controller.service"] in calls
    assert not any(
        command[:3] in (["systemctl", "--user", "stop"], ["systemctl", "--user", "start"])
        and "turbofit-gateway.service" in command
        for command in calls
    )


def test_failed_controller_suspend_removes_campaign_marker(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        if command[:4] == ["systemctl", "--user", "is-active", "--quiet"]:
            return SimpleNamespace(returncode=0)
        if command[:3] == ["systemctl", "--user", "stop"]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("turbofit_runtime.backend.subprocess.run", fake_run)
    marker = tmp_path / "campaign-lease.json"
    backend = CampaignBackend(
        gateway_script=tmp_path / "gateway.py", result_dir=tmp_path,
        runtime_state=tmp_path / "runtime.json", campaign_lease_path=marker,
    )

    with pytest.raises(subprocess.CalledProcessError):
        backend.acquire_campaign_lease()

    assert not marker.exists()


def test_wait_port_reusable_requires_three_clear_samples(tmp_path: Path, monkeypatch) -> None:
    backend = CampaignBackend(
        gateway_script=tmp_path / "gateway.py", result_dir=tmp_path,
        runtime_state=tmp_path / "runtime.json",
    )
    samples = iter((True, False, False, False))
    monkeypatch.setattr(backend, "_port_open", lambda port: next(samples))
    monkeypatch.setattr(backend, "_port_in_tcp_tables", lambda port: False)
    monkeypatch.setattr("turbofit_runtime.backend.time.sleep", lambda seconds: None)

    backend._wait_port_reusable(11605, timeout_s=1)
