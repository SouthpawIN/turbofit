from __future__ import annotations

from turbofit_runtime.engine_check import (
    check_engines,
    engine_specs,
    is_wsl_release,
    parse_driver_major,
    probe_engines,
)
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint


def machine(os_name: str, architecture: str, *, backend: str | None = None) -> HardwareFingerprint:
    devices = () if backend is None else (
        AcceleratorDevice(
            index=0,
            uuid="device-0",
            name="accelerator",
            vendor="apple" if backend == "metal" else "nvidia",
            backend=backend,
            memory_total_mb=24_576,
            compute_capability="8.6" if backend == "cuda" else None,
            bus_id=None,
        ),
    )
    return HardwareFingerprint(os_name, architecture, 65_536, devices=devices)


def by_id(reports):
    return {report.engine_id: report for report in reports}


def test_registry_contains_every_turbofit_check_engine() -> None:
    assert tuple(spec.engine_id for spec in engine_specs()) == (
        "llama.cpp",
        "mlx",
        "sglang",
        "vllm",
        "freetoken",
        "turbohaul-manager",
    )
    turbohaul = engine_specs()[-1]
    assert turbohaul.source_url == "https://github.com/MrTrenchTrucker/turbohaul-manager"
    assert turbohaul.source_ref == "v0.7.0"
    assert turbohaul.source_revision == "905b4506883313b17e1d4e0480a8e6ca6c63399b"
    assert turbohaul.openai_port == 11401


def test_linux_cuda_checks_all_engines_and_admits_native_linux_engines() -> None:
    reports = by_id(check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=580))

    assert set(reports) == {"llama.cpp", "mlx", "sglang", "vllm", "freetoken", "turbohaul-manager"}
    assert reports["llama.cpp"].compatible
    assert not reports["mlx"].compatible
    assert reports["sglang"].compatible
    assert reports["vllm"].compatible
    assert reports["freetoken"].compatible
    assert reports["turbohaul-manager"].compatible


def test_macos_apple_silicon_admits_llama_mlx_and_vllm_metal() -> None:
    reports = by_id(check_engines(machine("darwin", "arm64", backend="metal")))

    assert reports["llama.cpp"].compatible
    assert reports["mlx"].compatible
    assert reports["vllm"].compatible
    assert not reports["sglang"].compatible
    assert not reports["freetoken"].compatible
    assert not reports["turbohaul-manager"].compatible


def test_native_windows_reports_wsl_only_engines_without_omitting_them() -> None:
    reports = by_id(check_engines(machine("windows", "amd64", backend="cuda"), driver_major=580))

    assert reports["llama.cpp"].compatible
    assert not reports["mlx"].compatible
    assert not reports["sglang"].compatible
    assert reports["sglang"].support_mode == "wsl"
    assert not reports["vllm"].compatible
    assert reports["vllm"].support_mode == "wsl"
    assert not reports["freetoken"].compatible
    assert reports["freetoken"].support_mode == "wsl"
    assert not reports["turbohaul-manager"].compatible
    assert reports["turbohaul-manager"].support_mode == "wsl"


def test_wsl_is_checked_as_linux_not_native_windows() -> None:
    reports = by_id(
        check_engines(
            machine("linux", "x86_64", backend="cuda"),
            driver_major=580,
            is_wsl=True,
        )
    )

    assert reports["sglang"].compatible
    assert reports["vllm"].compatible
    assert reports["freetoken"].compatible
    assert reports["freetoken"].support_mode == "wsl"
    assert reports["turbohaul-manager"].compatible
    assert reports["turbohaul-manager"].support_mode == "wsl"


def test_turbohaul_manager_requires_linux_x86_64_nvidia() -> None:
    cpu = by_id(check_engines(machine("linux", "x86_64")))["turbohaul-manager"]
    arm = by_id(check_engines(machine("linux", "aarch64", backend="cuda")))["turbohaul-manager"]

    assert not cpu.compatible and "NVIDIA" in cpu.reason
    assert not arm.compatible and "x86_64" in arm.reason


def test_freetoken_requires_linux_x86_cuda_and_r580_or_newer() -> None:
    old_driver = by_id(
        check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=579)
    )["freetoken"]
    cpu = by_id(check_engines(machine("linux", "x86_64"), driver_major=None))["freetoken"]
    arm = by_id(
        check_engines(machine("linux", "aarch64", backend="cuda"), driver_major=580)
    )["freetoken"]

    assert not old_driver.compatible and "r580" in old_driver.reason
    assert not cpu.compatible and "CUDA" in cpu.reason
    assert not arm.compatible and "x86_64" in arm.reason


def test_probe_reports_installed_and_running_separately_from_compatibility() -> None:
    reports = by_id(
        probe_engines(
            machine("linux", "x86_64", backend="cuda"),
            driver_major=580,
            which=lambda command: "/usr/bin/llama-server" if command == "llama-server" else None,
            module_available=lambda module: module == "vllm",
            endpoint_ready=lambda url: url == "http://127.0.0.1:8000",
        )
    )

    assert reports["llama.cpp"].compatible and reports["llama.cpp"].installed
    assert not reports["llama.cpp"].running
    assert reports["vllm"].installed and reports["vllm"].running
    assert not reports["sglang"].installed and not reports["sglang"].running
    assert not reports["mlx"].compatible and not reports["mlx"].eligible


def test_probe_only_calls_endpoint_for_endpoint_discoverable_uninstalled_runtime() -> None:
    calls: list[str] = []

    reports = probe_engines(
        machine("linux", "x86_64", backend="cuda"),
        driver_major=580,
        which=lambda _command: None,
        module_available=lambda _module: False,
        endpoint_ready=lambda url: calls.append(url) or False,
    )

    assert calls == ["http://127.0.0.1:11401"]
    assert all(not report.running for report in reports)


def test_probe_discovers_containerized_turbohaul_manager_from_its_endpoint() -> None:
    reports = by_id(
        probe_engines(
            machine("linux", "x86_64", backend="cuda"),
            driver_major=580,
            which=lambda _command: None,
            module_available=lambda _module: False,
            endpoint_ready=lambda url: url == "http://127.0.0.1:11401",
        )
    )

    turbohaul = reports["turbohaul-manager"]
    assert turbohaul.installed
    assert turbohaul.running
    assert turbohaul.eligible


def test_maple_pair_is_fork_llama_or_turbohaul_not_vllm() -> None:
    from pathlib import Path
    from turbofit_runtime.engine_check import audition_pair, pair_engine_eligibility

    assert pair_engine_eligibility("maple-preview-tq2", "llama.cpp")[0] == "eligible"
    assert pair_engine_eligibility("maple-preview-tq2", "turbohaul-manager")[0] == "eligible"
    assert pair_engine_eligibility("maple-preview-tq2", "vllm")[0] == "ineligible"
    assert pair_engine_eligibility("maple-preview-tq2", "sglang")[0] == "ineligible"
    assert pair_engine_eligibility("maple-preview-tq2", "mlx")[0] == "alternate"
    rows = audition_pair(
        machine("linux", "x86_64", backend="cuda"),
        main_alias="maple-preview-tq2",
        aux_alias="auto",
        context=131072,
        probes=probe_engines(
            machine("linux", "x86_64", backend="cuda"),
            driver_major=580,
            which=lambda command: "/usr/bin/llama-server" if command == "llama-server" else None,
            module_available=lambda _module: False,
            endpoint_ready=lambda _url: False,
        ),
    )
    by_engine = {row["engine_id"]: row for row in rows}
    assert by_engine["llama.cpp"]["audition"] == "installed"
    assert by_engine["vllm"]["audition"] == "ineligible"
    assert Path(__file__).parents[1].joinpath("references/engine-serve-matrix.json").is_file()


def test_driver_and_wsl_detection_are_deterministic() -> None:
    assert parse_driver_major("580.159.03") == 580
    assert parse_driver_major("NVIDIA-SMI has failed") is None
    assert is_wsl_release("6.6.87.2-microsoft-standard-WSL2")
    assert not is_wsl_release("6.8.0-generic")
