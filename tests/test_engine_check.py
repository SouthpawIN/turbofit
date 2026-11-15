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
    from turbofit_runtime.engine_check import load_serve_matrix

    assert tuple(spec.engine_id for spec in engine_specs()) == (
        "llama.cpp",
        "mlx",
        "sglang",
        "vllm",
        "freetoken",
        "turbohaul-manager",
        "strata",
        "prism-llama.cpp",
        "salt-core",
        "saliency",
    )
    # the code registry stays in lockstep with the data registry (same engines;
    # order differs — probe order vs per-model preference)
    matrix = load_serve_matrix()
    assert {spec.engine_id for spec in engine_specs()} == set(matrix["preference_order"])
    assert {spec.engine_id for spec in engine_specs()} == set(matrix["engines"])
    turbohaul = next(spec for spec in engine_specs() if spec.engine_id == "turbohaul-manager")
    assert turbohaul.source_url == "https://github.com/MrTrenchTrucker/turbohaul-manager"
    assert turbohaul.source_ref == "v0.7.0"
    assert turbohaul.source_revision == "905b4506883313b17e1d4e0480a8e6ca6c63399b"
    assert turbohaul.openai_port == 11401


def test_linux_cuda_checks_all_engines_and_admits_native_linux_engines() -> None:
    reports = by_id(check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=580))

    assert set(reports) == {
        "llama.cpp", "mlx", "sglang", "vllm", "freetoken",
        "turbohaul-manager", "strata", "prism-llama.cpp", "salt-core", "saliency",
    }
    assert reports["llama.cpp"].compatible
    assert not reports["mlx"].compatible
    assert reports["sglang"].compatible
    assert reports["vllm"].compatible
    assert reports["freetoken"].compatible
    assert reports["turbohaul-manager"].compatible
    assert reports["salt-core"].compatible
    assert reports["saliency"].compatible


def test_macos_apple_silicon_admits_llama_mlx_and_vllm_metal() -> None:
    reports = by_id(check_engines(machine("darwin", "arm64", backend="metal")))

    assert reports["llama.cpp"].compatible
    assert reports["mlx"].compatible
    assert reports["vllm"].compatible
    assert not reports["sglang"].compatible
    assert not reports["freetoken"].compatible
    assert not reports["turbohaul-manager"].compatible
    assert not reports["salt-core"].compatible
    assert not reports["saliency"].compatible


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

    assert calls == ["http://127.0.0.1:11401", "http://127.0.0.1:8082", "http://127.0.0.1:8083", "http://127.0.0.1:8084", "http://127.0.0.1:8085"]
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


def test_low_vram_dense_candidates_report_as_candidate_not_eligible() -> None:
    from turbofit_runtime.engine_check import audition_pair, pair_engine_eligibility

    linux_cuda = by_id(check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=580))
    mac = by_id(check_engines(machine("darwin", "arm64", backend="metal")))
    assert linux_cuda["salt-core"].compatible and linux_cuda["saliency"].compatible
    assert not mac["salt-core"].compatible and not mac["saliency"].compatible

    status, reason = pair_engine_eligibility("qwen3-8-27b-gsq-rco-iq3s", "salt-core")
    assert status == "candidate" and "qualification gate" in reason
    assert pair_engine_eligibility("qwen3-8-27b-gsq-rco-iq3s", "saliency")[0] == "candidate"
    # released priority is untouched — turbohaul still leads the GSQ-RCO pair
    assert pair_engine_eligibility("qwen3-8-27b-gsq-rco-iq3s", "turbohaul-manager")[0] == "eligible"

    rows = {row["engine_id"]: row for row in audition_pair(
        machine("linux", "x86_64", backend="cuda"),
        main_alias="qwen3-8-27b-gsq-rco-iq3s",
        probes=probe_engines(machine("linux", "x86_64", backend="cuda"), driver_major=580,
                             which=lambda _c: None, module_available=lambda _m: False,
                             endpoint_ready=lambda _u: False),
    )}
    assert rows["salt-core"]["audition"] == "candidate"
    assert rows["salt-core"]["eligible"] is False
    assert rows["saliency"]["audition"] == "candidate"
    assert rows["turbohaul-manager"]["audition"] == "not-installed"
    assert rows["llama.cpp"]["audition"] == "not-installed"


def test_engine_registry_is_built_from_the_serve_matrix() -> None:
    from turbofit_runtime.engine_check import load_serve_matrix

    matrix = load_serve_matrix()
    specs = {spec.engine_id: spec for spec in engine_specs(matrix)}
    assert set(specs) == set(matrix["engines"])
    assert [spec.engine_id for spec in engine_specs(matrix)] == list(matrix["probe_order"])
    for engine_id, entry in matrix["engines"].items():
        spec = specs[engine_id]
        assert spec.display_name == entry["display_name"]
        assert spec.openai_port == entry["openai_port"]
        assert spec.commands == tuple(entry.get("detect", {}).get("commands", ()))
        assert spec.modules == tuple(entry.get("detect", {}).get("modules", ()))
        assert spec.source_url == entry.get("source", "")
        assert spec.source_ref == entry.get("ref", "")
        assert spec.source_revision == entry.get("revision", "")
        assert spec.endpoint_discoverable is bool(entry.get("endpoint_discoverable", False))
        assert spec.os == tuple(entry.get("os", ()))
        assert spec.backends == tuple(entry.get("backends", ()))
        assert spec.min_driver == int(entry.get("min_driver") or 0)
        assert spec.status == entry.get("status", "released")


def test_matrix_validation_rejects_probe_order_mismatch() -> None:
    import copy

    import pytest

    from turbofit_runtime.engine_check import load_serve_matrix

    bad = copy.deepcopy(load_serve_matrix())
    bad["probe_order"] = list(bad["probe_order"]) + ["ghost-engine"]
    with pytest.raises(ValueError, match="probe_order"):
        engine_specs(bad)

    unlisted = copy.deepcopy(load_serve_matrix())
    unlisted["engines"]["unlisted-engine"] = {"display_name": "Unlisted", "openai_port": 1}
    with pytest.raises(ValueError, match="exactly the same engines"):
        engine_specs(unlisted)


def test_generic_data_driven_fallback_classifies_matrix_only_engine() -> None:
    import copy

    from turbofit_runtime.engine_check import load_serve_matrix

    matrix = copy.deepcopy(load_serve_matrix())
    matrix["engines"]["ghost-engine"] = {
        "display_name": "Ghost Engine",
        "openai_port": 9999,
        "install_hint": "Install ghost-engine from example/ghost.",
        "detect": {"commands": ["ghost"], "modules": []},
        "os": ["linux"],
        "backends": ["cuda"],
        "min_driver": 580,
        "status": "candidate",
    }
    matrix["probe_order"] = list(matrix["probe_order"]) + ["ghost-engine"]

    strong = by_id(check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=590, matrix=matrix))
    assert strong["ghost-engine"].compatible
    assert strong["ghost-engine"].openai_port == 9999
    assert "supports" in strong["ghost-engine"].reason

    weak = by_id(check_engines(machine("linux", "x86_64", backend="cuda"), driver_major=570, matrix=matrix))
    assert not weak["ghost-engine"].compatible
    assert "r580" in weak["ghost-engine"].reason

    on_win = by_id(check_engines(machine("windows", "amd64", backend="cuda"), driver_major=590, matrix=matrix))
    assert on_win["ghost-engine"].compatible
    assert on_win["ghost-engine"].support_mode == "wsl"

    on_mac = by_id(check_engines(machine("darwin", "arm64", backend="metal"), matrix=matrix))
    assert not on_mac["ghost-engine"].compatible
    assert "requires linux" in on_mac["ghost-engine"].reason


def test_engine_status_and_promotion_hook(tmp_path) -> None:
    import json

    import pytest

    from turbofit_runtime.engine_check import engine_status, load_serve_matrix, set_engine_status

    matrix = load_serve_matrix()
    assert engine_status("saliency", matrix) == "candidate"
    assert engine_status("llama.cpp", matrix) == "released"
    with pytest.raises(ValueError):
        engine_status("ghost-engine", matrix)

    target = tmp_path / "matrix.json"
    target.write_text(json.dumps(matrix), encoding="utf-8")
    updated = set_engine_status(str(target), "saliency", "released")
    assert updated["engines"]["saliency"]["status"] == "released"
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["engines"]["saliency"]["status"] == "released"
    with pytest.raises(ValueError):
        set_engine_status(str(target), "saliency", "retired")


def test_load_serve_matrix_missing_file_raises(tmp_path) -> None:
    import pytest

    from turbofit_runtime.engine_check import load_serve_matrix

    with pytest.raises(FileNotFoundError, match="engine-serve matrix missing"):
        load_serve_matrix(tmp_path / "nope.json")
