from pathlib import Path

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


def test_process_environment_prefers_libraries_next_to_binary(tmp_path: Path) -> None:
    binary_dir = tmp_path / "atomic" / "build" / "bin"
    binary_dir.mkdir(parents=True)
    binary = binary_dir / "llama-server"
    binary.write_text("")
    (binary_dir / "libllama.so.0").write_text("")

    env = CampaignBackend.process_environment(
        (str(binary), "-m", "/models/model.gguf"),
        gpu="1",
        base={"LD_LIBRARY_PATH": "/stock/lib", "KEEP": "yes"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["LD_LIBRARY_PATH"] == f"{binary_dir}:/stock/lib"
    assert env["KEEP"] == "yes"
