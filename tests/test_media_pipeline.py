from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from turbofit_runtime import media_pipeline as mp
from turbofit_runtime.multimodal import MultimodalCatalog, recommend_multimodal
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "references" / "multimodal-models.json"
VIDEO_MODEL = "animatediff-sd15-motion"


def catalog() -> MultimodalCatalog:
    return MultimodalCatalog.load(CATALOG)


def no_modules(_name: str):
    return None


def all_modules(_name: str):
    return object()


def hardware(*, ram_mb: int = 377000, device_mb: int = 24576) -> HardwareFingerprint:
    devices = (
        AcceleratorDevice(
            index=0,
            uuid="gpu",
            name="RTX 3090",
            vendor="nvidia",
            memory_total_mb=device_mb,
            backend="cuda",
            compute_capability="8.6",
            bus_id="0000:01:00.0",
        ),
    )
    return HardwareFingerprint(os="linux", architecture="x86_64", system_ram_mb=ram_mb, devices=devices)


def test_every_catalog_runtime_has_an_adapter() -> None:
    runtimes = {model.runtime for model in catalog().models}
    undeclared = sorted(runtimes - set(mp.RUNTIMES))
    assert undeclared == [], f"catalog declares runtimes with no adapter: {undeclared}"


def test_missing_engine_names_the_install_command() -> None:
    status = mp.probe_runtime("animatediff", find_spec=no_modules)

    assert status.status == mp.WAITING_RUNTIME
    assert status.ready is False
    assert "diffusers" in status.missing_modules
    assert "uv pip install" in status.command
    assert "torch" in status.command


def test_missing_weights_names_the_download_command(tmp_path: Path) -> None:
    status = mp.probe_runtime(
        "animatediff",
        weights=(tmp_path / "missing-base",),
        find_spec=all_modules,
        download_command="hf download Meina/MeinaMix_V11 --revision abc --local-dir /models/base",
    )

    assert status.status == mp.WAITING_DOWNLOAD
    assert status.missing_paths == (str(tmp_path / "missing-base"),)
    assert status.command.startswith("hf download Meina/MeinaMix_V11")


def test_require_runtime_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(mp.MediaRuntimeError) as excinfo:
        mp.require_runtime("animatediff", weights=(tmp_path / "nope",), find_spec=all_modules)

    error = excinfo.value
    assert error.status == mp.WAITING_DOWNLOAD
    assert "fix:" in str(error)
    assert str(tmp_path / "nope") in str(error)


def test_unknown_runtime_is_reported_not_guessed() -> None:
    status = mp.probe_runtime("not-a-runtime", find_spec=all_modules)

    assert status.status == mp.UNKNOWN_RUNTIME
    assert status.ready is False
    assert "RUNTIMES" in status.command


def test_video_job_rejects_frames_outside_one_motion_context() -> None:
    job = mp.VideoJob(
        prompt="garage", output=Path("out.mp4"), model_root=Path("base"),
        motion_adapter=Path("motion"), frames=20,
    )
    with pytest.raises(ValueError, match="multiple of 16"):
        job.validate()


def test_video_job_rejects_unaligned_canvas() -> None:
    job = mp.VideoJob(
        prompt="garage", output=Path("out.mp4"), model_root=Path("base"),
        motion_adapter=Path("motion"), width=641,
    )
    with pytest.raises(ValueError, match="multiple of 8"):
        job.validate()


def test_video_command_is_the_documented_invocation(tmp_path: Path) -> None:
    job = mp.VideoJob(
        prompt="anime garage", output=tmp_path / "clip.mp4",
        model_root=tmp_path / "base", motion_adapter=tmp_path / "motion",
        frames=16, width=640, height=384, steps=25, fps=8, seed=7,
    )

    argv = mp.video_command(job, python="/venv/bin/python")

    assert argv[:3] == ["/venv/bin/python", "-m", "turbofit_runtime.animatediff_worker"]
    joined = " ".join(argv)
    for value in ("--frames 16", "--width 640", "--height 384", "--steps 25", "--fps 8", "--seed 7"):
        assert value in joined
    assert "anime garage" in joined


def test_run_video_refuses_to_start_without_engine(tmp_path: Path) -> None:
    calls = []

    def runner(argv, **kwargs):  # pragma: no cover - must not be reached
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    job = mp.VideoJob(
        prompt="garage", output=tmp_path / "clip.mp4", model_root=tmp_path / "base",
        motion_adapter=tmp_path / "motion",
    )
    with pytest.raises(mp.MediaRuntimeError) as excinfo:
        mp.run_video(job, runner=runner, find_spec=no_modules)

    assert excinfo.value.status == mp.WAITING_RUNTIME
    assert calls == []


def test_run_video_reports_worker_failure(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "motion").mkdir()
    job = mp.VideoJob(
        prompt="garage", output=tmp_path / "clip.mp4", model_root=tmp_path / "base",
        motion_adapter=tmp_path / "motion",
    )

    def failing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2)

    with pytest.raises(mp.MediaRuntimeError, match="exited 2"):
        mp.run_video(job, runner=failing, find_spec=all_modules)


def test_run_video_rejects_a_silent_no_op(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "motion").mkdir()
    job = mp.VideoJob(
        prompt="garage", output=tmp_path / "clip.mp4", model_root=tmp_path / "base",
        motion_adapter=tmp_path / "motion",
    )

    def do_nothing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(mp.MediaRuntimeError, match="missing"):
        mp.run_video(job, runner=do_nothing, find_spec=all_modules)


def test_run_video_invokes_the_worker_and_returns_the_clip(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "motion").mkdir()
    job = mp.VideoJob(
        prompt="garage", output=tmp_path / "clip.mp4", model_root=tmp_path / "base",
        motion_adapter=tmp_path / "motion",
    )
    seen = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        Path(job.output).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(argv, 0)

    assert mp.run_video(job, runner=runner, find_spec=all_modules) == job.output
    assert seen["argv"][:3] == ["python", "-m", "turbofit_runtime.animatediff_worker"]


def test_assemble_command_crossfades_multiple_shots(tmp_path: Path) -> None:
    first = tmp_path / "shot1.mp4"
    second = tmp_path / "shot2.mp4"
    for shot in (first, second):
        shot.write_bytes(b"x")

    argv = mp.assemble_command([first, second], tmp_path / "final.mp4", fps=8, transition_seconds=0.5)

    assert argv.count("-i") == 2
    filter_complex = argv[argv.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.5" in filter_complex
    assert "-movflags" in argv


def test_assemble_command_rejects_missing_shot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mp.assemble_command([tmp_path / "nope.mp4"], tmp_path / "final.mp4")


def test_catalog_download_command_pins_every_component(tmp_path: Path) -> None:
    model = catalog().by_id(VIDEO_MODEL)

    command = mp.catalog_download_command(model, "/models")

    for _, source, revision in model.components:
        repo = source.split("huggingface.co/")[-1]
        assert f"hf download {repo} --revision {revision}" in command


def test_catalog_component_paths_follow_pinned_sources(tmp_path: Path) -> None:
    model = catalog().by_id(VIDEO_MODEL)

    paths = mp.catalog_component_paths(model, tmp_path)

    assert len(paths) == len(model.components)
    assert all(path.parent == tmp_path for path in paths)


def test_recommendation_never_uses_now_when_the_runtime_is_missing() -> None:
    probed = mp.probe_all(find_spec=no_modules)

    result = recommend_multimodal(hardware=hardware(), catalog=catalog(), runtimes=probed)
    rows = {row["id"]: row for rows in result["modalities"].values() for row in rows}

    local_video = rows[VIDEO_MODEL]
    assert local_video["fit"] is True
    assert local_video["runtime_status"] == mp.WAITING_RUNTIME
    assert local_video["action"] == "install-adapter"
    assert local_video["fix_command"]
    for row in rows.values():
        if row["runtime_ready"] is False:
            assert row["action"] != "use-now"


def test_recommendation_marks_video_ready_when_engine_and_weights_exist(tmp_path: Path) -> None:
    status = mp.probe_runtime("animatediff", weights=(), find_spec=all_modules)
    assert status.status == mp.READY

    result = recommend_multimodal(
        hardware=hardware(), catalog=catalog(), runtimes={"animatediff": status.as_dict()}
    )
    row = next(item for item in result["modalities"]["video"] if item["id"] == VIDEO_MODEL)

    assert row["runtime_ready"] is True
    assert row["action"] == "run-turbofit-local"


def test_recommendation_defaults_remain_memory_only() -> None:
    result = recommend_multimodal(hardware=hardware(), catalog=catalog())
    row = next(item for item in result["modalities"]["video"] if item["id"] == VIDEO_MODEL)

    assert row["runtime_status"] is None
    assert row["runtime_ready"] is None


def test_promo_manifest_matches_the_catalog() -> None:
    manifest = json.loads((ROOT / "promo" / "local-video" / "prompts.json").read_text())
    model = catalog().by_id(manifest["model"])

    assert model.runtime == "animatediff"
    assert manifest["frames"] % 16 == 0
    base = Path(manifest["base_path"]).name
    motion = Path(manifest["motion_adapter_path"]).name
    assert base and motion


def test_plugin_snapshot_probes_runtimes_against_the_models_root(tmp_path: Path) -> None:
    import plugin_tools

    catalog_model = catalog()
    states = plugin_tools.probe_multimodal_runtimes(catalog_model, tmp_path)

    assert states, "the catalog references local runtimes"
    assert all(state["ready"] is False for state in states.values())
    assert states["animatediff"]["status"] in {mp.WAITING_RUNTIME, mp.WAITING_DOWNLOAD}
    assert states["animatediff"]["command"]


def test_models_root_can_be_overridden_by_config(tmp_path: Path) -> None:
    import plugin_tools

    config = {"turbofit": {"multimodal": {"models_root": str(tmp_path)}}}

    assert plugin_tools._multimodal_models_root(config) == tmp_path
    assert plugin_tools._multimodal_models_root(None) == Path.home() / "Models" / "storage"


def test_hermes_video_is_not_ready_without_a_configured_provider() -> None:
    import plugin_tools

    payload = recommend_multimodal(hardware=hardware(), catalog=catalog())
    plugin_tools.apply_hermes_provider_status(payload, {"model": {"provider": "nous"}})
    row = next(item for item in payload["modalities"]["video"] if item["id"] == "hermes-video")

    assert row["action"] == "install-adapter"
    assert row["blocked_by"] == "waiting-config"
    assert "video_gen" in row["fix_command"]


def test_hermes_video_stays_ready_when_a_provider_is_configured() -> None:
    import plugin_tools

    payload = recommend_multimodal(hardware=hardware(), catalog=catalog())
    plugin_tools.apply_hermes_provider_status(
        payload, {"video_gen": {"provider": "nous", "model": "some-model"}}
    )
    row = next(item for item in payload["modalities"]["video"] if item["id"] == "hermes-video")

    assert row["action"] == "use-now"
    assert row["blocked_by"] == ""
