"""Local media runtime registry: readiness probes, plans, and invocations.

The multimodal catalog declares a ``runtime`` string for every model.  Until this
module existed nothing mapped those strings to an executable invocation, so a
declared runtime could be advertised as usable while no engine, no weights and
no code path existed.  Everything here is deliberately import-light: the GPU
libraries are imported only inside the worker subprocess, so the readiness
probes and plan builders stay testable on a CPU-only box.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "turbofit.media-runtime/v1"

#: Status vocabulary shared with the rest of Turbofit (see ``intelligence_campaign``).
READY = "ready"
DELEGATED = "delegated"
WAITING_RUNTIME = "waiting-runtime"
WAITING_DOWNLOAD = "waiting-download"
WAITING_MEMORY = "waiting-memory"
UNKNOWN_RUNTIME = "unknown-runtime"

TORCH_INSTALL = "uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
DIFFUSERS_INSTALL = "uv pip install diffusers transformers accelerate safetensors"


class MediaRuntimeError(RuntimeError):
    """A local media runtime cannot run, with an actionable next command."""

    def __init__(self, runtime: str, status: str, message: str, command: str = "") -> None:
        self.runtime = runtime
        self.status = status
        self.command = command
        text = f"{runtime}: {message}"
        if command:
            text = f"{text}\n  fix: {command}"
        super().__init__(text)


@dataclass(frozen=True)
class MediaRuntime:
    """One declared runtime, its engine dependencies and its entrypoint."""

    name: str
    modalities: tuple[str, ...]
    modules: tuple[str, ...]
    install_command: str
    entrypoint: str = ""
    notes: str = ""

    def missing_modules(self, find_spec: Callable[[str], Any]) -> tuple[str, ...]:
        missing = []
        for module in self.modules:
            try:
                found = find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.append(module)
        return tuple(missing)


RUNTIMES: dict[str, MediaRuntime] = {
    "animatediff": MediaRuntime(
        name="animatediff",
        modalities=("video",),
        modules=("torch", "diffusers", "transformers", "safetensors"),
        install_command=f"{TORCH_INSTALL}\n  {DIFFUSERS_INSTALL}",
        entrypoint="turbofit_runtime.animatediff_worker",
        notes="SD1.5 motion adapter; 16-frame macro-conditions, assembled to mp4 with ffmpeg",
    ),
    "diffusers": MediaRuntime(
        name="diffusers",
        modalities=("image", "video", "music"),
        modules=("torch", "diffusers"),
        install_command=f"{TORCH_INSTALL}\n  {DIFFUSERS_INSTALL}",
        notes="text-to-image; video pipelines need the diffusers-video runtime",
    ),
    "diffusers-modular": MediaRuntime(
        name="diffusers-modular",
        modalities=("image", "video", "music"),
        modules=("torch", "diffusers"),
        install_command=f"{TORCH_INSTALL}\n  {DIFFUSERS_INSTALL}",
        notes="ModularPipeline workflows",
    ),
    "ace-step": MediaRuntime(
        name="ace-step",
        modalities=("music",),
        modules=("torch", "acestep"),
        install_command="uv pip install ace-step",
    ),
    "transformers-command": MediaRuntime(
        name="transformers-command",
        modalities=("tts", "stt", "image"),
        modules=("torch", "transformers"),
        install_command=f"{TORCH_INSTALL}\n  uv pip install transformers",
    ),
    "faster-whisper": MediaRuntime(
        name="faster-whisper",
        modalities=("stt",),
        modules=("faster_whisper",),
        install_command="uv pip install faster-whisper",
    ),
    "local-stt-command": MediaRuntime(
        name="local-stt-command",
        modalities=("stt",),
        modules=("torch", "transformers"),
        install_command=f"{TORCH_INSTALL}\n  uv pip install transformers",
    ),
    "personaplex": MediaRuntime(
        name="personaplex",
        modalities=("stt", "tts"),
        modules=("torch", "moshi"),
        install_command="uv pip install moshi",
    ),
    "soprano": MediaRuntime(
        name="soprano",
        modalities=("tts",),
        modules=("torch", "transformers"),
        install_command=f"{TORCH_INSTALL}\n  uv pip install transformers",
    ),
    # Builtin runtimes are configured inside Hermes, not installed by Turbofit.
    "hermes": MediaRuntime(
        name="hermes",
        modalities=("image", "video", "music"),
        modules=(),
        install_command="",
        notes="delegated to the Hermes provider configuration, not a local engine",
    ),
    "edge": MediaRuntime(
        name="edge",
        modalities=("tts",),
        modules=("edge_tts",),
        install_command="uv pip install edge-tts",
        notes="hosted Edge TTS voice",
    ),
}

DELEGATED_RUNTIMES = frozenset({"hermes", "edge"})


@dataclass(frozen=True)
class RuntimeStatus:
    runtime: str
    status: str
    ready: bool
    detail: str
    command: str = ""
    missing_modules: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "status": self.status,
            "ready": self.ready,
            "detail": self.detail,
            "command": self.command,
            "missing_modules": list(self.missing_modules),
            "missing_paths": list(self.missing_paths),
        }


def _default_find_spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None


def probe_runtime(
    runtime: str,
    *,
    weights: Sequence[str | Path] = (),
    find_spec: Callable[[str], Any] | None = None,
    download_command: str = "",
) -> RuntimeStatus:
    """Report whether ``runtime`` can actually execute on this machine.

    ``weights`` are paths that must exist before the runtime can generate;
    ``download_command`` is the copy-paste fix when they do not.
    """
    spec = RUNTIMES.get(runtime)
    if spec is None:
        return RuntimeStatus(
            runtime=runtime,
            status=UNKNOWN_RUNTIME,
            ready=False,
            detail=f"no Turbofit adapter declares the runtime {runtime!r}",
            command="add it to turbofit_runtime.media_pipeline.RUNTIMES",
        )
    if runtime in DELEGATED_RUNTIMES and runtime != "edge":
        return RuntimeStatus(
            runtime=runtime,
            status=DELEGATED,
            ready=True,
            detail="configured by Hermes; Turbofit does not bundle this engine",
            command="turbofit-configure-multimodal" if runtime == "hermes" else spec.install_command,
        )
    missing_modules = spec.missing_modules(find_spec or _default_find_spec)
    if missing_modules:
        return RuntimeStatus(
            runtime=runtime,
            status=WAITING_RUNTIME,
            ready=False,
            detail=f"missing engine module(s): {', '.join(missing_modules)}",
            command=spec.install_command,
            missing_modules=missing_modules,
        )
    missing_paths = tuple(str(path) for path in weights if not Path(path).exists())
    if missing_paths:
        return RuntimeStatus(
            runtime=runtime,
            status=WAITING_DOWNLOAD,
            ready=False,
            detail=f"missing weights: {', '.join(missing_paths)}",
            command=download_command or _default_download_hint(missing_paths),
            missing_paths=missing_paths,
        )
    return RuntimeStatus(
        runtime=runtime,
        status=READY,
        ready=True,
        detail=spec.notes or "engine and weights present",
    )


def _default_download_hint(paths: Sequence[str]) -> str:
    first = shlex.quote(str(paths[0]))
    return f"hf download <org/repo> --local-dir {first}"


def probe_all(
    *, weights: Mapping[str, Sequence[str | Path]] | None = None,
    find_spec: Callable[[str], Any] | None = None,
) -> dict[str, RuntimeStatus]:
    weights = weights or {}
    return {
        name: probe_runtime(name, weights=weights.get(name, ()), find_spec=find_spec)
        for name in RUNTIMES
    }


def require_runtime(runtime: str, **kwargs: Any) -> RuntimeStatus:
    status = probe_runtime(runtime, **kwargs)
    if not status.ready:
        raise MediaRuntimeError(
            runtime,
            status.status,
            status.detail,
            status.command,
        )
    return status


# --------------------------------------------------------------------------
# video jobs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoJob:
    """A single local text-to-video request."""

    prompt: str
    output: Path
    model_root: Path
    motion_adapter: Path
    frames: int = 16
    width: int = 640
    height: int = 384
    steps: int = 25
    guidance_scale: float = 7.5
    fps: int = 8
    seed: int = 0
    negative_prompt: str = ""

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("video job needs a prompt")
        if self.frames < 1 or self.frames % 16:
            raise ValueError(
                f"frames must be a positive multiple of 16 (one motion context), got {self.frames}"
            )
        for label, value in (("width", self.width), ("height", self.height)):
            if value % 8 or value < 256:
                raise ValueError(f"{label} must be a multiple of 8 and at least 256, got {value}")
        if not 0 < self.fps <= 60:
            raise ValueError(f"fps out of range: {self.fps}")
        if self.steps < 1:
            raise ValueError("steps must be positive")


VIDEO_RUNTIME = "animatediff"


def video_command(job: VideoJob, *, python: str = "python") -> list[str]:
    """Exact argv that produces ``job.output``; the worker validates again."""
    job.validate()
    return [
        python,
        "-m",
        "turbofit_runtime.animatediff_worker",
        "--model-root", str(job.model_root),
        "--motion-adapter", str(job.motion_adapter),
        "--prompt", job.prompt,
        "--negative-prompt", job.negative_prompt,
        "--output", str(job.output),
        "--frames", str(job.frames),
        "--width", str(job.width),
        "--height", str(job.height),
        "--steps", str(job.steps),
        "--guidance-scale", str(job.guidance_scale),
        "--fps", str(job.fps),
        "--seed", str(job.seed),
    ]


Runner = Callable[..., subprocess.CompletedProcess]


def run_video(
    job: VideoJob,
    *,
    runner: Runner | None = None,
    python: str = "python",
    find_spec: Callable[[str], Any] | None = None,
    download_command: str = "",
) -> Path:
    """Generate one clip, raising ``MediaRuntimeError`` with a fix command when blocked."""
    job.validate()
    require_runtime(
        VIDEO_RUNTIME,
        weights=(job.model_root, job.motion_adapter),
        find_spec=find_spec,
        download_command=download_command,
    )
    job.output.parent.mkdir(parents=True, exist_ok=True)
    argv = video_command(job, python=python)
    run = runner or subprocess.run
    result = run(argv, capture_output=False)
    returncode = getattr(result, "returncode", 0)
    if returncode:
        raise MediaRuntimeError(
            VIDEO_RUNTIME,
            "failed",
            f"worker exited {returncode} for {job.output}",
            " ".join(shlex.quote(part) for part in argv),
        )
    if not job.output.exists():
        raise MediaRuntimeError(
            VIDEO_RUNTIME,
            "failed",
            f"worker reported success but {job.output} is missing",
            " ".join(shlex.quote(part) for part in argv),
        )
    return job.output


def assemble_command(
    shots: Sequence[Path],
    output: Path,
    *,
    fps: int = 8,
    transition_seconds: float = 0.5,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """ffmpeg argv that concatenates clips with an xfade between consecutive shots."""
    if not shots:
        raise ValueError("assemble needs at least one shot")
    for shot in shots:
        if not Path(shot).is_file():
            raise FileNotFoundError(f"missing shot: {shot}")
    argv = [ffmpeg, "-y", "-v", "error"]
    for shot in shots:
        argv.extend(["-i", str(shot)])
    if len(shots) == 1:
        argv.extend(["-c:v", "libx264", "-crf", "18", "-r", str(fps),
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
        return argv
    filters = []
    previous = "0:v"
    offset = 0.0
    duration = _shot_duration(shots[0])
    for index in range(1, len(shots)):
        offset += duration - transition_seconds
        label = f"v{index}"
        filters.append(
            f"[{previous}][{index}:v]xfade=transition=fade:duration={transition_seconds}"
            f":offset={offset:.3f}[{label}]"
        )
        previous = label
        duration = _shot_duration(shots[index])
    filters.append(f"[{previous}]format=yuv420p[vout]")
    argv.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-r", str(fps),
        "-movflags", "+faststart", str(output),
    ])
    return argv


def _shot_duration(shot: Path, fps: int = 8) -> float:
    try:
        import subprocess as _sp

        probe = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(shot)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(probe) / fps
    except Exception:  # ffprobe missing or unreadable file
        return 2.0


def assemble_shots(
    shots: Sequence[Path],
    output: Path,
    *,
    fps: int = 8,
    transition_seconds: float = 0.5,
    runner: Runner | None = None,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise MediaRuntimeError(
            "ffmpeg",
            WAITING_RUNTIME,
            "ffmpeg is not on PATH, so generated frames cannot be encoded",
            "sudo pacman -S ffmpeg",
        )
    argv = assemble_command(shots, output, fps=fps, transition_seconds=transition_seconds)
    run = runner or subprocess.run
    result = run(argv)
    if getattr(result, "returncode", 0):
        raise MediaRuntimeError("ffmpeg", "failed", f"ffmpeg exited {result.returncode}", " ".join(argv))
    return output


def catalog_component_paths(model: Any, models_root: str | Path) -> tuple[Path, ...]:
    """Local directories a catalog entry needs before it can generate."""
    root = Path(models_root)
    components = getattr(model, "components", ())
    if not components:
        return (root / _repo_dirname(getattr(model, "source", "")),)
    return tuple(root / _repo_dirname(source) for _, source, _ in components)


def _repo_dirname(source: str) -> str:
    return source.rstrip("/").split("huggingface.co/")[-1].replace("/", "__")


def catalog_download_command(model: Any, models_root: str | Path) -> str:
    """Copy-paste download commands for every pinned component of an entry."""
    root = Path(models_root)
    components = getattr(model, "components", ())
    if not components:
        components = (("weights", getattr(model, "source", ""), getattr(model, "revision", "")),)
    lines = []
    for _, source, revision in components:
        repo = source.rstrip("/").split("huggingface.co/")[-1]
        lines.append(
            f"hf download {repo} --revision {revision} "
            f"--local-dir {shlex.quote(str(root / _repo_dirname(source)))}"
        )
    return " && ".join(lines)


def vram_mb() -> int:
    """Total VRAM across local accelerators, 0 when nothing is detectable."""
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        return sum(int(line) for line in out.split() if line.isdigit())
    except Exception:
        return 0


def check_video_memory(required_mb: int, *, available_mb: int | None = None) -> None:
    available = vram_mb() if available_mb is None else available_mb
    if available and available < required_mb:
        raise MediaRuntimeError(
            "animatediff",
            WAITING_MEMORY,
            f"needs {required_mb} MB of accelerator memory, found {available} MB",
            "free a GPU or lower --width/--height/--frames",
        )


def evidence(runtime: str, status: RuntimeStatus, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "host": socket.gethostname(),
        "runtime": runtime,
        "status": status.as_dict(),
        **extra,
    }


def write_evidence(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
