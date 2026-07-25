from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.profile_io import (
    ProfileFormatError,
    canonical_json,
    load_json_profile,
    load_profile,
    load_yaml_profile,
    profile_digest,
)
from turbofit_runtime.runtime_profile import Turbofile
from test_runtime_profile import valid_mapping


def test_json_loading_is_dependency_free_and_validates_schema(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(valid_mapping()))

    profile = load_json_profile(path)

    assert isinstance(profile, Turbofile)
    assert profile.id == "quality-24gb"


def test_json_errors_and_duplicate_keys_are_not_hidden(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        load_json_profile(invalid)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ProfileFormatError, match="duplicate key.*schema"):
        load_json_profile(duplicate)


def test_yaml_loading_validates_the_real_schema_example() -> None:
    pytest.importorskip("yaml")
    path = Path(__file__).parents[1] / "runtime-profiles" / "schema-example.yaml"

    profile = load_yaml_profile(path)

    assert profile.id == "quality-24gb"


def test_yaml_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema: one\nschema: two\n")

    with pytest.raises(ProfileFormatError, match="duplicate key.*schema"):
        load_yaml_profile(path)


def test_yaml_unsafe_python_tags_are_rejected_without_execution(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    marker = tmp_path / "must-not-exist"
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        f'!!python/object/apply:os.system ["touch {marker}"]\n',
        encoding="utf-8",
    )

    with pytest.raises(yaml.YAMLError):
        load_yaml_profile(path)
    assert not marker.exists()


def test_yaml_dependency_failure_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from turbofit_runtime import profile_io

    path = tmp_path / "profile.yaml"
    path.write_text("schema: turbofit.runtime/v1\n")

    def missing() -> object:
        raise ModuleNotFoundError("No module named 'yaml'")

    monkeypatch.setattr(profile_io, "_load_yaml_module", missing)
    with pytest.raises(ImportError, match="PyYAML.*JSON"):
        load_yaml_profile(path)


def test_unknown_fields_and_schema_mismatch_delegate_to_turbofile(tmp_path: Path) -> None:
    unknown = valid_mapping()
    unknown["surprise"] = True
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="unknown field"):
        load_profile(unknown_path)

    mismatch = valid_mapping()
    mismatch["schema"] = "turbofit.runtime/v2"
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatch))
    with pytest.raises(ValueError, match="schema"):
        load_profile(mismatch_path)


def test_canonical_serialization_and_digest_are_deterministic() -> None:
    profile = Turbofile.from_mapping(valid_mapping())

    first = canonical_json(profile)
    second = canonical_json(profile.to_mapping())

    assert first == second
    assert " " not in first
    assert canonical_json(json.loads(first)) == first
    assert profile_digest(profile) == profile_digest(profile.to_mapping())
    assert profile_digest(profile).startswith("sha256:")
    assert len(profile_digest(profile)) == len("sha256:") + 64


def test_load_profile_dispatches_suffix_and_preserves_file_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "missing.json")

    unsupported = tmp_path / "profile.toml"
    unsupported.write_text("")
    with pytest.raises(ProfileFormatError, match="unsupported profile format"):
        load_profile(unsupported)
