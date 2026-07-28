from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bonsai_entrypoint_activates_dspark_draft_runtime() -> None:
    entrypoint = (ROOT / "container/capsules/bonsai-1bit/entrypoint.sh").read_text()

    assert 'DRAFT_MODEL="${DRAFT_MODEL:-}"' in entrypoint
    assert '--model-draft "$DRAFT_MODEL"' in entrypoint
    assert '--spec-type draft-dspark' in entrypoint
    assert '--spec-draft-n-max "$SPEC_DRAFT_N_MAX"' in entrypoint
    assert '-ngld "$DRAFT_NGL"' in entrypoint
