from __future__ import annotations

from pathlib import Path

import pytest

from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.wiki import WikiPublisher, validate_profile_evidence


ROOT = Path(__file__).parents[1]


def make_wiki(tmp_path: Path) -> Path:
    topic = tmp_path / "topics" / "turbofit"
    topic.mkdir(parents=True)
    (topic / "README.md").write_text("# Turbofit\n\n## Key Paths\n\n- existing\n")
    (topic / "main-aux-inference-checklist.md").write_text(
        "# Checklist\n\n## Checklist\n\n- [ ] existing\n"
    )
    return tmp_path


def test_generated_views_are_deterministic_and_bidirectionally_linked(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    publisher = WikiPublisher(ROOT, wiki)

    first = publisher.render()
    second = publisher.render()

    assert first == second
    readme = first[wiki / "topics" / "turbofit" / "README.md"]
    checklist = first[wiki / "topics" / "turbofit" / "main-aux-inference-checklist.md"]
    assert "Generated Adaptive Recommendations" in readme
    assert "main-aux-inference-checklist.md" in readme
    assert "README.md" in checklist
    assert "| 24 GB | `1x24` | measured-winner |" in readme
    assert "| 48 GB | `2x24` | measured-winner |" in readme


def test_publish_is_idempotent_and_preserves_unmanaged_content(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    publisher = WikiPublisher(ROOT, wiki)

    changed = publisher.publish()
    first = {path: path.read_text() for path in changed}
    changed_again = publisher.publish()

    assert len(changed) == 2
    assert changed_again == ()
    assert all("existing" in content for content in first.values())


def test_unresolved_or_unchecked_evidence_is_rejected(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    publisher = WikiPublisher(ROOT, wiki)
    with pytest.raises(ValueError, match="missing evidence source"):
        publisher.validate_evidence_index({"sha256:" + "a" * 64: {"source": "missing.md"}})

    measured = load_yaml_profile(ROOT / "runtime-profiles" / "24gb.yaml")
    with pytest.raises(ValueError, match="unresolved evidence"):
        validate_profile_evidence(measured, set())


def test_candidate_queue_is_rendered_from_canonical_data(tmp_path: Path) -> None:
    wiki = make_wiki(tmp_path)
    readme = WikiPublisher(ROOT, wiki).render()[wiki / "topics" / "turbofit" / "README.md"]

    assert "Candidate queue" in readme
    assert "Canonical source: `research/candidates.json`" in readme
