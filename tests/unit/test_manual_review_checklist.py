from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


def _load_manual_review_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "manual_review_checklist.py"
    spec = importlib.util.spec_from_file_location("manual_review_checklist", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scenarios_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir()
    for i in range(3):
        scenario_yaml = {
            "id": f"gen-s{i}",
            "domain": "ecommerce",
            "description": f"Scenario {i}",
        }
        (directory / f"gen-s{i}.yaml").write_text(yaml.safe_dump(scenario_yaml))
    return directory


def test_scaffold_creates_one_yaml_per_scenario(scenarios_dir: Path, tmp_path: Path) -> None:
    module = _load_manual_review_module()
    output_dir = tmp_path / "reviews"
    module.scaffold_review_files(scenarios_dir, output_dir)
    review_files = list(output_dir.glob("*.yaml"))
    assert len(review_files) == 3


def test_scaffolded_yaml_has_required_keys(scenarios_dir: Path, tmp_path: Path) -> None:
    module = _load_manual_review_module()
    output_dir = tmp_path / "reviews"
    module.scaffold_review_files(scenarios_dir, output_dir)
    review = yaml.safe_load((output_dir / "gen-s0.yaml").read_text())
    assert "scenario_id" in review
    assert "reviewer" in review
    assert "review_date" in review
    assert "structurally_coherent" in review
    assert "publication_worthy" in review
    assert "domain_realistic" in review
    assert "notes" in review


def test_check_completeness_passes_when_all_filled(tmp_path: Path) -> None:
    module = _load_manual_review_module()
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    for i in range(2):
        review = {
            "scenario_id": f"gen-s{i}",
            "reviewer": "deepak",
            "review_date": "2026-06-14",
            "structurally_coherent": True,
            "publication_worthy": True,
            "domain_realistic": True,
            "notes": "looks good",
        }
        (review_dir / f"gen-s{i}.yaml").write_text(yaml.safe_dump(review))
    missing = module.check_review_completeness(review_dir)
    assert missing == []


def test_check_completeness_flags_unfilled_reviews(tmp_path: Path) -> None:
    module = _load_manual_review_module()
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    review = {
        "scenario_id": "gen-s0",
        "reviewer": None,
        "review_date": None,
        "structurally_coherent": None,
        "publication_worthy": None,
        "domain_realistic": None,
        "notes": "",
    }
    (review_dir / "gen-s0.yaml").write_text(yaml.safe_dump(review))
    missing = module.check_review_completeness(review_dir)
    assert "gen-s0" in missing
