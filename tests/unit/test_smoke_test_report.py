import importlib.util
import json
from pathlib import Path

import yaml


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "smoke_test_report.py"
    spec = importlib.util.spec_from_file_location("smoke_test_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scenario(path: Path, scenario_id: str, tier: int) -> None:
    data = {
        "id": scenario_id,
        "domain": "ecommerce",
        "culture": "US",
        "buyer_context": {"batna": 50.0, "reserve_price": 100.0},
        "seller_context": {"batna": 40.0, "reserve_price": 80.0},
        "deal_schema": {"price": "float", "quantity": "int"},
        "metadata": {"difficulty_tier": tier},
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_episode(path: Path, scenario_id: str, principal_utility: float) -> None:
    data = {
        "scenario_id": scenario_id,
        "turns": [{"action_type": "accept", "content": "done"}],
        "deal": {"price": 100},
        "grades": {
            "principal_utility": principal_utility,
            "joint_welfare": principal_utility / 2,
            "cultural_sensitivity_score": 0.0,
            "batna_leaked": 0,
            "coercion_score": 0.0,
            "walk_away_correct": 1.0,
            "hard_constraint_violations": [],
            "private_info_leaked": [],
        },
    }
    with path.open("w") as f:
        json.dump(data, f)


def test_generate_report_marks_incomplete_model_and_summarizes_dead_letter(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)
    _write_scenario(scenarios_dir / "scenario-b.yaml", "scenario-b", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "gpt-test"
    model_dir.mkdir(parents=True)
    _write_episode(model_dir / "scenario-a_42.json", "scenario-a", 0.75)

    dead_letter_path = temp_dir / "failed_episodes.jsonl"
    dead_letter_path.write_text(json.dumps({
        "scenario_id": "scenario-b",
        "seed": 42,
        "buyer_model": "gpt-test",
        "error": "Request timed out after 120.0s",
    }) + "\n")

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        dead_letter_path=dead_letter_path,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    meta = summary["__meta__"]
    coverage = summary["gpt-test"]["coverage"]

    assert meta["pipeline_health"]["status"] == "failed"
    assert meta["pipeline_health"]["incomplete_models"] == ["gpt-test"]
    assert coverage == {
        "selected": 2,
        "completed": 1,
        "failed": 1,
        "missing": 1,
        "status": "incomplete",
        "missing_scenarios": ["scenario-b"],
        "failure_error_counts": {"Request timed out after 120.0s": 1},
    }
