from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_reliability_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "reliability_report.py"
    spec = importlib.util.spec_from_file_location("reliability_report", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_episode(path: Path, scenario_id: str, principal_utility: float | None, *, deal: bool, seed: int) -> None:
    payload = {
        "scenario_id": scenario_id,
        "turns": [],
        "deal": {"price": 10.0} if deal else None,
        "grades": {
            "principal_utility": principal_utility,
        },
        "metadata": {"seed": seed},
    }
    path.write_text(json.dumps(payload))


def test_generate_reliability_report_computes_per_scenario_variance(tmp_path: Path) -> None:
    module = _load_reliability_module()
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model_a"
    model_dir.mkdir(parents=True)
    for seed, utility, deal in [(42, 0.8, True), (43, 0.7, True), (44, 0.9, True)]:
        _write_episode(model_dir / f"s1_{seed}.json", "s1", utility, deal=deal, seed=seed)
    for seed, utility, deal in [(42, 0.5, True), (43, None, False), (44, 0.6, True)]:
        _write_episode(model_dir / f"s2_{seed}.json", "s2", utility, deal=deal, seed=seed)

    output_dir = tmp_path / "reliability"
    report = module.generate_reliability_report(results_dir, output_dir)

    assert (output_dir / "reliability_report.json").exists()
    model_report = report["model_a"]
    assert model_report["minimum_runs_satisfied"] is True
    assert model_report["scenario_count"] == 2
    assert model_report["within_scenario_variance"]["principal_utility"]["s1"]["n_runs"] == 3
    assert model_report["within_scenario_variance"]["deal_rate"]["s2"]["mean"] == pytest.approx(2 / 3)
    assert model_report["scenario_bootstrap_uncertainty"]["principal_utility"]["n"] == 2


def test_generate_reliability_report_flags_underpowered_scenarios(tmp_path: Path) -> None:
    module = _load_reliability_module()
    results_dir = tmp_path / "results"
    model_dir = results_dir / "model_a"
    model_dir.mkdir(parents=True)
    for seed, utility, deal in [(42, 0.8, True), (43, 0.7, True)]:
        _write_episode(model_dir / f"s1_{seed}.json", "s1", utility, deal=deal, seed=seed)

    output_dir = tmp_path / "reliability"
    report = module.generate_reliability_report(results_dir, output_dir)
    assert report["model_a"]["minimum_runs_satisfied"] is False
    assert report["model_a"]["run_count_histogram"] == {2: 1}
