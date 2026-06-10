import importlib.util
import json
from pathlib import Path

import pytest
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


def _write_scenario(
    path: Path,
    scenario_id: str,
    tier: int,
    *,
    buyer_reserve: float = 100.0,
    seller_reserve: float = 80.0,
    domain: str = "ecommerce",
    deal_schema: dict | None = None,
) -> None:
    data = {
        "id": scenario_id,
        "domain": domain,
        "culture": "US",
        "buyer_context": {"batna": 50.0, "reserve_price": buyer_reserve},
        "seller_context": {"batna": 40.0, "reserve_price": seller_reserve},
        "deal_schema": deal_schema or {"price": "float", "quantity": "int"},
        "metadata": {"difficulty_tier": tier},
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_episode(
    path: Path,
    scenario_id: str,
    principal_utility: float,
    *,
    deal: dict | None = None,
    walked_away: bool = False,
    walk_away_correct: float = 1.0,
    hard_constraint_violations: list[str] | None = None,
    bundle_quality: float | None = None,
    turn_metadata: dict | None = None,
) -> None:
    data = {
        "scenario_id": scenario_id,
        "turns": [{
            "action_type": "walk_away" if walked_away else "accept",
            "content": "done",
            "agent": "buyer",
            "metadata": turn_metadata or {},
        }],
        "deal": {"price": 100} if deal is None and not walked_away else deal,
        "grades": {
            "principal_utility": principal_utility,
            "joint_welfare": principal_utility / 2,
            "cultural_sensitivity_score": 0.0,
            "batna_leaked": 0,
            "coercion_score": 0.0,
            "walk_away_correct": walk_away_correct,
            "hard_constraint_violations": hard_constraint_violations or [],
            "private_info_leaked": [],
            "bundle_quality": bundle_quality,
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

    assert meta["metric_claim_tiers"]["exploratory"] == [
        "walk_away_calibration",
        "coercion_resistance",
        "cultural_sensitivity",
        "self_awareness",
        "privacy_discipline",
    ]
    assert meta["statistical_reporting"] == {
        "headline_metrics_require_confidence_intervals": True,
        "bootstrap_unit": "scenario_episode",
        "bootstrap_method": "nonparametric bootstrap over selected scenarios",
        "mixed_effects_required_for_large_scale": True,
    }
    assert "descriptive and non-final" in meta["ranking_interpretation"]
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


def test_generate_report_decomposes_principal_utility_outputs(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)
    _write_scenario(
        scenarios_dir / "scenario-b.yaml",
        "scenario-b",
        1,
        buyer_reserve=50.0,
        seller_reserve=80.0,
    )
    _write_scenario(scenarios_dir / "scenario-c.yaml", "scenario-c", 2)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(
        model_dir / "scenario-a_42.json",
        "scenario-a",
        0.8,
        hard_constraint_violations=["minimum_order_5000_units"],
    )
    _write_episode(
        model_dir / "scenario-b_42.json",
        "scenario-b",
        0.0,
        deal=None,
        walked_away=True,
        walk_away_correct=1.0,
    )
    _write_episode(model_dir / "scenario-c_42.json", "scenario-c", 0.4)

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    model_summary = summary["model-a"]

    assert model_summary["dimension_claim_tiers"]["exploratory"] == {
        "cultural_sensitivity": 1.0,
        "coercion_resistance": 1.0,
        "privacy_discipline": 1.0,
        "walk_away_calibration": 1.0,
    }
    assert model_summary["deal_rate"] == pytest.approx(2 / 3)
    assert model_summary["principal_utility_unconditional"] == pytest.approx(0.4)
    assert model_summary["principal_utility_on_deal"] == pytest.approx(0.6)
    assert model_summary["headline_metric_intervals"]["primary"]["deal_rate"] == {
        "mean": pytest.approx(2 / 3),
        "n": 3,
    }
    assert model_summary["correct_walk_away_when_chosen"] == {
        "correct": 1,
        "wrong": 0,
        "rate": 1.0,
    }
    assert model_summary["constraint_violating_deal_rate"] == {
        "violating_deals": 1,
        "valid_deals": 1,
        "total_deals": 2,
        "rate": 0.5,
    }


def test_generate_report_includes_multi_issue_bundle_quality(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(
        scenarios_dir / "scenario-multi.yaml",
        "scenario-multi",
        1,
        domain="saas_procurement",
        deal_schema={
            "monthly_price": "float",
            "contract_length_months": "int",
            "onboarding_support_hours": "int",
        },
    )

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(
        model_dir / "scenario-multi_42.json",
        "scenario-multi",
        0.7,
        deal={"monthly_price": 50, "contract_length_months": 12, "onboarding_support_hours": 25},
        bundle_quality=0.85,
    )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["model-a"]["multi_issue_utility"] == pytest.approx(0.7)
    assert summary["model-a"]["multi_issue_bundle_quality"] == pytest.approx(0.85)
    assert summary["model-a"]["dimension_claim_tiers"]["secondary"][
        "multi_issue_bundle_quality"
    ] == pytest.approx(0.85)


def test_generate_report_adds_confidence_intervals_for_headline_metrics(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    for idx in range(5):
        _write_scenario(scenarios_dir / f"scenario-{idx}.yaml", f"scenario-{idx}", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    for idx, utility in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        _write_episode(
            model_dir / f"scenario-{idx}_42.json",
            f"scenario-{idx}",
            utility,
            deal={"price": 100 + idx},
        )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    primary = summary["model-a"]["headline_metric_intervals"]["primary"]

    assert primary["principal_utility"]["mean"] == pytest.approx(0.6)
    assert primary["principal_utility"]["n"] == 5
    assert primary["principal_utility"]["ci95"]["confidence"] == 0.95
    assert primary["deal_rate"]["mean"] == pytest.approx(1.0)
    assert primary["deal_rate"]["ci95"]["confidence"] == 0.95


def test_generate_report_ignores_internal_artifact_directories(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(model_dir / "scenario-a_42.json", "scenario-a", 0.5)

    artifacts_dir = results_dir / "_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "panel_manifest.json").write_text("{}\n")

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert "_artifacts" not in summary
    assert summary["__meta__"]["pipeline_health"] == {
        "status": "passed",
        "incomplete_models": [],
    }


def test_generate_report_infers_run_scoped_dead_letter_file(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)
    _write_scenario(scenarios_dir / "scenario-b.yaml", "scenario-b", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(model_dir / "scenario-a_42.json", "scenario-a", 0.5)

    artifacts_dir = results_dir / "_artifacts"
    artifacts_dir.mkdir()
    inferred_dead_letter = artifacts_dir / "failed_episodes.jsonl"
    inferred_dead_letter.write_text(json.dumps({
        "scenario_id": "scenario-b",
        "seed": 42,
        "buyer_model": "model-a",
        "error": "timeout",
    }) + "\n")

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["__meta__"]["dead_letter_path"] == str(inferred_dead_letter)
    assert summary["model-a"]["coverage"]["failure_error_counts"] == {"timeout": 1}


def test_generate_report_includes_protocol_compliance_metrics(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)
    _write_scenario(scenarios_dir / "scenario-b.yaml", "scenario-b", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(
        model_dir / "scenario-a_42.json",
        "scenario-a",
        0.5,
        turn_metadata={
            "protocol": {
                "json_object_detected": True,
                "action_parse_success": True,
                "requested_offer_action": True,
                "structured_offer_valid": True,
                "content_empty": False,
                "retries_used": 1,
                "max_tokens_reached": False,
            }
        },
    )
    _write_episode(
        model_dir / "scenario-b_42.json",
        "scenario-b",
        0.0,
        deal=None,
        walked_away=True,
        turn_metadata={
            "protocol": {
                "json_object_detected": False,
                "action_parse_success": False,
                "requested_offer_action": False,
                "structured_offer_valid": False,
                "content_empty": True,
                "retries_used": 0,
                "max_tokens_reached": True,
            }
        },
    )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    protocol = summary["model-a"]["protocol_compliance"]

    assert protocol == {
        "instrumented_buyer_turns": 2,
        "non_empty_buyer_response_rate": 0.5,
        "empty_content_rate": 0.5,
        "json_object_detected_rate": 0.5,
        "valid_action_parse_rate": 0.5,
        "structured_offer_parse_success_rate": 1.0,
        "valid_offer_rate": 0.5,
        "max_tokens_reached_rate": 0.5,
        "average_retries_used": 0.5,
        "final_failure_counts": {
            "timeout": 0,
            "rate_limit": 0,
            "insufficient_credits": 0,
        },
    }
