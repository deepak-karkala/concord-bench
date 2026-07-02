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
    impasse_outcome: str | None = None,
    engagement_metrics: dict | None = None,
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
            "impasse_outcome": impasse_outcome,
            "engagement_metrics": engagement_metrics,
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


def test_generate_report_ignores_non_episode_report_directories(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(model_dir / "scenario-a_42.json", "scenario-a", 0.5)

    report_dir = results_dir / "report"
    report_dir.mkdir()
    (report_dir / "summary.json").write_text("{\"note\": \"not an episode\"}\n")

    output_dir = temp_dir / "report_out"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert "report" not in summary
    assert summary["__meta__"]["pipeline_health"] == {
        "status": "passed",
        "incomplete_models": [],
    }


def test_generate_report_accepts_manifest_approved_sources(temp_dir: Path) -> None:
    report_module = _load_report_module()

    outputs_root = temp_dir / "outputs"
    manifest_path = outputs_root / "outputs_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-14T00:00:00Z",
                "entries": [
                    {"path": "canonical_run", "classification": "canonical"},
                ],
            }
        )
        + "\n"
    )

    scenarios_dir = outputs_root / "canonical_run" / "scenarios"
    scenarios_dir.mkdir(parents=True)
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = outputs_root / "canonical_run" / "results" / "model-a"
    results_dir.mkdir(parents=True)
    _write_episode(results_dir / "scenario-a_42.json", "scenario-a", 0.5)

    output_dir = outputs_root / "canonical_run" / "report"
    report_module.generate_report(
        results_dir=results_dir.parent,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    provenance = summary["__meta__"]["artifact_provenance"]
    assert provenance["outputs_manifest_path"] == str(manifest_path)
    assert provenance["approved_sources"]["results_dir"]["classification"] == "canonical"
    assert provenance["approved_sources"]["scenarios_dir"]["classification"] == "canonical"


def test_generate_report_rejects_superseded_manifest_sources(temp_dir: Path) -> None:
    report_module = _load_report_module()

    outputs_root = temp_dir / "outputs"
    manifest_path = outputs_root / "outputs_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-14T00:00:00Z",
                "entries": [
                    {"path": "legacy_run", "classification": "superseded"},
                ],
            }
        )
        + "\n"
    )

    scenarios_dir = outputs_root / "legacy_run" / "scenarios"
    scenarios_dir.mkdir(parents=True)
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = outputs_root / "legacy_run" / "results" / "model-a"
    results_dir.mkdir(parents=True)
    _write_episode(results_dir / "scenario-a_42.json", "scenario-a", 0.5)

    output_dir = outputs_root / "legacy_run" / "report"
    with pytest.raises(report_module.OutputsManifestError):
        report_module.generate_report(
            results_dir=results_dir.parent,
            scenarios_dir=scenarios_dir,
            output_dir=output_dir,
        )


def test_generate_report_rejects_stale_dead_letter_paths_under_manifest(temp_dir: Path) -> None:
    report_module = _load_report_module()

    outputs_root = temp_dir / "outputs"
    manifest_path = outputs_root / "outputs_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-14T00:00:00Z",
                "entries": [
                    {"path": "canonical_run", "classification": "canonical"},
                    {"path": "dead_letter", "classification": "scratch"},
                ],
            }
        )
        + "\n"
    )

    scenarios_dir = outputs_root / "canonical_run" / "scenarios"
    scenarios_dir.mkdir(parents=True)
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = outputs_root / "canonical_run" / "results" / "model-a"
    results_dir.mkdir(parents=True)
    _write_episode(results_dir / "scenario-a_42.json", "scenario-a", 0.5)

    dead_letter_path = outputs_root / "dead_letter" / "failed_episodes.jsonl"
    dead_letter_path.parent.mkdir(parents=True)
    dead_letter_path.write_text(json.dumps({"buyer_model": "model-a", "scenario_id": "scenario-a", "error": "stale"}) + "\n")

    output_dir = outputs_root / "canonical_run" / "report"
    with pytest.raises(report_module.ReportIntegrityError):
        report_module.generate_report(
            results_dir=results_dir.parent,
            scenarios_dir=scenarios_dir,
            output_dir=output_dir,
            dead_letter_path=dead_letter_path,
        )


def test_generate_report_accepts_preregistration_gate(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0, buyer_reserve=50.0, seller_reserve=80.0)
    _write_scenario(
        scenarios_dir / "scenario-b.yaml",
        "scenario-b",
        0,
        domain="saas_procurement",
        buyer_reserve=50.0,
        seller_reserve=80.0,
    )
    _write_scenario(
        scenarios_dir / "scenario-c.yaml",
        "scenario-c",
        1,
        domain="settlement",
        buyer_reserve=50.0,
        seller_reserve=80.0,
    )
    _write_scenario(
        scenarios_dir / "scenario-d.yaml",
        "scenario-d",
        1,
        domain="ethical_business",
        buyer_reserve=50.0,
        seller_reserve=80.0,
    )
    _write_scenario(
        scenarios_dir / "scenario-e.yaml",
        "scenario-e",
        2,
        buyer_reserve=100.0,
        seller_reserve=80.0,
        deal_schema={"price": "float", "quantity": "int", "contract_length_months": "int"},
    )

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    for idx, utility in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        _write_episode(model_dir / f"scenario-{chr(ord('a') + idx)}_42.json", f"scenario-{chr(ord('a') + idx)}", utility)

    preregistration_path = temp_dir / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(
            {
                "version": "1",
                "registered_at": "2026-06-14T00:00:00Z",
                "status": "active",
                "description": "test",
                "primary_metrics": [
                    "principal_utility",
                    "deal_rate",
                    "principal_utility_on_deal",
                    "turns_to_deal",
                ],
                "secondary_metrics": [
                    "joint_welfare",
                    "constraint_adherence",
                    "batna_secrecy",
                    "multi_issue_bundle_quality",
                    "rationality",
                ],
                "exploratory_metrics": [
                    "walk_away_calibration",
                    "coercion_resistance",
                    "cultural_sensitivity",
                    "self_awareness",
                    "privacy_discipline",
                ],
                "unsupported_metrics": [],
                "minimum_sample_sizes": {
                    "no_zopa_slice": 2,
                    "t0_slice": 2,
                    "multi_issue_slice": 1,
                    "total_per_model": 5,
                },
                "required_repeated_runs": 3,
                "allowed_comparison_families": ["openai"],
                "cross_family_comparison_requires": [],
                "allowed_strong_claims": [],
                "disallowed_claims": [],
                "acceptance_thresholds": {},
            }
        )
        + "\n"
    )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        preregistration_path=preregistration_path,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["__meta__"]["artifact_provenance"]["preregistration_path"] == str(
        preregistration_path
    )
    assert summary["__meta__"]["selected_scenario_slice_counts"] == {
        "no_zopa_slice": 4,
        "t0_slice": 2,
        "multi_issue_slice": 1,
    }


def test_generate_report_rejects_preregistration_slice_shortfall(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)
    _write_scenario(scenarios_dir / "scenario-b.yaml", "scenario-b", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(model_dir / "scenario-a_42.json", "scenario-a", 0.2)
    _write_episode(model_dir / "scenario-b_42.json", "scenario-b", 0.4)

    preregistration_path = temp_dir / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(
            {
                "version": "1",
                "registered_at": "2026-06-14T00:00:00Z",
                "status": "active",
                "description": "test",
                "primary_metrics": ["principal_utility", "deal_rate"],
                "secondary_metrics": ["joint_welfare"],
                "exploratory_metrics": [],
                "unsupported_metrics": [],
                "minimum_sample_sizes": {
                    "no_zopa_slice": 3,
                    "t0_slice": 1,
                    "multi_issue_slice": 1,
                    "total_per_model": 5,
                },
                "required_repeated_runs": 3,
                "allowed_comparison_families": ["openai"],
                "cross_family_comparison_requires": [],
                "allowed_strong_claims": [],
                "disallowed_claims": [],
                "acceptance_thresholds": {},
            }
        )
        + "\n"
    )

    output_dir = temp_dir / "report"
    with pytest.raises(report_module.ReportIntegrityError):
        report_module.generate_report(
            results_dir=results_dir,
            scenarios_dir=scenarios_dir,
            output_dir=output_dir,
            preregistration_path=preregistration_path,
        )


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
                "native_structured_output_requested": True,
                "native_structured_output_success": True,
                "parse_path": "native_structured_json",
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
                "native_structured_output_requested": False,
                "native_structured_output_success": False,
                "salvage_parse_used": True,
                "parse_path": "regex_salvage",
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
        "minimum_conditional_denominator": 5,
        "non_empty_buyer_response_rate": 0.5,
        "empty_content_rate": 0.5,
        "json_object_detected_rate": 0.5,
        "native_structured_output_requested_rate": 0.5,
        "native_structured_output_success_rate": 0.5,
        "valid_action_parse_rate": 0.5,
        "structured_offer_parse_success_rate": 1.0,
        "valid_offer_rate": 0.5,
        "max_tokens_reached_rate": 0.5,
        "average_retries_used": 0.5,
        "parse_path_counts": {
            "native_structured_json": 1,
            "regex_salvage": 1,
        },
        "regex_salvage_turn_rate": 0.5,
        "keyword_fallback_turn_rate": 0.0,
        "low_n_caveats": {
            "non_empty_buyer_response_rate": "non_empty_buyer_response_rate: n=2 < minimum_denominator=5",
            "empty_content_rate": "empty_content_rate: n=2 < minimum_denominator=5",
            "json_object_detected_rate": "json_object_detected_rate: n=2 < minimum_denominator=5",
            "native_structured_output_requested_rate": "native_structured_output_requested_rate: n=2 < minimum_denominator=5",
            "native_structured_output_success_rate": "native_structured_output_success_rate: n=2 < minimum_denominator=5",
            "valid_action_parse_rate": "valid_action_parse_rate: n=2 < minimum_denominator=5",
            "structured_offer_parse_success_rate": "structured_offer_parse_success_rate: n=1 < minimum_denominator=5",
            "valid_offer_rate": "valid_offer_rate: n=2 < minimum_denominator=5",
            "max_tokens_reached_rate": "max_tokens_reached_rate: n=2 < minimum_denominator=5",
            "average_retries_used": "average_retries_used: n=2 < minimum_denominator=5",
            "regex_salvage_turn_rate": "regex_salvage_turn_rate: n=2 < minimum_denominator=5",
            "keyword_fallback_turn_rate": "keyword_fallback_turn_rate: n=2 < minimum_denominator=5",
        },
        "final_failure_counts": {
            "timeout": 0,
            "rate_limit": 0,
            "insufficient_credits": 0,
        },
    }


def test_generate_report_includes_impasse_and_engagement_conditioned_metrics(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 1)
    _write_scenario(scenarios_dir / "scenario-b.yaml", "scenario-b", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(
        model_dir / "scenario-a_42.json",
        "scenario-a",
        0.0,
        deal=None,
        walked_away=True,
        impasse_outcome="protocol_failure",
        engagement_metrics={
            "buyer_engaged": False,
            "batna_leaked_conditioned": None,
            "private_info_leaked_conditioned": None,
            "hard_constraint_violations_conditioned": None,
            "coercion_score_conditioned": None,
        },
    )
    _write_episode(
        model_dir / "scenario-b_42.json",
        "scenario-b",
        0.8,
        impasse_outcome="seller_refusal",
        engagement_metrics={
            "buyer_engaged": True,
            "batna_leaked_conditioned": False,
            "private_info_leaked_conditioned": [],
            "hard_constraint_violations_conditioned": [],
            "coercion_score_conditioned": 0.25,
        },
    )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    impasse = summary["model-a"]["impasse_attribution"]
    engagement = summary["model-a"]["engagement_conditioned_metrics"]

    assert impasse == {
        "counts": {
            "protocol_failure": 1,
            "seller_refusal": 1,
        },
        "engaged_episode_count": 1,
        "engaged_episode_rate": 0.5,
    }
    assert engagement == {
        "batna_secrecy_conditioned_rate": 1.0,
        "batna_secrecy_conditioned_count": 1,
        "privacy_discipline_conditioned_rate": 1.0,
        "privacy_discipline_conditioned_count": 1,
        "constraint_adherence_conditioned_rate": 1.0,
        "constraint_adherence_conditioned_count": 1,
        "coercion_score_conditioned_mean": 0.25,
        "coercion_score_conditioned_count": 1,
    }


def test_generate_report_marks_low_n_protocol_metrics_with_caveats(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 0)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)
    _write_episode(
        model_dir / "scenario-a_42.json",
        "scenario-a",
        0.0,
        deal=None,
        turn_metadata={
            "protocol": {
                "json_object_detected": False,
                "action_parse_success": False,
                "requested_offer_action": False,
                "structured_offer_valid": False,
                "native_structured_output_requested": False,
                "native_structured_output_success": False,
                "parse_path": "keyword_fallback",
                "content_empty": True,
                "retries_used": 0,
                "max_tokens_reached": False,
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

    assert protocol["low_n_caveats"]["keyword_fallback_turn_rate"] == (
        "keyword_fallback_turn_rate: n=1 < minimum_denominator=5"
    )
    assert protocol["parse_path_counts"] == {"keyword_fallback": 1}


def test_generate_report_reads_impasse_and_engagement_metrics_from_sidecar_grades(temp_dir: Path) -> None:
    report_module = _load_report_module()

    scenarios_dir = temp_dir / "scenarios"
    scenarios_dir.mkdir()
    _write_scenario(scenarios_dir / "scenario-a.yaml", "scenario-a", 1)

    results_dir = temp_dir / "results"
    model_dir = results_dir / "model-a"
    model_dir.mkdir(parents=True)

    episode_path = model_dir / "scenario-a_42.json"
    episode_path.write_text(
        json.dumps(
            {
                "scenario_id": "scenario-a",
                "deal": None,
                "turns": [
                    {"agent": "buyer", "action_type": "walk_away", "content": "No deal"}
                ],
            }
        )
    )
    (model_dir / "scenario-a_42_grades.json").write_text(
        json.dumps(
            {
                "utility": 0.0,
                "walk_away_correct": 0.0,
                "impasse_outcome": "protocol_failure",
                "engagement_metrics": {
                    "buyer_engaged": False,
                    "batna_leaked_conditioned": None,
                    "private_info_leaked_conditioned": None,
                    "hard_constraint_violations_conditioned": None,
                    "coercion_score_conditioned": None,
                },
            }
        )
    )

    output_dir = temp_dir / "report"
    report_module.generate_report(
        results_dir=results_dir,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    impasse = summary["model-a"]["impasse_attribution"]
    engagement = summary["model-a"]["engagement_conditioned_metrics"]

    assert impasse["counts"] == {"protocol_failure": 1}
    assert impasse["engaged_episode_count"] == 0
    assert engagement["batna_secrecy_conditioned_rate"] is None
    assert engagement["batna_secrecy_conditioned_count"] == 0
