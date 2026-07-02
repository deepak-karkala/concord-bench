from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_prereg_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "src" / "concord" / "analysis" / "preregistration.py"
    spec = importlib.util.spec_from_file_location("preregistration", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_preregistration_parses_known_metrics(tmp_path: Path) -> None:
    module = _load_prereg_module()
    prereg_path = tmp_path / "preregistration.json"
    prereg_path.write_text(
        json.dumps(
            {
                "version": "1",
                "registered_at": "2026-06-14T00:00:00+00:00",
                "status": "active",
                "description": "test",
                "primary_metrics": ["principal_utility"],
                "secondary_metrics": ["deal_rate"],
                "exploratory_metrics": ["privacy_discipline"],
                "unsupported_metrics": ["cultural_sensitivity"],
                "minimum_sample_sizes": {"total_per_model": 10},
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

    prereg = module.load_preregistration(prereg_path)
    assert prereg.primary_metrics == ["principal_utility"]
    assert prereg.required_repeated_runs == 3
    assert prereg.category_for_metric("principal_utility") == "primary"
    assert prereg.category_for_metric("privacy_discipline") == "exploratory"


def test_validate_metric_status_rejects_unknown_and_unsupported(tmp_path: Path) -> None:
    module = _load_prereg_module()
    prereg = module.PreRegistration.model_validate(
        {
            "version": "1",
            "registered_at": "2026-06-14T00:00:00+00:00",
            "status": "active",
            "description": "test",
            "primary_metrics": ["principal_utility"],
            "secondary_metrics": ["deal_rate"],
            "exploratory_metrics": ["privacy_discipline"],
            "unsupported_metrics": ["cultural_sensitivity"],
            "minimum_sample_sizes": {"total_per_model": 10},
            "required_repeated_runs": 3,
            "allowed_comparison_families": ["openai"],
            "cross_family_comparison_requires": [],
            "allowed_strong_claims": [],
            "disallowed_claims": [],
            "acceptance_thresholds": {},
        }
    )

    module.validate_metric_status("principal_utility", prereg)
    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_metric_status("invented_metric", prereg)
    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_metric_status("cultural_sensitivity", prereg)


def test_validate_report_against_preregistration_checks_tiers_and_sizes() -> None:
    module = _load_prereg_module()
    prereg = module.PreRegistration.model_validate(
        {
            "version": "1",
            "registered_at": "2026-06-14T00:00:00+00:00",
            "status": "active",
            "description": "test",
            "primary_metrics": ["principal_utility", "deal_rate"],
            "secondary_metrics": ["joint_welfare"],
            "exploratory_metrics": ["privacy_discipline"],
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

    module.validate_report_against_preregistration(
        metric_claim_tiers={
            "primary": ["principal_utility", "deal_rate"],
            "secondary": ["joint_welfare"],
            "exploratory": ["privacy_discipline"],
        },
        slice_counts={
            "no_zopa_slice": 4,
            "t0_slice": 2,
            "multi_issue_slice": 1,
            "total_per_model": 5,
        },
        preregistration=prereg,
    )

    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_report_against_preregistration(
            metric_claim_tiers={"primary": ["privacy_discipline"]},
            slice_counts={"total_per_model": 5},
            preregistration=prereg,
        )

    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_report_against_preregistration(
            metric_claim_tiers={"primary": ["principal_utility"]},
            slice_counts={"total_per_model": 4},
            preregistration=prereg,
        )

    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_report_against_preregistration(
            metric_claim_tiers={"exploratory": ["principal_utility"]},
            slice_counts={"total_per_model": 5},
            preregistration=prereg,
        )

    with pytest.raises(module.PreRegistrationViolationError):
        module.validate_report_against_preregistration(
            metric_claim_tiers={"exploratory": ["joint_welfare"]},
            slice_counts={"total_per_model": 5},
            preregistration=prereg,
        )
