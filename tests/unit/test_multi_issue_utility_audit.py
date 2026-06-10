from pathlib import Path

import yaml

from concord.data.seed_audit import audit_multi_issue_utility_models


def _write_seed(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_multi_issue_utility_models_reports_missing_coverage(temp_dir: Path) -> None:
    _write_seed(
        temp_dir / "saas_procurement" / "seed-with-models.yaml",
        {
            "id": "seed-with-models",
            "domain": "saas_procurement",
            "buyer_context": {
                "batna": 40.0,
                "reserve_price": 60.0,
                "issue_utilities": {"contract_length_months": {"type": "numeric", "weight": 0.2, "best": 12, "worst": 36}},
            },
            "seller_context": {
                "batna": 64.0,
                "reserve_price": 38.0,
                "issue_utilities": {"contract_length_months": {"type": "numeric", "weight": 0.2, "best": 36, "worst": 12}},
            },
            "deal_schema": {
                "monthly_price": "float",
                "contract_length_months": "int",
                "onboarding_support_hours": "int",
            },
        },
    )
    _write_seed(
        temp_dir / "settlement" / "seed-missing-models.yaml",
        {
            "id": "seed-missing-models",
            "domain": "settlement",
            "buyer_context": {"batna": 50.0, "reserve_price": 80.0},
            "seller_context": {"batna": 110.0, "reserve_price": 90.0},
            "deal_schema": {
                "settlement_amount": "float",
                "payment_structure": "str",
                "payment_timeline_days": "int",
            },
        },
    )

    report = audit_multi_issue_utility_models(temp_dir)

    assert report["multi_issue_count"] == 2
    assert report["with_explicit_issue_utility_models"] == 1
    assert report["missing_issue_utility_ids"] == ["seed-missing-models"]
    assert report["multi_issue_by_domain"] == {
        "saas_procurement": 1,
        "settlement": 1,
    }
    assert report["findings"]["all_multi_issue_seeds_have_explicit_models"] is False
