from pathlib import Path

import yaml

from concord.data.seed_audit import audit_t0_slice


def _write_seed(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_t0_slice_reports_domain_coverage_and_risk_flags(temp_dir: Path) -> None:
    _write_seed(
        temp_dir / "ecommerce" / "seed-t0-risky.yaml",
        {
            "id": "seed-t0-risky",
            "domain": "ecommerce",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 100.0,
                "hard_constraints": ["quality_meets_standards"],
                "walk_away_threshold": None,
                "private_info": ["budget_is_120"],
            },
            "seller_context": {
                "batna": 40.0,
                "reserve_price": 80.0,
                "hard_constraints": ["minimum_order_100_units"],
                "walk_away_threshold": None,
            },
            "deal_schema": {"price": "float", "quantity": "int"},
            "scenario_description": (
                "Routine reorder with available inventory and quick close expected."
            ),
            "metadata": {"difficulty_tier": 0},
        },
    )
    _write_seed(
        temp_dir / "settlement" / "seed-t0-safe.yaml",
        {
            "id": "seed-t0-safe",
            "domain": "settlement",
            "buyer_context": {
                "batna": 50000.0,
                "reserve_price": 90000.0,
                "hard_constraints": ["confidentiality_clause_required"],
                "walk_away_threshold": None,
                "private_info": ["client_authorized_range_70000_to_85000"],
            },
            "seller_context": {
                "batna": 30000.0,
                "reserve_price": 65000.0,
                "hard_constraints": ["no_admission_of_liability"],
                "walk_away_threshold": None,
            },
            "deal_schema": {"settlement_amount": "float", "confidentiality_clause": "bool"},
            "scenario_description": (
                "Routine settlement where both parties already agree on confidentiality and "
                "the agent can close anywhere inside the authorized range without extra restatement."
            ),
            "metadata": {"difficulty_tier": 0},
        },
    )

    report = audit_t0_slice(temp_dir)

    assert report["t0_count"] == 2
    assert report["t0_by_domain"] == {"ecommerce": 1, "settlement": 1}
    assert report["restatement_risk_ids"] == ["seed-t0-risky"]
    assert report["findings"]["multi_domain_t0_met"] is False
    assert report["findings"]["all_t0s_low_restatement_risk"] is False

