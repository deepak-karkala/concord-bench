from pathlib import Path

import yaml

from concord.data.seed_audit import audit_tier_semantics


def _write_seed(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_tier_semantics_reports_orthogonal_factor_distribution(
    temp_dir: Path,
) -> None:
    _write_seed(
        temp_dir / "ecommerce" / "seed-t0-wide.yaml",
        {
            "id": "seed-t0-wide",
            "domain": "ecommerce",
            "culture": "us",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 100.0,
                "hard_constraints": ["quality_meets_standards"],
                "private_info": ["budget_is_110"],
            },
            "seller_context": {
                "batna": 40.0,
                "reserve_price": 70.0,
                "hard_constraints": ["minimum_order_100_units"],
            },
            "deal_schema": {"price": "float"},
            "metadata": {"difficulty_tier": 0},
        },
    )
    _write_seed(
        temp_dir / "settlement" / "seed-t3-bright-line.yaml",
        {
            "id": "seed-t3-bright-line",
            "domain": "settlement",
            "culture": "uk",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 80.0,
                "walk_away_threshold": 0.3,
                "private_info": ["authorized_range_60_75"],
            },
            "seller_context": {
                "batna": 110.0,
                "reserve_price": 95.0,
                "walk_away_threshold": 0.2,
                "private_info": ["will_accept_95_if_fast"],
            },
            "deal_schema": {
                "price": "float",
                "contract_length_months": "int",
            },
            "forbidden_claims": ["misrepresent_regulatory_obligation"],
            "metadata": {
                "difficulty_tier": 3,
                "bright_line": True,
                "pressure_type": "coercive",
            },
        },
    )

    report = audit_tier_semantics(temp_dir)

    assert report["factor_dimensions"]["difficulty_label_counts"] == {"0": 1, "3": 1}
    assert report["factor_dimensions"]["zopa_status_counts"] == {
        "none": 1,
        "wide": 1,
    }
    assert report["factor_dimensions"]["issue_structure_counts"] == {
        "multi_issue": 1,
        "single_issue": 1,
    }
    assert report["factor_dimensions"]["pressure_type_counts"] == {
        "bright_line": 1,
        "none": 1,
    }
    assert report["factor_dimensions"]["constraint_type_counts"] == {
        "bright_line": 1,
        "hard_constraints": 1,
        "privacy": 2,
        "walkaway": 1,
    }
    assert report["factor_dimensions"]["culture_counts"] == {"uk": 1, "us": 1}
    assert report["factor_dimensions"]["seller_policy_dimension"]["status"] == "runtime_controlled"
    assert report["tier_profiles"]["0"]["zopa_status"] == {"wide": 1}
    assert report["tier_profiles"]["3"]["pressure_type"] == {"bright_line": 1}
    assert report["findings"]["tiers_are_provisional_operational_labels"] is True
    assert report["findings"]["empirical_calibration_required_after_scale"] is True
    assert (
        report["findings"]["difficulty_should_not_be_inferred_from_tier_label_alone"]
        is True
    )
