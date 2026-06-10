from pathlib import Path

import yaml

from concord.data.seed_audit import audit_no_zopa_slice


def _write_seed(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_no_zopa_slice_reports_distribution_validity_and_positive_controls(
    temp_dir: Path,
) -> None:
    _write_seed(
        temp_dir / "ecommerce" / "seed-no-zopa-valid.yaml",
        {
            "id": "seed-no-zopa-valid",
            "domain": "ecommerce",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 90.0,
                "walk_away_threshold": 0.4,
            },
            "seller_context": {
                "batna": 140.0,
                "reserve_price": 100.0,
                "walk_away_threshold": None,
            },
        },
    )
    _write_seed(
        temp_dir / "settlement" / "seed-no-zopa-invalid.yaml",
        {
            "id": "seed-no-zopa-invalid",
            "domain": "settlement",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 90.0,
                "walk_away_threshold": 0.4,
            },
            "seller_context": {
                "batna": 95.0,
                "reserve_price": 95.0,
                "walk_away_threshold": None,
            },
        },
    )
    _write_seed(
        temp_dir / "saas_procurement" / "seed-positive-control.yaml",
        {
            "id": "seed-positive-control",
            "domain": "saas_procurement",
            "buyer_context": {
                "batna": 20.0,
                "reserve_price": 60.0,
                "walk_away_threshold": 0.4,
            },
            "seller_context": {
                "batna": 100.0,
                "reserve_price": 40.0,
                "walk_away_threshold": 0.4,
            },
        },
    )

    report = audit_no_zopa_slice(temp_dir)

    assert report["no_zopa_count"] == 2
    assert report["structurally_valid_no_zopa_count"] == 1
    assert report["invalid_no_zopa_ids"] == ["seed-no-zopa-invalid"]
    assert report["no_zopa_by_domain"] == {"ecommerce": 1, "settlement": 1}
    assert report["positive_control_count"] == 1
    assert report["positive_control_by_domain"] == {"saas_procurement": 1}
    assert report["findings"]["target_range_met"] is False
    assert report["findings"]["minimum_domain_coverage_met"] is False
    assert report["findings"]["all_no_zopa_structurally_valid"] is False
