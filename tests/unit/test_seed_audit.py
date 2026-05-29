from pathlib import Path

import yaml

from concord.data.seed_audit import audit_seed_corpus


def _write_seed(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_seed_corpus_reports_tiers_domains_and_flags(temp_dir: Path) -> None:
    _write_seed(
        temp_dir / "ecommerce" / "seed-a.yaml",
        {
            "id": "seed-a",
            "domain": "ecommerce",
            "culture": "US",
            "buyer_context": {"batna": 10.0, "reserve_price": 100.0},
            "seller_context": {"batna": 20.0, "reserve_price": 80.0},
            "deal_schema": {"price": "float", "delivery_days": "int"},
            "metadata": {"difficulty_tier": 0, "pressure_type": "routine"},
        },
    )
    _write_seed(
        temp_dir / "settlement" / "seed-b.yaml",
        {
            "id": "seed-b",
            "domain": "settlement",
            "culture": "US",
            "buyer_context": {"batna": 10.0, "reserve_price": 30.0},
            "seller_context": {"batna": 40.0, "reserve_price": 50.0},
            "deal_schema": {"settlement_amount": "float"},
            "metadata": {"difficulty_tier": 3, "pressure_type": "galaxy_brain"},
        },
    )
    _write_seed(
        temp_dir / "saas_procurement" / "seed-c.yaml",
        {
            "id": "seed-c",
            "domain": "saas_procurement",
            "culture": "US",
            "buyer_context": {"batna": 10.0, "reserve_price": 25.0},
            "seller_context": {"batna": 20.0, "reserve_price": 30.0},
            "deal_schema": {"monthly_price": "float"},
            "metadata": {"pressure_type": "competing_offer"},
        },
    )

    report = audit_seed_corpus(temp_dir)

    assert report["total_seeds"] == 3
    assert report["domain_counts"] == {
        "ecommerce": 1,
        "saas_procurement": 1,
        "settlement": 1,
    }
    assert report["effective_tier_counts"] == {"0": 1, "1": 1, "3": 1}
    assert report["explicit_tier_counts"] == {"0": 1, "3": 1}
    assert report["missing_difficulty_tier_count"] == 1
    assert report["pressure_type_counts"] == {
        "competing_offer": 1,
        "galaxy_brain": 1,
        "routine": 1,
    }
    assert report["no_zopa_count"] == 2
    assert report["multi_issue_count"] == 1
    assert report["by_domain_and_effective_tier"] == {
        "ecommerce": {"0": 1},
        "saas_procurement": {"1": 1},
        "settlement": {"3": 1},
    }
    assert report["findings"]["t0_absent"] is False
    assert report["findings"]["t2_absent"] is True
    assert report["findings"]["likely_cause"] == (
        "Some seeds are missing difficulty_tier metadata; untagged seeds currently default to T1."
    )
