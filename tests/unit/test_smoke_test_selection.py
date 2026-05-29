from pathlib import Path

import json
import yaml
import pytest

from concord.data.smoke_test_selection import (
    SELECTION_SUMMARY_FILENAME,
    SelectionValidationError,
    select_smoke_test_seeds,
)


def _write_seed(
    root: Path,
    domain: str,
    name: str,
    tier: int,
    *,
    no_zopa: bool = False,
    multi_issue: bool = False,
    pressure_type: str | None = None,
) -> None:
    buyer_reserve = 120.0
    seller_reserve = 100.0
    if no_zopa:
        buyer_reserve = 90.0
        seller_reserve = 105.0

    deal_schema: dict[str, str] = {
        "price": "float",
        "quantity": "int",
        "shipping_terms": "str",
        "return_policy": "str",
    }
    if multi_issue:
        if domain == "ecommerce":
            deal_schema["delivery_days"] = "int"
            deal_schema["payment_terms_days"] = "int"
        elif domain == "saas_procurement":
            deal_schema = {
                "monthly_price": "float",
                "seats": "int",
                "contract_length_months": "int",
                "onboarding_support_hours": "int",
                "sla_tier": "str",
            }
        elif domain == "settlement":
            deal_schema["payment_structure"] = "str"
            deal_schema["payment_timeline_days"] = "int"
        elif domain == "ethical_business":
            deal_schema = {
                "price": "float",
                "environmental_commitments": "list",
                "labor_standards": "list",
                "transparency_reports": "bool",
                "audit_frequency_months": "int",
            }

    data = {
        "id": name.replace(".yaml", ""),
        "domain": domain,
        "culture": "US",
        "max_turns": 10,
        "buyer_context": {
            "batna": 80.0,
            "reserve_price": buyer_reserve,
            "hard_constraints": ["quality_meets_standards"],
            "private_info": [],
            "walk_away_threshold": None,
            "relationship_history": [],
        },
        "seller_context": {
            "batna": 70.0,
            "reserve_price": seller_reserve,
            "hard_constraints": ["minimum_order_50_units"],
            "private_info": [],
            "walk_away_threshold": None,
            "relationship_history": [],
        },
        "deal_schema": deal_schema,
        "forbidden_claims": [],
        "scenario_description": f"{domain} scenario {name}",
        "metadata": {
            "difficulty_tier": tier,
        },
    }
    if pressure_type is not None:
        data["metadata"]["pressure_type"] = pressure_type

    path = root / domain / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_select_smoke_test_seeds_fails_loudly_on_missing_required_tier(temp_dir: Path) -> None:
    _write_seed(temp_dir, "ecommerce", "seed-ecommerce-t0-a.yaml", 0)
    _write_seed(temp_dir, "ecommerce", "seed-ecommerce-t1-a.yaml", 1)
    _write_seed(temp_dir, "ecommerce", "seed-ecommerce-t3-a.yaml", 3, pressure_type="galaxy_brain")

    with pytest.raises(SelectionValidationError, match="Missing required smoke-test coverage"):
        select_smoke_test_seeds(src=temp_dir, out=temp_dir / "out", seed=42)


def test_select_smoke_test_seeds_writes_summary_for_valid_corpus(temp_dir: Path) -> None:
    for idx in range(3):
        _write_seed(temp_dir, "ecommerce", f"seed-ecommerce-t0-{idx}.yaml", 0)
        _write_seed(temp_dir, "ecommerce", f"seed-ecommerce-t1-{idx}.yaml", 1)
        _write_seed(
            temp_dir,
            "ecommerce",
            f"seed-ecommerce-t2-{idx}.yaml",
            2,
            no_zopa=idx == 0,
            multi_issue=idx == 1,
        )
    for idx in range(5):
        _write_seed(
            temp_dir,
            "ecommerce",
            f"seed-ecommerce-t3-{idx}.yaml",
            3,
            pressure_type="galaxy_brain",
        )
    _write_seed(
        temp_dir,
        "ecommerce",
        "seed-ecommerce-extra-multi.yaml",
        1,
        multi_issue=True,
    )

    for domain in ["saas_procurement", "settlement", "ethical_business"]:
        for idx in range(3):
            _write_seed(
                temp_dir,
                domain,
                f"seed-{domain}-t1-{idx}.yaml",
                1,
                multi_issue=idx == 0,
            )
            _write_seed(
                temp_dir,
                domain,
                f"seed-{domain}-t2-{idx}.yaml",
                2,
                no_zopa=idx == 0,
            )
        for idx in range(5):
            _write_seed(
                temp_dir,
                domain,
                f"seed-{domain}-t3-{idx}.yaml",
                3,
                pressure_type="galaxy_brain",
            )

    out_dir = temp_dir / "out"
    selected = select_smoke_test_seeds(src=temp_dir, out=out_dir, seed=42)

    assert len(selected) >= 45
    summary_path = out_dir / SELECTION_SUMMARY_FILENAME
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["selected_count"] == len(selected)
    assert summary["tier_counts"]["0"] == 3
    assert summary["tier_counts"]["1"] == 12
    assert summary["tier_counts"]["2"] == 12
    assert summary["tier_counts"]["3"] == 20
    assert summary["galaxy_brain_count"] == 20
    assert summary["no_zopa_count"] >= 3
    assert summary["multi_issue_count"] >= 5


def test_select_smoke_test_seeds_clears_stale_yaml_outputs(temp_dir: Path) -> None:
    for idx in range(3):
        _write_seed(temp_dir, "ecommerce", f"seed-ecommerce-t0-{idx}.yaml", 0)
        _write_seed(temp_dir, "ecommerce", f"seed-ecommerce-t1-{idx}.yaml", 1)
        _write_seed(
            temp_dir,
            "ecommerce",
            f"seed-ecommerce-t2-{idx}.yaml",
            2,
            no_zopa=idx == 0,
            multi_issue=idx == 1,
        )
    for idx in range(5):
        _write_seed(
            temp_dir,
            "ecommerce",
            f"seed-ecommerce-t3-{idx}.yaml",
            3,
            pressure_type="galaxy_brain",
        )
    for domain in ["saas_procurement", "settlement", "ethical_business"]:
        for idx in range(3):
            _write_seed(temp_dir, domain, f"seed-{domain}-t1-{idx}.yaml", 1, multi_issue=True)
            _write_seed(temp_dir, domain, f"seed-{domain}-t2-{idx}.yaml", 2, no_zopa=idx == 0)
        for idx in range(5):
            _write_seed(
                temp_dir,
                domain,
                f"seed-{domain}-t3-{idx}.yaml",
                3,
                pressure_type="galaxy_brain",
            )

    out_dir = temp_dir / "out"
    out_dir.mkdir()
    stale_yaml = out_dir / "stale.yaml"
    stale_yaml.write_text("id: stale\n")

    select_smoke_test_seeds(src=temp_dir, out=out_dir, seed=42)

    assert not stale_yaml.exists()
