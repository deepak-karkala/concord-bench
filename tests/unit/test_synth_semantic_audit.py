from pathlib import Path

import yaml

from concord.synth.audit import audit_generated_scenarios


def _write_scenario(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_audit_generated_scenarios_flags_semantic_quality_failures(temp_dir: Path) -> None:
    _write_scenario(
        temp_dir / "valid.yaml",
        {
            "id": "valid-1",
            "domain": "ecommerce",
            "culture": "US",
            "buyer_context": {
                "batna": 50.0,
                "reserve_price": 100.0,
                "hard_constraints": ["quality_meets_standards"],
                "private_info": ["budget_is_120", "inventory_runs_out_in_3_weeks"],
            },
            "seller_context": {
                "batna": 120.0,
                "reserve_price": 80.0,
                "hard_constraints": ["minimum_order_100_units"],
                "private_info": ["cost_basis_is_60", "inventory_position_is_strong"],
            },
            "deal_schema": {
                "price": "float",
                "quantity": "int",
                "delivery_days": "int",
                "payment_terms_days": "int",
                "shipping_terms": "str",
                "return_policy": "str",
            },
            "scenario_description": "A distributor and supplier are negotiating a replenishment order before the buyer's inventory window closes.",
        },
    )
    _write_scenario(
        temp_dir / "bad.yaml",
        {
            "id": "bad-1",
            "domain": "saas_procurement",
            "culture": "US",
            "buyer_context": {
                "batna": 70.0,
                "reserve_price": 60.0,
                "hard_constraints": ["quality_meets_standards"],
                "private_info": ["TBD", "quality_meets_standards"],
            },
            "seller_context": {
                "batna": 40.0,
                "reserve_price": 50.0,
                "hard_constraints": ["minimum_order_100_units"],
                "private_info": ["placeholder"],
            },
            "deal_schema": {"price": "float"},
            "scenario_description": "TODO: generic negotiation template for a software deal.",
        },
    )
    _write_scenario(
        temp_dir / "dup.yaml",
        {
            "id": "dup-1",
            "domain": "ecommerce",
            "culture": "US",
            "buyer_context": {
                "batna": 55.0,
                "reserve_price": 105.0,
                "hard_constraints": ["quality_meets_standards"],
                "private_info": ["budget_is_140", "launch_deadline_in_2_weeks"],
            },
            "seller_context": {
                "batna": 110.0,
                "reserve_price": 85.0,
                "hard_constraints": ["minimum_order_100_units"],
                "private_info": ["cost_basis_is_62", "inventory_position_is_strong"],
            },
            "deal_schema": {
                "price": "float",
                "quantity": "int",
                "delivery_days": "int",
                "payment_terms_days": "int",
                "shipping_terms": "str",
                "return_policy": "str",
            },
            "scenario_description": "A distributor and supplier are negotiating a replenishment order before the buyer's inventory window closes.",
        },
    )

    report = audit_generated_scenarios(temp_dir)

    assert report["total_scenarios"] == 3
    assert report["invalid_batna_math_ids"] == ["bad-1"]
    assert report["domain_schema_mismatch_ids"] == ["bad-1"]
    assert report["unmeasurable_constraint_ids"] == ["bad-1"]
    assert report["private_info_realism_ids"] == ["bad-1"]
    assert report["template_artifact_ids"] == ["bad-1"]
    assert report["duplicate_description_groups"] == [["dup-1", "valid-1"]]
    assert report["findings"]["blocking_failures_present"] is True
