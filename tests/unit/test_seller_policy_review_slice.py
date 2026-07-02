from __future__ import annotations

import json
from pathlib import Path

from concord.analysis.seller_policy_review_slice import (
    load_seller_policy_slice_inventory,
    select_phase4_review_slice,
    write_phase4_review_slice,
)


def _write_scenario(
    path: Path,
    *,
    scenario_id: str,
    domain: str,
    buyer_reserve: float,
    seller_reserve: float,
    tiering_reason: str,
    multi_issue: bool = False,
) -> None:
    payload = f"""
id: {scenario_id}
domain: {domain}
culture: US
max_turns: 10
buyer_context:
  batna: 10.0
  reserve_price: {buyer_reserve}
  hard_constraints: []
  private_info: []
  issue_utilities: {{"contract_length_months": {{"type": "numeric", "weight": 0.2, "best": 12, "worst": 36}}}}{'' if multi_issue else '{}'}
seller_context:
  batna: 10.0
  reserve_price: {seller_reserve}
  hard_constraints: []
  private_info: []
  issue_utilities: {{}}
deal_schema:
  price: float
forbidden_claims: []
scenario_description: test
metadata:
  difficulty_tier: 1
  tiering_reason: {tiering_reason}
"""
    payload = payload.replace('{"contract_length_months": {"type": "numeric", "weight": 0.2, "best": 12, "worst": 36}}{}', '{}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def test_select_phase4_review_slice_covers_domains_and_no_zopa(tmp_path: Path) -> None:
    for domain in ("ecommerce", "ethical_business", "saas_procurement", "settlement"):
        _write_scenario(
            tmp_path / domain / f"{domain}-nozopa-a.yaml",
            scenario_id=f"{domain}-nozopa-a",
            domain=domain,
            buyer_reserve=30.0,
            seller_reserve=40.0,
            tiering_reason="walkaway_or_constraint_pressure",
            multi_issue=True,
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-nozopa-b.yaml",
            scenario_id=f"{domain}-nozopa-b",
            domain=domain,
            buyer_reserve=25.0,
            seller_reserve=45.0,
            tiering_reason="walkaway_or_constraint_pressure",
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-multi.yaml",
            scenario_id=f"{domain}-multi",
            domain=domain,
            buyer_reserve=70.0,
            seller_reserve=40.0,
            tiering_reason="multi_issue_tradeoff",
            multi_issue=True,
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-wide.yaml",
            scenario_id=f"{domain}-wide",
            domain=domain,
            buyer_reserve=80.0,
            seller_reserve=30.0,
            tiering_reason="wide_zopa_simple",
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-pressure.yaml",
            scenario_id=f"{domain}-pressure",
            domain=domain,
            buyer_reserve=75.0,
            seller_reserve=35.0,
            tiering_reason="walkaway_or_constraint_pressure",
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-multi-extra.yaml",
            scenario_id=f"{domain}-multi-extra",
            domain=domain,
            buyer_reserve=78.0,
            seller_reserve=34.0,
            tiering_reason="other",
            multi_issue=True,
        )
        _write_scenario(
            tmp_path / domain / f"{domain}-wide-extra.yaml",
            scenario_id=f"{domain}-wide-extra",
            domain=domain,
            buyer_reserve=82.0,
            seller_reserve=32.0,
            tiering_reason="wide_zopa_simple",
        )

    inventory = load_seller_policy_slice_inventory(tmp_path)
    selected = select_phase4_review_slice(inventory, per_domain=6)

    assert len(selected) == 24
    assert {item.domain for item in selected} == {
        "ecommerce",
        "ethical_business",
        "saas_procurement",
        "settlement",
    }
    for domain in {item.domain for item in selected}:
        domain_items = [item for item in selected if item.domain == domain]
        assert len(domain_items) == 6
        assert sum(1 for item in domain_items if not item.zopa_exists) >= 2
        assert any(item.tiering_reason == "multi_issue_tradeoff" for item in domain_items)
        assert any(item.tiering_reason == "wide_zopa_simple" for item in domain_items)
        assert any(item.tiering_reason == "walkaway_or_constraint_pressure" for item in domain_items)
        assert sum(1 for item in domain_items if item.multi_issue) >= 3


def test_write_phase4_review_slice_copies_yaml_and_writes_manifest(tmp_path: Path) -> None:
    for domain in ("ecommerce", "ethical_business", "saas_procurement", "settlement"):
        _write_scenario(
            tmp_path / "src" / domain / f"seed-{domain}-001.yaml",
            scenario_id=f"seed-{domain}-001",
            domain=domain,
            buyer_reserve=80.0,
            seller_reserve=30.0,
            tiering_reason="wide_zopa_simple",
        )
    inventory = load_seller_policy_slice_inventory(tmp_path / "src")
    selected = select_phase4_review_slice(inventory, per_domain=1)

    output_root = tmp_path / "out"
    manifest_path = write_phase4_review_slice(selected, output_root=output_root)

    copied = output_root / "scenarios" / "ecommerce" / "seed-ecommerce-001.yaml"
    assert copied.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["selected_count"] == 4
    assert manifest["domains"] == {
        "ecommerce": 1,
        "ethical_business": 1,
        "saas_procurement": 1,
        "settlement": 1,
    }
    assert manifest["selected_scenarios"][0]["scenario_id"] == "seed-ecommerce-001"
