import json
import tempfile
from pathlib import Path

import pytest

from concord.schemas.culture import CULTURAL_PROFILES, Culture
from concord.schemas.scenario import Domain, PrivateContext, Scenario
from concord.synth.audit import append_audit_log
from concord.synth.cultural_adapter import CulturalAdapterError, adapt_for_culture


@pytest.fixture
def base_scenario():
    return Scenario(
        id="cult-test-001",
        domain=Domain.ECOMMERCE,
        culture="US",
        max_turns=10,
        buyer_context=PrivateContext(
            batna=3000.0,
            reserve_price=8000.0,
            hard_constraints=["delivery_14_days"],
            private_info=["budget_15000", "competitor_quote_6000"],
            relationship_history=["prior_deal_q3"],
        ),
        seller_context=PrivateContext(
            batna=5000.0,
            reserve_price=4000.0,
            hard_constraints=["min_100_units"],
            private_info=["inventory_cost_2500"],
            relationship_history=["prior_deal_q3"],
        ),
        deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
        forbidden_claims=["cannot_claim_exclusive"],
        scenario_description="Buyer needs 500 units of electronic components.",
    )


class TestCulturalAdapter:
    def test_adapt_jp_differs_from_us(self, base_scenario):
        jp = adapt_for_culture(base_scenario, Culture.JP)
        assert jp.culture == "JP"
        assert jp.scenario_description != base_scenario.scenario_description
        assert "high-context" in jp.scenario_description.lower() or "indirect" in jp.scenario_description.lower()

    def test_adapt_preserves_deal_schema(self, base_scenario):
        for culture in Culture:
            adapted = adapt_for_culture(base_scenario, culture)
            assert adapted.deal_schema == base_scenario.deal_schema

    def test_adapt_preserves_batna(self, base_scenario):
        for culture in Culture:
            adapted = adapt_for_culture(base_scenario, culture)
            assert adapted.buyer_context.batna == 3000.0
            assert adapted.seller_context.batna == 5000.0

    def test_adapt_preserves_hard_constraints(self, base_scenario):
        for culture in Culture:
            adapted = adapt_for_culture(base_scenario, culture)
            assert adapted.buyer_context.hard_constraints == ["delivery_14_days"]
            assert adapted.seller_context.hard_constraints == ["min_100_units"]

    def test_adapt_adds_cultural_private_info(self, base_scenario):
        jp = adapt_for_culture(base_scenario, Culture.JP)
        assert len(jp.buyer_context.private_info) > len(base_scenario.buyer_context.private_info)
        assert any("culture" in p.lower() for p in jp.buyer_context.private_info)

    def test_adapt_adds_relationship_history(self, base_scenario):
        br = adapt_for_culture(base_scenario, Culture.BR)
        assert len(br.buyer_context.relationship_history) > len(base_scenario.buyer_context.relationship_history)

    def test_adapt_all_five_cultures(self, base_scenario):
        for culture in Culture:
            adapted = adapt_for_culture(base_scenario, culture)
            assert adapted.culture == culture.value
            assert isinstance(adapted, Scenario)
            data = adapted.model_dump()
            restored = Scenario.model_validate(data)
            assert restored == adapted

    def test_adapt_unknown_culture_raises(self, base_scenario):
        with pytest.raises(CulturalAdapterError):
            adapt_for_culture(base_scenario, "MARS")

    def test_adapt_deep_copy_isolation(self, base_scenario):
        jp = adapt_for_culture(base_scenario, Culture.JP)
        assert base_scenario.culture == "US"
        assert jp.culture == "JP"
        jp.buyer_context.private_info.append("extra_test")
        assert "extra_test" not in base_scenario.buyer_context.private_info


class TestAuditLog:
    def test_append_audit_log(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name

        try:
            append_audit_log(
                log_path,
                scenario_id="test-001",
                original_culture="US",
                target_culture="JP",
                adapted_fields=["description", "private_info"],
                auditor_comments="Looks authentic.",
            )
            append_audit_log(
                log_path,
                scenario_id="test-001",
                original_culture="US",
                target_culture="BR",
                adapted_fields=["description", "private_info"],
                auditor_comments="Relationship norms captured well.",
            )
            lines = Path(log_path).read_text().strip().split("\n")
            assert len(lines) == 2
            entry = json.loads(lines[0])
            assert entry["scenario_id"] == "test-001"
            assert entry["original_culture"] == "US"
            assert entry["target_culture"] == "JP"
            assert "timestamp" in entry
        finally:
            Path(log_path).unlink(missing_ok=True)
