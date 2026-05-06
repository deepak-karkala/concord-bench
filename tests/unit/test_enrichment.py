import pytest

from concord.schemas.scenario import Domain, Scenario
from concord.synth.enrichment import EnrichmentError, enrich_awm_scenario


def _awm_scenario(**overrides) -> dict:
    base = {
        "scenario_id": "awm-001",
        "name": "Test Marketplace",
        "description": "A test negotiation scenario",
        "category": "e-commerce",
        "feature_list": ["secure_payment", "seller_verification", "return_handling"],
    }
    base.update(overrides)
    return base


class TestEnrichAWMScenario:
    def test_ecommerce_enrichment(self):
        s = enrich_awm_scenario(_awm_scenario(), domain="ecommerce")
        assert isinstance(s, Scenario)
        assert s.id == "awm-001"
        assert s.domain == Domain.ECOMMERCE
        assert s.culture == "US"
        assert s.buyer_context.batna > 0
        assert s.seller_context.batna > 0
        assert s.deal_schema == {"price": "float", "quantity": "int", "delivery_days": "int", "payment_terms_days": "int", "shipping_terms": "str", "return_policy": "str"}

    def test_saas_enrichment(self):
        s = enrich_awm_scenario(
            _awm_scenario(scenario_id="awm-saas"), domain="saas_procurement", culture="JP"
        )
        assert s.domain == Domain.SAAS_PROCUREMENT
        assert s.culture == "JP"
        assert s.deal_schema == {"monthly_price": "float", "seats": "int", "contract_length_months": "int", "onboarding_support_hours": "int", "sla_tier": "str"}

    def test_settlement_enrichment(self):
        s = enrich_awm_scenario(
            _awm_scenario(scenario_id="awm-settle"), domain="settlement"
        )
        assert s.domain == Domain.SETTLEMENT
        assert "confidentiality_clause" in s.deal_schema
        assert "non_disparagement" in s.deal_schema

    def test_ethical_business_enrichment(self):
        s = enrich_awm_scenario(
            _awm_scenario(scenario_id="awm-ethic"), domain="ethical_business"
        )
        assert s.domain == Domain.ETHICAL_BUSINESS
        assert "environmental_commitments" in s.deal_schema
        assert "transparency_reports" in s.deal_schema

    def test_constraints_from_features(self):
        s = enrich_awm_scenario(
            _awm_scenario(feature_list=["fast_shipping", "bulk_discount", "custom_labeling"]),
            domain="ecommerce",
        )
        assert any("fast_shipping" in c for c in s.buyer_context.hard_constraints)
        assert any("bulk_discount" in c for c in s.buyer_context.hard_constraints)
        assert "deal_must_cover_core_requirements" in s.buyer_context.hard_constraints

    def test_forbidden_claims_generated(self):
        s = enrich_awm_scenario(
            _awm_scenario(feature_list=["secure_auth"]),
            domain="saas_procurement",
        )
        assert len(s.forbidden_claims) == 1
        assert "secure_auth" in s.forbidden_claims[0]

    def test_missing_scenario_id(self):
        with pytest.raises(EnrichmentError, match="Missing required field 'scenario_id'"):
            enrich_awm_scenario({}, domain="ecommerce")

    def test_empty_scenario_id(self):
        with pytest.raises(EnrichmentError, match="Empty required field 'scenario_id'"):
            enrich_awm_scenario({"scenario_id": ""}, domain="ecommerce")

    def test_whitespace_scenario_id(self):
        with pytest.raises(EnrichmentError, match="Empty required field 'scenario_id'"):
            enrich_awm_scenario({"scenario_id": "   "}, domain="ecommerce")

    def test_unknown_domain(self):
        with pytest.raises(EnrichmentError, match="Unknown domain: "):
            enrich_awm_scenario(_awm_scenario(), domain="invalid_domain")

    def test_minimal_awm_scenario(self):
        s = enrich_awm_scenario({"scenario_id": "minimal-1"}, domain="ecommerce")
        assert s.id == "minimal-1"
        assert s.buyer_context.batna == 3000.0  # base, no features
        assert s.seller_context.batna == 5000.0  # base, no features

    def test_output_validates_against_scenario(self):
        s = enrich_awm_scenario(
            _awm_scenario(scenario_id="validate-me"), domain="settlement"
        )
        data = s.model_dump()
        restored = Scenario.model_validate(data)
        assert restored == s

    def test_public_description_includes_name(self):
        s = enrich_awm_scenario(
            _awm_scenario(name="Acme Corp Procurement"), domain="saas_procurement"
        )
        assert "Acme Corp Procurement" in s.scenario_description

    def test_public_description_includes_category(self):
        s = enrich_awm_scenario(
            _awm_scenario(category="enterprise-software"), domain="saas_procurement"
        )
        assert "enterprise-software" in s.scenario_description
