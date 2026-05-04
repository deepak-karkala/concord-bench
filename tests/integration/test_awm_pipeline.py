import pytest

from concord.synth.enrichment import enrich_awm_scenario
from concord.schemas.scenario import Scenario


@pytest.mark.slow
@pytest.mark.requires_synth
def test_enrich_awm_scenario_yields_valid_scenario():
    awm_output = {
        "scenario_id": "int-test-001",
        "name": "Integration Test Marketplace",
        "description": "A scenario from AWM's output format",
        "category": "e-commerce",
        "feature_list": ["user_auth", "cart_checkout", "payment_gateway"],
    }
    scenario = enrich_awm_scenario(awm_output, domain="ecommerce", culture="US")
    assert isinstance(scenario, Scenario)
    assert scenario.id == "int-test-001"
    assert scenario.domain.value == "ecommerce"
    assert scenario.buyer_context.batna > 0
    assert scenario.seller_context.batna > 0
