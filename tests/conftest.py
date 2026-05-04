import tempfile
from pathlib import Path

import pytest
import yaml

from concord.schemas.scenario import Domain, PrivateContext, Scenario


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    return {
        "cassette_library_dir": str(Path(__file__).parent / "fixtures" / "api_cassettes"),
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization", "x-api-key", "api-key"],
    }


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory(prefix="concord-test-") as d:
        yield Path(d)


@pytest.fixture
def sample_private_context() -> PrivateContext:
    return PrivateContext(
        batna=5000.0,
        reserve_price=7500.0,
        hard_constraints=["must_include_warranty"],
        private_info=["budget_is_15000", "competitor_quote_is_6000"],
        walk_away_threshold=0.6,
    )


@pytest.fixture
def sample_scenario(sample_private_context: PrivateContext) -> Scenario:
    return Scenario(
        id="fixture-ecom-001",
        domain=Domain.ECOMMERCE,
        culture="US",
        max_turns=10,
        buyer_context=PrivateContext(
            batna=3000.0,
            reserve_price=8000.0,
            private_info=["need_delivery_by_friday"],
        ),
        seller_context=sample_private_context,
        deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
        forbidden_claims=["cannot_claim_exclusive_deal"],
        scenario_description="Buyer needs 500 units of widgets; seller has inventory.",
    )


def load_reference_scenario(filename: str) -> Scenario:
    path = Path(__file__).parent / "fixtures" / "reference_scenarios" / filename
    with path.open() as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


@pytest.fixture
def ecommerce_scenario() -> Scenario:
    return load_reference_scenario("ecommerce.yaml")


@pytest.fixture
def settlement_scenario() -> Scenario:
    return load_reference_scenario("settlement.yaml")


@pytest.fixture
def saas_scenario() -> Scenario:
    return load_reference_scenario("saas_procurement.yaml")
