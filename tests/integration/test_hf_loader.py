import pytest

from concord.data.loader import load_scenarios
from concord.schemas.scenario import Domain


@pytest.mark.slow
@pytest.mark.requires_api
def test_load_scenarios_from_hf_by_domain():
    scenarios = load_scenarios(version="v0.1.0", domain="ecommerce")
    assert len(scenarios) > 0
    for s in scenarios:
        assert s.domain == Domain.ECOMMERCE


@pytest.mark.slow
@pytest.mark.requires_api
def test_load_scenarios_from_hf_by_culture():
    scenarios = load_scenarios(version="v0.1.0", domain="ecommerce", culture="US")
    assert len(scenarios) > 0
    for s in scenarios:
        assert s.culture == "US"
