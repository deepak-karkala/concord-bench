import pytest

from concord.data.loader import load_seeds
from concord.schemas.scenario import Scenario


@pytest.mark.slow
def test_all_seed_yamls_validate():
    scenarios = load_seeds()
    for s in scenarios:
        assert isinstance(s, Scenario)
        assert s.id
        assert s.domain
        assert s.buyer_context.batna is not None
        assert s.seller_context.batna is not None
        assert isinstance(s.deal_schema, dict)
