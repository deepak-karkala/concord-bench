import pytest

from concord.agents.closed_api_adapter import ClosedAPIAdapter
from concord.env.core import NegotiationEnv
from concord.schemas.scenario import Domain, PrivateContext, Scenario


@pytest.mark.requires_api
@pytest.mark.slow
class TestClosedAPIIntegration:
    @pytest.mark.asyncio
    async def test_claude_adapter_smoke(self):
        adapter = ClosedAPIAdapter("claude-opus-4-7", timeout=30.0)
        scenario = Scenario(
            id="api-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=1000, private_info=["budget_5000"]),
            seller_context=PrivateContext(batna=3000),
            deal_schema={"price": "float", "quantity": "int"},
        )
        env = NegotiationEnv()
        env.reset(scenario)
        action = await adapter.act(env.state, scenario.buyer_context)
        assert action.content
        assert adapter.total_prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_openai_adapter_smoke(self):
        adapter = ClosedAPIAdapter("gpt-5.2", timeout=30.0)
        scenario = Scenario(
            id="api-test-2",
            domain=Domain.SAAS_PROCUREMENT,
            buyer_context=PrivateContext(batna=50000, private_info=["budget_100000"]),
            seller_context=PrivateContext(batna=75000),
            deal_schema={"monthly_price": "float", "seats": "int", "contract_length_months": "int"},
        )
        env = NegotiationEnv()
        env.reset(scenario)
        action = await adapter.act(env.state, scenario.buyer_context)
        assert action.content
