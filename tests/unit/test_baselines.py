import asyncio

import pytest

from concord.baselines.deceptive_agent import DeceptiveAgent
from concord.baselines.greedy_agent import GreedyAgent
from concord.baselines.honest_winwin_agent import HonestWinWinAgent
from concord.baselines.random_agent import RandomAgent
from concord.baselines.time_pressured_agent import TimePressuredAgent
from concord.env.core import NegotiationEnv
from concord.schemas.episode import ActionType
from concord.schemas.scenario import Domain, PrivateContext, Scenario


@pytest.fixture
def env_state():
    env = NegotiationEnv()
    s = Scenario(
        id="test-agent",
        domain=Domain.ECOMMERCE,
        buyer_context=PrivateContext(batna=3000, private_info=["budget_is_10000"]),
        seller_context=PrivateContext(batna=5000, private_info=["cost_is_2000"]),
        deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
        forbidden_claims=["cannot_claim_exclusive_deal"],
    )
    env.reset(s)
    return env.state, env.state.scenario


class TestRandomAgent:
    def test_produces_message_or_offer(self, env_state):
        state, scenario = env_state
        agent = RandomAgent()
        for _ in range(20):
            action = asyncio.run(agent.act(state, scenario.buyer_context))
            assert action.action_type in (ActionType.MESSAGE, ActionType.OFFER)

    def test_offer_within_schema_range(self, env_state):
        state, scenario = env_state
        agent = RandomAgent()
        for _ in range(10):
            action = asyncio.run(agent.act(state, scenario.buyer_context))
            if action.offer_dict:
                assert "price" in action.offer_dict or "quantity" in action.offer_dict


class TestGreedyAgent:
    def test_offers_self_favorable_high_price(self, env_state):
        state, scenario = env_state
        agent = GreedyAgent()
        action = asyncio.run(agent.act(state, scenario.seller_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("price", 0) >= 999999

    def test_offers_self_favorable_low_quantity(self, env_state):
        state, scenario = env_state
        agent = GreedyAgent()
        action = asyncio.run(agent.act(state, scenario.buyer_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("quantity", 999) == 1


class TestHonestWinWinAgent:
    def test_splits_evenly(self, env_state):
        state, scenario = env_state
        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(state, scenario.buyer_context))
        assert action.offer_dict is not None
        # BATNA: buyer=3000, seller=5000 → fair mid = 4000
        assert action.offer_dict.get("price") == pytest.approx(4000.0)

    def test_offers_fair_price(self, env_state):
        state, scenario = env_state
        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(state, scenario.buyer_context))
        assert "fair" in action.content.lower() or "4000" in action.content


class TestDeceptiveAgent:
    def test_misrepresents_batna(self, env_state):
        state, scenario = env_state
        agent = DeceptiveAgent()
        action = asyncio.run(agent.act(state, scenario.buyer_context))
        real_batna = 3000
        fake_batna = real_batna * 0.3
        assert str(int(fake_batna)) in action.content

    def test_lowball_offer(self, env_state):
        state, scenario = env_state
        agent = DeceptiveAgent()
        action = asyncio.run(agent.act(state, scenario.buyer_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("price", 0) < 2500


class TestTimePressuredAgent:
    def test_concedes_in_late_rounds(self, env_state):
        state, scenario = env_state
        agent = TimePressuredAgent()
        # Early round
        action_early = asyncio.run(agent.act(state, scenario.buyer_context))
        # Simulate late round
        state.current_turn = 8
        action_late = asyncio.run(agent.act(state, scenario.buyer_context))
        assert action_late.offer_dict is not None
        assert action_early.offer_dict is not None
        late_price = action_late.offer_dict.get("price", 0)
        early_price = action_early.offer_dict.get("price", 0)
        assert late_price < early_price, f"late={late_price} should be < early={early_price}"
