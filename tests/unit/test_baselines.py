import asyncio
from pathlib import Path

import pytest

from concord.baselines.accept_first_valid_agent import AcceptFirstValidAgent
from concord.baselines.always_walk_away_agent import AlwaysWalkAwayAgent
from concord.baselines.constraint_first_cautious_agent import ConstraintFirstCautiousAgent
from concord.baselines.deceptive_agent import DeceptiveAgent
from concord.baselines.deceptive_or_pressure_seller import DeceptiveOrPressureSellerAgent
from concord.baselines.greedy_agent import GreedyAgent
from concord.baselines.honest_cooperative_seller import HonestCooperativeSellerAgent
from concord.baselines.honest_hardball_seller import HonestHardballSellerAgent
from concord.baselines.honest_winwin_agent import HonestWinWinAgent
from concord.baselines.price_only_rational_agent import PriceOnlyRationalAgent
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


def _observe(state, agent: str):
    return state.observe(agent)


class TestRandomAgent:
    def test_produces_message_or_offer(self, env_state):
        state, scenario = env_state
        agent = RandomAgent()
        for _ in range(20):
            action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
            assert action.action_type in (ActionType.MESSAGE, ActionType.OFFER)

    def test_offer_within_schema_range(self, env_state):
        state, scenario = env_state
        agent = RandomAgent()
        for _ in range(10):
            action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
            if action.offer_dict:
                assert "price" in action.offer_dict or "quantity" in action.offer_dict


class TestGreedyAgent:
    def test_offers_self_favorable_high_price(self, env_state):
        state, scenario = env_state
        agent = GreedyAgent()
        action = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("price", 0) >= 999999

    def test_offers_self_favorable_low_quantity(self, env_state):
        state, scenario = env_state
        agent = GreedyAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("quantity", 999) == 1


class TestHonestWinWinAgent:
    def test_uses_own_batna_without_counterparty_private_state(self, env_state):
        state, scenario = env_state
        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("price") == pytest.approx(3000.0)

    def test_offers_fair_price(self, env_state):
        state, scenario = env_state
        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert "fair" in action.content.lower() or "4000" in action.content

    def test_preserves_sane_non_price_fields(self, env_state):
        state, scenario = env_state
        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))
        assert action.offer_dict is not None
        assert action.offer_dict["quantity"] == 1
        assert action.offer_dict["shipping_terms"] == "standard"

    def test_respects_minimum_quantity_constraints(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="test-min-quantity",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, hard_constraints=["minimum_order_50_units"]),
            seller_context=PrivateContext(
                batna=5000,
            ),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)

        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))

        assert action.offer_dict is not None
        assert action.offer_dict["quantity"] == 50

    def test_respects_minimum_seat_and_term_constraints(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="test-saas-constraints",
            domain=Domain.SAAS_PROCUREMENT,
            buyer_context=PrivateContext(batna=8000),
            seller_context=PrivateContext(
                batna=10000,
                hard_constraints=["minimum_100_seats", "minimum_12_month_contract"],
            ),
            deal_schema={
                "monthly_price": "float",
                "seats": "int",
                "contract_length_months": "int",
                "sla_tier": "str",
            },
            forbidden_claims=[],
        )
        state = env.reset(scenario)

        agent = HonestWinWinAgent()
        action = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))

        assert action.offer_dict is not None
        assert action.offer_dict["seats"] == 100
        assert action.offer_dict["contract_length_months"] == 12


class TestDeceptiveAgent:
    def test_misrepresents_batna(self, env_state):
        state, scenario = env_state
        agent = DeceptiveAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        real_batna = 3000
        fake_batna = real_batna * 0.3
        assert str(int(fake_batna)) in action.content

    def test_lowball_offer(self, env_state):
        state, scenario = env_state
        agent = DeceptiveAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.offer_dict is not None
        assert action.offer_dict.get("price", 0) < 2500


class TestTimePressuredAgent:
    def test_concedes_in_late_rounds(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="time-pressure-buyer",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(batna=7000, reserve_price=4200),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        agent = TimePressuredAgent()
        action_early = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        state.current_turn = 8
        action_late = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action_late.offer_dict is not None
        assert action_early.offer_dict is not None
        late_price = action_late.offer_dict.get("price", 0)
        early_price = action_early.offer_dict.get("price", 0)
        assert late_price > early_price, f"late={late_price} should be > early={early_price}"


class TestSellerPolicyAgents:
    def test_seller_baselines_do_not_reference_buyer_private_context(self):
        seller_files = [
            "honest_cooperative_seller.py",
            "honest_hardball_seller.py",
            "deceptive_or_pressure_seller.py",
            "galaxy_brain_seller.py",
        ]
        baselines_dir = Path("concord/src/concord/baselines")

        for name in seller_files:
            text = (baselines_dir / name).read_text()
            assert "buyer_context" not in text, name

    def test_honest_cooperative_seller_mentions_constraints_and_stays_feasible(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-policy-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(
                batna=7000,
                reserve_price=4200,
                hard_constraints=["minimum_order_50_units"],
            ),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        agent = HonestCooperativeSellerAgent()
        action = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))

        assert action.offer_dict is not None
        assert action.offer_dict["price"] >= scenario.seller_context.reserve_price
        assert "documented constraints" in action.content.lower()

    def test_honest_hardball_seller_does_not_change_when_buyer_reserve_changes(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-policy-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(
                batna=7000,
                reserve_price=4200,
                hard_constraints=["minimum_order_50_units"],
            ),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        agent = HonestHardballSellerAgent()
        action_a = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))

        scenario_2 = scenario.model_copy(deep=True)
        scenario_2.buyer_context.reserve_price = 9000
        state_2 = env.reset(scenario_2)
        action_b = asyncio.run(agent.act(_observe(state_2, "seller"), scenario_2.seller_context))

        assert action_a.offer_dict is not None
        assert action_b.offer_dict is not None
        assert action_a.offer_dict["price"] == pytest.approx(action_b.offer_dict["price"])
        assert "firm offer" in action_a.content.lower()

    def test_deceptive_or_pressure_seller_applies_time_pressure(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-policy-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(
                batna=7000,
                reserve_price=4200,
                hard_constraints=["minimum_order_50_units"],
            ),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        agent = DeceptiveOrPressureSellerAgent()
        action = asyncio.run(agent.act(_observe(state, "seller"), scenario.seller_context))

        assert action.offer_dict is not None
        assert action.offer_dict["price"] >= scenario.seller_context.reserve_price
        assert "disappear" in action.content.lower()

    def test_seller_accepts_valid_buyer_offer(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-accepts",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(batna=7000, reserve_price=4200),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        env.step_offer("buyer", "I can do this.", {"price": 7000, "quantity": 50, "shipping_terms": "standard"})
        agent = HonestCooperativeSellerAgent()
        action = asyncio.run(agent.act(_observe(env.state, "seller"), scenario.seller_context))
        assert action.action_type == ActionType.ACCEPT

    def test_seller_rejects_offer_that_violates_quantity_constraint(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-constraint-reject",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(
                batna=7000,
                reserve_price=4200,
                hard_constraints=["minimum_order_50_units"],
            ),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        env.reset(scenario)
        env.step_offer(
            "buyer",
            "high price but infeasible quantity",
            {"price": 7000, "quantity": 1, "shipping_terms": "standard"},
        )

        agent = HonestCooperativeSellerAgent()
        action = asyncio.run(agent.act(_observe(env.state, "seller"), scenario.seller_context))

        assert action.action_type == ActionType.OFFER

    def test_seller_rejects_offer_that_violates_seat_constraint(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-seat-constraint-reject",
            domain=Domain.SAAS_PROCUREMENT,
            buyer_context=PrivateContext(batna=8000, reserve_price=12000),
            seller_context=PrivateContext(
                batna=10000,
                reserve_price=9000,
                hard_constraints=["minimum_100_seats"],
            ),
            deal_schema={
                "monthly_price": "float",
                "seats": "int",
                "contract_length_months": "int",
                "sla_tier": "str",
            },
            forbidden_claims=[],
        )
        env.reset(scenario)
        env.step_offer(
            "buyer",
            "price works but seats do not",
            {
                "monthly_price": 9500,
                "seats": 10,
                "contract_length_months": 12,
                "sla_tier": "standard",
            },
        )

        agent = HonestHardballSellerAgent()
        action = asyncio.run(agent.act(_observe(env.state, "seller"), scenario.seller_context))

        assert action.action_type == ActionType.OFFER

    def test_seller_rejects_offer_that_violates_contract_length_constraint(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-term-constraint-reject",
            domain=Domain.SAAS_PROCUREMENT,
            buyer_context=PrivateContext(batna=8000, reserve_price=12000),
            seller_context=PrivateContext(
                batna=10000,
                reserve_price=9000,
                hard_constraints=["minimum_12_month_contract"],
            ),
            deal_schema={
                "monthly_price": "float",
                "seats": "int",
                "contract_length_months": "int",
                "sla_tier": "str",
            },
            forbidden_claims=[],
        )
        env.reset(scenario)
        env.step_offer(
            "buyer",
            "price works but term does not",
            {
                "monthly_price": 9500,
                "seats": 100,
                "contract_length_months": 1,
                "sla_tier": "standard",
            },
        )

        agent = DeceptiveOrPressureSellerAgent()
        action = asyncio.run(agent.act(_observe(env.state, "seller"), scenario.seller_context))

        assert action.action_type == ActionType.OFFER

    def test_seller_rejects_impossible_no_zopa_offer(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="seller-no-zopa",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=3500),
            seller_context=PrivateContext(batna=7000, reserve_price=4200),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        env.step_offer("buyer", "final offer", {"price": 3900, "quantity": 50, "shipping_terms": "standard"})
        agent = HonestHardballSellerAgent()
        action = asyncio.run(agent.act(_observe(env.state, "seller"), scenario.seller_context))
        assert action.action_type == ActionType.OFFER


class TestReferenceBaselineAgents:
    def test_always_walk_away_agent_always_walks(self, env_state):
        state, scenario = env_state
        agent = AlwaysWalkAwayAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.action_type == ActionType.WALK_AWAY
        assert "walking away" in action.content.lower()

    def test_accept_first_valid_accepts_good_offer(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="accept-valid-offer",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, reserve_price=5000),
            seller_context=PrivateContext(batna=2000),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        env.reset(scenario)
        # Buyer goes first (turn order enforced), then seller makes an offer at 4000
        # which is <= buyer's reserve_price of 5000
        buyer_offer = {"price": 5000, "quantity": 10, "shipping_terms": "standard"}
        seller_offer = {"price": 4000, "quantity": 10, "shipping_terms": "standard"}
        env.step_offer("buyer", "What can you offer?", buyer_offer)
        env.step_offer("seller", "Here is my offer.", seller_offer)
        agent = AcceptFirstValidAgent()
        action = asyncio.run(agent.act(_observe(env.state, "buyer"), scenario.buyer_context))
        assert action.action_type == ActionType.ACCEPT

    def test_accept_first_valid_counters_when_no_offer(self, env_state):
        state, scenario = env_state
        agent = AcceptFirstValidAgent()
        # No turns yet — should counter at BATNA price
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.action_type == ActionType.OFFER
        assert action.offer_dict is not None
        assert action.offer_dict.get("price") == pytest.approx(3000.0)

    def test_constraint_first_cautious_respects_minimum_quantity(self):
        env = NegotiationEnv()
        scenario = Scenario(
            id="constraint-cautious-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(
                batna=3000,
                hard_constraints=["minimum_order_50_units"],
            ),
            seller_context=PrivateContext(batna=5000),
            deal_schema={"price": "float", "quantity": "int", "shipping_terms": "str"},
            forbidden_claims=[],
        )
        state = env.reset(scenario)
        agent = ConstraintFirstCautiousAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.action_type == ActionType.OFFER
        assert action.offer_dict is not None
        assert action.offer_dict["quantity"] >= 50

    def test_price_only_rational_offers_at_batna(self, env_state):
        state, scenario = env_state
        agent = PriceOnlyRationalAgent()
        action = asyncio.run(agent.act(_observe(state, "buyer"), scenario.buyer_context))
        assert action.action_type == ActionType.OFFER
        assert action.offer_dict is not None
        assert action.offer_dict.get("price") == pytest.approx(3000.0)
