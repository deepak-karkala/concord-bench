import json

from concord.env.pettingzoo_wrapper import NegotiationAECWrapper
from concord.schemas.scenario import Domain, PrivateContext, Scenario


class TestPettingZooCompat:
    def test_aec_agent_cycle(self):
        scenario = Scenario(
            id="pz-comp-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000, private_info=["budget_10000"]),
            seller_context=PrivateContext(batna=5000, private_info=["cost_2000"]),
            deal_schema={"price": "float", "quantity": "int"},
        )
        env = NegotiationAECWrapper(scenario, seed=42)
        env.reset()

        assert env.agents == ["buyer", "seller"]
        assert env.num_agents == 2
        assert env.possible_agents == ["buyer", "seller"]

        obs = env.observe("buyer")
        assert obs["scenario_id"] == "pz-comp-test"
        assert obs["my_batna"] == 3000.0
        assert obs["current_turn"] == 0

        for agent in [env.agent_selection]:
            env.step({"action_type": 0, "content": "Hello"})
        assert env.agent_selection == "seller"

        env.step({
            "action_type": 1,
            "content": "I offer $99",
            "offer_json": json.dumps({"domain": "ecommerce", "price": 99.0, "quantity": 100}),
        })
        env.step({"action_type": 2, "content": "Accepted!"})

        assert env.terminations["buyer"] is True
        assert env.terminations["seller"] is True
        assert env.rewards["buyer"] == 100.0
        assert env.rewards["seller"] == 100.0

    def test_walk_away_yields_negative_rewards(self):
        scenario = Scenario(
            id="pz-walkaway",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=1000),
            seller_context=PrivateContext(batna=2000),
            deal_schema={"price": "float", "quantity": "int"},
        )
        env = NegotiationAECWrapper(scenario, seed=42)
        env.reset()
        env.step({"action_type": 4, "content": "No deal, walking away."})
        assert env.terminations["buyer"] is True
        assert env.rewards["buyer"] == -10.0

    def test_render_ansi(self):
        scenario = Scenario(
            id="pz-render",
            domain=Domain.SAAS_PROCUREMENT,
            buyer_context=PrivateContext(batna=50000),
            seller_context=PrivateContext(batna=75000),
            deal_schema={"monthly_price": "float", "seats": "int", "contract_length_months": "int"},
        )
        env = NegotiationAECWrapper(scenario, seed=42, render_mode="ansi")
        env.reset()
        env.step({"action_type": 0, "content": "We need 200 seats."})
        output = env.render()
        assert "pz-render" in output
        assert "200 seats" in output

    def test_native_vs_wrapper_produces_same_transcript(self):
        scenario = Scenario(
            id="compare-test",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=3000),
            seller_context=PrivateContext(batna=5000),
            deal_schema={"price": "float", "quantity": "int"},
        )

        from concord.env.core import NegotiationEnv
        native = NegotiationEnv()
        native.reset(scenario, seed=42)
        native.step_message("buyer", "Hello")
        native.step_offer("seller", "Take it", {"domain": "ecommerce", "price": 150, "quantity": 100})
        native_transcript = [t.content for t in native.state.turns]

        wrapper = NegotiationAECWrapper(scenario, seed=42)
        wrapper.reset()
        wrapper.step({"action_type": 0, "content": "Hello"})
        wrapper.step({"action_type": 1, "content": "Take it", "offer_json": json.dumps({"domain": "ecommerce", "price": 150, "quantity": 100})})
        wrapper_transcript = [t.content for t in wrapper._env.state.turns]

        assert native_transcript == wrapper_transcript
