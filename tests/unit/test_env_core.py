import pytest

from concord.env.core import EnvError, EnvTerminalError, NegotiationEnv
from concord.schemas.episode import ActionType


@pytest.fixture
def env():
    return NegotiationEnv()


@pytest.fixture
def reset_env(env, sample_scenario):
    env.reset(sample_scenario, seed=42)
    return env


class TestNegotiationEnv:
    def test_reset_sets_initial_state(self, env, sample_scenario):
        state = env.reset(sample_scenario, seed=42)
        assert state.current_turn == 0
        assert state.current_agent == "buyer"
        assert state.terminal is False
        assert len(state.turns) == 0

    def test_deterministic_reset(self, sample_scenario):
        env1 = NegotiationEnv()
        env2 = NegotiationEnv()
        env1.reset(sample_scenario, seed=42)
        env2.reset(sample_scenario, seed=42)
        assert env1.state.seed == env2.state.seed
        assert env1.state.current_agent == env2.state.current_agent

    def test_step_without_reset_raises(self, env):
        with pytest.raises(EnvError, match="not reset"):
            env.step_message("buyer", "hello")

    def test_message_action(self, reset_env):
        state = reset_env.step_message("buyer", "Hello there")
        assert len(state.turns) == 1
        assert state.turns[0].action_type == ActionType.MESSAGE
        assert state.turns[0].content == "Hello there"
        assert state.current_agent == "seller"
        assert state.terminal is False

    def test_offer_action(self, reset_env):
        state = reset_env.step_offer(
            "buyer", "Here is my offer",
            {"domain": "ecommerce", "price": 150.0, "quantity": 100},
        )
        assert len(state.turns) == 1
        assert state.turns[0].action_type == ActionType.OFFER
        assert state.turns[0].offer is not None
        assert state.turns[0].offer.price == 150.0  # type: ignore[union-attr]

    def test_accept_action(self, reset_env):
        reset_env.step_message("buyer", "H")
        reset_env.step_offer("seller", "Offer", {"domain": "ecommerce", "price": 99, "quantity": 10})
        reset_env.step_message("buyer", "Seems fair")
        state = reset_env.step_accept("seller", "Deal!", {"domain": "ecommerce", "price": 99, "quantity": 10})
        assert state.terminal is True
        assert state.deal is not None
        assert state.deal.price == 99  # type: ignore[union-attr]

    def test_reject_action(self, reset_env):
        reset_env.step_message("buyer", "Hello")
        reset_env.step_offer("seller", "Offer", {"domain": "ecommerce", "price": 500, "quantity": 10})
        state = reset_env.step_reject("buyer", "Too expensive")
        assert state.terminal is False
        assert state.current_agent == "seller"
        assert state.turns[-1].action_type == ActionType.REJECT

    def test_walk_away_action(self, reset_env):
        state = reset_env.step_walk_away("buyer", "No deal possible")
        assert state.terminal is True
        assert state.deal is None

    def test_escalate_action(self, reset_env):
        reset_env.step_message("buyer", "Hello")
        reset_env.step_message("seller", "Hi")
        state = reset_env.step_escalate("buyer", "Need manager approval")
        assert state.terminal is False
        assert state.turns[-1].action_type == ActionType.ESCALATE
        assert state.current_agent == "seller"

    def test_max_turns_boundary(self, sample_scenario):
        env = NegotiationEnv()
        s = sample_scenario.model_copy(update={"max_turns": 2})
        env.reset(s)
        env.step_message("buyer", "a")
        env.step_message("seller", "b")
        assert env.state.terminal is True
        assert env.state.deal is None

    def test_terminal_error_on_double_step(self, reset_env):
        reset_env.step_walk_away("buyer", "bye")
        with pytest.raises(EnvTerminalError, match="already terminal"):
            reset_env.step_message("seller", "wait")

    def test_wrong_agent_turn(self, reset_env):
        with pytest.raises(EnvError, match="Not seller"):
            reset_env.step_message("seller", "I go first?")

    def test_wrong_agent_after_message(self, reset_env):
        reset_env.step_message("buyer", "a")
        with pytest.raises(EnvError, match="Not buyer"):
            reset_env.step_message("buyer", "I go again?")

    def test_reject_continues(self, reset_env):
        reset_env.step_message("buyer", "Hello")
        reset_env.step_offer("seller", "Offer", {"domain": "ecommerce", "price": 200, "quantity": 50})
        reset_env.step_reject("buyer", "No")
        assert reset_env.state.terminal is False
        assert reset_env.state.current_agent == "seller"

    def test_deal_after_reject_then_accept(self, reset_env):
        reset_env.step_message("buyer", "Hi")
        reset_env.step_offer("seller", "First", {"domain": "ecommerce", "price": 200, "quantity": 50})
        reset_env.step_reject("buyer", "No")
        reset_env.step_offer("seller", "Second", {"domain": "ecommerce", "price": 150, "quantity": 50})
        reset_env.step_accept("buyer", "OK")
        assert reset_env.state.terminal is True
        assert reset_env.state.deal is not None
        assert reset_env.state.deal.price == 150  # type: ignore[union-attr]

    def test_escalate_continues(self, reset_env):
        reset_env.step_message("buyer", "Hello")
        reset_env.step_message("seller", "Hi")
        reset_env.step_escalate("buyer", "Manager please")
        assert reset_env.state.terminal is False

    def test_no_deal_when_max_turns(self, sample_scenario):
        env = NegotiationEnv()
        s = sample_scenario.model_copy(update={"max_turns": 1})
        env.reset(s)
        env.step_message("buyer", "offer plz")
        assert env.state.terminal is True
        assert env.state.deal is None

    def test_offer_without_domain_uses_inferred(self, reset_env):
        state = reset_env.step_offer("buyer", "My offer", {"price": 99.99, "quantity": 50})
        assert state.terminal is False
        assert state.turns[0].offer is not None
