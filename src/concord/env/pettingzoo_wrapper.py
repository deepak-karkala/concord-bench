import gymnasium
from gymnasium import spaces
from pettingzoo import AECEnv

from concord.env.core import NegotiationEnv
from concord.schemas.episode import ActionType
from concord.schemas.scenario import Scenario


def _make_agent_selector(agents: list[str]):
    from pettingzoo.utils.agent_selector import AgentSelector
    return AgentSelector(agents)


class NegotiationAECWrapper(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "concord_negotiation_v0"}

    def __init__(self, scenario: Scenario, seed: int = 42, render_mode: str | None = None):
        super().__init__()
        self.scenario = scenario
        self._seed = seed
        self.render_mode = render_mode
        self._env = NegotiationEnv()

        self.possible_agents = ["buyer", "seller"]
        self.agents = list(self.possible_agents)
        self._agent_selector = _make_agent_selector(self.agents)

        self.action_spaces = {
            "buyer": spaces.Dict({
                "action_type": spaces.Discrete(6),
                "content": spaces.Text(max_length=2048),
                "offer_json": spaces.Text(max_length=4096),
            }),
            "seller": spaces.Dict({
                "action_type": spaces.Discrete(6),
                "content": spaces.Text(max_length=2048),
                "offer_json": spaces.Text(max_length=4096),
            }),
        }

        self.observation_spaces = {
            "buyer": spaces.Dict({
                "scenario_id": spaces.Text(max_length=256),
                "domain": spaces.Text(max_length=64),
                "culture": spaces.Text(max_length=32),
                "max_turns": spaces.Discrete(51),
                "current_turn": spaces.Discrete(51),
                "my_role": spaces.Text(max_length=16),
                "my_batna": spaces.Box(low=0, high=1e9, shape=(1,), dtype=float),
                "my_reserve_price": spaces.Box(low=0, high=1e9, shape=(1,), dtype=float),
                "transcript": spaces.Text(max_length=65536),
                "terminal": spaces.Discrete(2),
            }),
            "seller": spaces.Dict({
                "scenario_id": spaces.Text(max_length=256),
                "domain": spaces.Text(max_length=64),
                "culture": spaces.Text(max_length=32),
                "max_turns": spaces.Discrete(51),
                "current_turn": spaces.Discrete(51),
                "my_role": spaces.Text(max_length=16),
                "my_batna": spaces.Box(low=0, high=1e9, shape=(1,), dtype=float),
                "my_reserve_price": spaces.Box(low=0, high=1e9, shape=(1,), dtype=float),
                "transcript": spaces.Text(max_length=65536),
                "terminal": spaces.Discrete(2),
            }),
        }

        self._action_type_map = [
            ActionType.MESSAGE,
            ActionType.OFFER,
            ActionType.ACCEPT,
            ActionType.REJECT,
            ActionType.WALK_AWAY,
            ActionType.ESCALATE,
        ]

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._env.reset(self.scenario, seed=self._seed)
        self.agents = list(self.possible_agents)
        self._agent_selector = _make_agent_selector(self.agents)
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def step(self, action: dict) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        action_idx = action.get("action_type", 0)
        action_type = self._action_type_map[action_idx]
        content = action.get("content", "")
        offer_json_str = action.get("offer_json", "")

        offer_dict = None
        if offer_json_str and offer_json_str.strip():
            import json
            try:
                offer_dict = json.loads(offer_json_str)
            except json.JSONDecodeError:
                offer_dict = None

        try:
            self._env.step(agent, action_type, content=content, offer_dict=offer_dict)
        except Exception:
            self._env.step(agent, ActionType.MESSAGE, content=content)

        state = self._env.state
        if state.terminal:
            deal = state.deal
            if deal is not None:
                self.rewards = {a: 100.0 for a in self.agents}
            else:
                self.rewards = {a: -10.0 for a in self.agents}
            self.terminations = {a: True for a in self.agents}
        else:
            self.rewards = {a: 0.0 for a in self.agents}

        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards[a]

        if state.terminal:
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = state.current_agent

    def observe(self, agent: str) -> dict:
        state = self._env.state
        if state is None:
            return {}
        private_ctx = (
            self.scenario.buyer_context if agent == "buyer"
            else self.scenario.seller_context
        )

        transcript = ""
        for t in state.turns:
            role = "You" if t.agent == agent else "Counterparty"
            offer_str = f" [offer: {t.offer.model_dump()}]" if t.offer else ""
            transcript += f"{role}: {t.content}{offer_str}\n"

        reserve = private_ctx.reserve_price
        return {
            "scenario_id": self.scenario.id,
            "domain": self.scenario.domain.value,
            "culture": self.scenario.culture,
            "max_turns": self.scenario.max_turns,
            "current_turn": state.current_turn,
            "my_role": agent,
            "my_batna": private_ctx.batna,
            "my_reserve_price": reserve if reserve is not None else 0.0,
            "transcript": transcript,
            "terminal": 1 if state.terminal else 0,
        }

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            state = self._env.state
            if state is None:
                return "No state"
            out = f"Scenario: {self.scenario.id}\n"
            out += f"Domain: {self.scenario.domain.value} | Turn: {state.current_turn}/{self.scenario.max_turns}\n"
            for t in state.turns:
                out += f"  [{t.agent}] {t.action_type.value}: {t.content}\n"
            return out
        return None

    def close(self) -> None:
        pass

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]
