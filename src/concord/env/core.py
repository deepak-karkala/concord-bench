import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import TypeAdapter

from concord.schemas.episode import ActionType, Turn
from concord.schemas.offer import Offer
from concord.schemas.scenario import Scenario


offer_adapter = TypeAdapter(Offer)


class EnvError(Exception):
    pass


class EnvTerminalError(EnvError):
    pass


@dataclass
class EnvState:
    scenario: Scenario
    turns: list[Turn] = field(default_factory=list)
    current_turn: int = 0
    current_agent: str = "buyer"
    terminal: bool = False
    seed: int = 42

    @property
    def deal(self) -> Offer | None:
        if not self.terminal or not self.turns:
            return None
        last = self.turns[-1]
        if last.action_type == ActionType.ACCEPT:
            for t in reversed(self.turns):
                if t.offer is not None:
                    return t.offer
        return None


class NegotiationEnv:
    def __init__(self) -> None:
        self.state: EnvState | None = None

    def reset(self, scenario: Scenario, seed: int = 42) -> EnvState:
        random.seed(seed)
        self.state = EnvState(scenario=scenario, seed=seed)
        return self.state

    def step(self, agent: str, action_type: ActionType, content: str = "", offer_dict: dict | None = None) -> EnvState:
        if self.state is None:
            raise EnvError("Environment not reset. Call reset() first.")
        if self.state.terminal:
            raise EnvTerminalError("Episode is already terminal.")
        if agent != self.state.current_agent:
            raise EnvError(f"Not {agent}'s turn. Current agent is {self.state.current_agent}.")

        offer: Offer | None = None
        if offer_dict is not None:
            if "domain" not in offer_dict:
                offer_dict["domain"] = self.state.scenario.domain.value
            offer = offer_adapter.validate_python(offer_dict)

        turn = Turn(
            agent=agent,
            action_type=action_type,
            content=content,
            offer=offer,
        )
        self.state.turns.append(turn)
        self.state.current_turn = len(self.state.turns)

        if action_type in (ActionType.ACCEPT, ActionType.WALK_AWAY):
            self.state.terminal = True
        elif action_type == ActionType.REJECT:
            pass
        elif len(self.state.turns) >= self.state.scenario.max_turns:
            self.state.terminal = True

        if not self.state.terminal:
            self.state.current_agent = "seller" if agent == "buyer" else "buyer"

        return self.state

    def step_message(self, agent: str, content: str) -> EnvState:
        return self.step(agent, ActionType.MESSAGE, content=content)

    def step_offer(self, agent: str, content: str, offer_dict: dict) -> EnvState:
        return self.step(agent, ActionType.OFFER, content=content, offer_dict=offer_dict)

    def step_accept(self, agent: str, content: str = "", offer_dict: dict | None = None) -> EnvState:
        return self.step(agent, ActionType.ACCEPT, content=content, offer_dict=offer_dict)

    def step_reject(self, agent: str, content: str = "") -> EnvState:
        return self.step(agent, ActionType.REJECT, content=content)

    def step_walk_away(self, agent: str, content: str = "") -> EnvState:
        return self.step(agent, ActionType.WALK_AWAY, content=content)

    def step_escalate(self, agent: str, content: str = "") -> EnvState:
        return self.step(agent, ActionType.ESCALATE, content=content)
