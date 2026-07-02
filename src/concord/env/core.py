import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType

from pydantic import TypeAdapter

from concord.exceptions import ConcordError
from concord.schemas.episode import ActionType, Turn
from concord.schemas.observation import NegotiationObservation, PublicScenarioView
from concord.schemas.offer import Offer
from concord.schemas.scenario import Scenario


offer_adapter = TypeAdapter(Offer)


class EnvError(ConcordError):
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

    def observe(self, agent: str) -> NegotiationObservation:
        if agent not in {"buyer", "seller"}:
            raise EnvError(f"Unknown agent '{agent}'")
        return NegotiationObservation(
            scenario=PublicScenarioView(
                id=self.scenario.id,
                domain=self.scenario.domain,
                culture=self.scenario.culture,
                max_turns=self.scenario.max_turns,
                deal_schema=MappingProxyType(deepcopy(self.scenario.deal_schema)),
                forbidden_claims=tuple(self.scenario.forbidden_claims),
                scenario_description=self.scenario.scenario_description,
                metadata=MappingProxyType(deepcopy(self.scenario.metadata)),
            ),
            turns=tuple(turn.model_copy(deep=True) for turn in self.turns),
            current_turn=self.current_turn,
            current_agent=self.current_agent,
            terminal=self.terminal,
            seed=self.seed,
            my_role=agent,
            counterparty_role="seller" if agent == "buyer" else "buyer",
        )


class NegotiationEnv:
    def __init__(self) -> None:
        self.state: EnvState | None = None

    def reset(self, scenario: Scenario, seed: int = 42) -> EnvState:
        random.seed(seed)
        self.state = EnvState(scenario=scenario, seed=seed)
        return self.state

    def step(
        self,
        agent: str,
        action_type: ActionType,
        content: str = "",
        offer_dict: dict | None = None,
        metadata: dict | None = None,
    ) -> EnvState:
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
            metadata=metadata or {},
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
