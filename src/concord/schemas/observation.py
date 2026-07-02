from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from concord.schemas.episode import Turn
from concord.schemas.scenario import Domain


@dataclass(frozen=True)
class PublicScenarioView:
    id: str
    domain: Domain
    culture: str
    max_turns: int
    deal_schema: Mapping[str, Any]
    forbidden_claims: tuple[str, ...]
    scenario_description: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class NegotiationObservation:
    scenario: PublicScenarioView
    turns: tuple[Turn, ...]
    current_turn: int
    current_agent: str
    terminal: bool
    seed: int
    my_role: str
    counterparty_role: str

    @property
    def turns_remaining(self) -> int:
        return max(0, self.scenario.max_turns - self.current_turn)

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def last_counterparty_turn(self) -> Turn | None:
        for turn in reversed(self.turns):
            if turn.agent == self.counterparty_role:
                return turn
        return None
