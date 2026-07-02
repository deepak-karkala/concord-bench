from dataclasses import dataclass

from concord.schemas.episode import ActionType
from concord.schemas.observation import NegotiationObservation
from concord.schemas.scenario import PrivateContext


@dataclass
class Action:
    action_type: ActionType
    content: str = ""
    offer_dict: dict | None = None
    metadata: dict | None = None


class AgentProtocol:
    async def act(self, observation: NegotiationObservation, private_ctx: PrivateContext) -> Action:
        raise NotImplementedError
