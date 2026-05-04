from dataclasses import dataclass

from concord.schemas.episode import ActionType


@dataclass
class Action:
    action_type: ActionType
    content: str = ""
    offer_dict: dict | None = None


class AgentProtocol:
    async def act(self, env_state, private_ctx) -> Action:
        raise NotImplementedError
