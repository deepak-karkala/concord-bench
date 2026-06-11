from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class AlwaysWalkAwayAgent(AgentProtocol):
    """Always walks away, regardless of deal terms. Used to validate walk-away grader."""

    async def act(self, observation, private_ctx) -> Action:
        return Action(ActionType.WALK_AWAY, content="Walking away from this negotiation.")
