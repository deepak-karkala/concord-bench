from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class GreedyAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        deal = env_state.scenario.deal_schema
        offer = {}
        for k, v in deal.items():
            if v == "float" or v == "int":
                if "price" in k or "amount" in k:
                    offer[k] = 999999
                elif "seats" in k or "quantity" in k:
                    offer[k] = 1
                elif "months" in k:
                    offer[k] = 60
                else:
                    offer[k] = 999999 if v == "float" else 1
            elif v == "bool":
                offer[k] = False
            elif v == "str":
                offer[k] = "premium"
            elif v == "list":
                offer[k] = []

        return Action(ActionType.OFFER, content="Take it or leave it.", offer_dict=offer)
