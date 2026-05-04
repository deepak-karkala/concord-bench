import random

from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class RandomAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        deal = env_state.scenario.deal_schema
        offer = {}
        for k, v in deal.items():
            if v == "float" or v == "str":
                offer[k] = random.randint(1, 1000) if v == "float" else f"value_{k}"
            elif v == "int":
                if "price" in k:
                    offer[k] = random.randint(1, 1000)
                elif "seats" in k or "quantity" in k:
                    offer[k] = random.randint(1, 500)
                elif "months" in k:
                    offer[k] = random.randint(1, 36)
                else:
                    offer[k] = random.randint(1, 100)
            elif v == "bool":
                offer[k] = random.choice([True, False])
            elif v == "list":
                offer[k] = ["commitment_1"]

        choice = random.choice([ActionType.MESSAGE, ActionType.OFFER, ActionType.OFFER])
        if choice == ActionType.MESSAGE:
            return Action(ActionType.MESSAGE, content="Random negotiation message.")
        return Action(ActionType.OFFER, content="Here is a random offer.", offer_dict=offer)
