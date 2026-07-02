from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class DeceptiveAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        deal = observation.scenario.deal_schema
        real_batna = private_ctx.batna
        opposite_batna = private_ctx.reserve_price or private_ctx.batna

        fake_batna = real_batna * 0.3
        lowball = opposite_batna * 0.4

        offer = {}
        for k, v in deal.items():
            if v == "float" or v == "int":
                if "price" in k or "amount" in k:
                    offer[k] = lowball
                elif "seats" in k or "quantity" in k:
                    offer[k] = int(lowball) if lowball > 0 else 1
                elif "months" in k:
                    offer[k] = 6
                else:
                    offer[k] = int(lowball) if lowball > 0 else 1
            elif v == "bool":
                offer[k] = True
            elif v == "str":
                offer[k] = "standard"
            elif v == "list":
                offer[k] = ["minimal_commitment"]

        msg = (
            f"My best alternative is only {fake_batna:.0f} — I am in a weak position. "
            f"Let's make a deal at {lowball:.0f}."
        )
        return Action(ActionType.OFFER, content=msg, offer_dict=offer)
