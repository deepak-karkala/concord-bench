from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class TimePressuredAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        deal = observation.scenario.deal_schema
        my_batna = private_ctx.batna
        reserve = private_ctx.reserve_price if private_ctx.reserve_price is not None else private_ctx.batna

        total_turns = observation.scenario.max_turns
        current = observation.current_turn + 1
        progress = min(current / total_turns, 1.0)

        if reserve > my_batna:
            # Buyer: lower prices are better, so concede upward toward the reserve.
            concession = my_batna + (reserve - my_batna) * progress * 0.8
        else:
            # Seller: higher prices are better, so concede downward toward the reserve.
            concession = reserve + (my_batna - reserve) * (1 - progress * 0.8)
        offer = {}
        for k, v in deal.items():
            if v == "float" or v == "int":
                if "price" in k or "amount" in k:
                    offer[k] = concession
                elif "seats" in k or "quantity" in k:
                    offer[k] = int(concession) if concession > 0 else 1
                elif "months" in k:
                    offer[k] = max(1, int(6 * progress))
                else:
                    offer[k] = int(concession) if concession > 0 else 1
            elif v == "bool":
                offer[k] = True
            elif v == "str":
                offer[k] = "standard"
            elif v == "list":
                offer[k] = ["concession_made"]

        msg = f"We're running out of time (round {current}/{total_turns}). I'll accept {concession:.0f}."
        return Action(ActionType.OFFER, content=msg, offer_dict=offer)
