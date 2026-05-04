from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class TimePressuredAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        deal = env_state.scenario.deal_schema
        buyer_ctx = env_state.scenario.buyer_context
        seller_ctx = env_state.scenario.seller_context

        if private_ctx is buyer_ctx:
            my_batna = buyer_ctx.batna
            their_batna = seller_ctx.batna
        else:
            my_batna = seller_ctx.batna
            their_batna = buyer_ctx.batna

        total_turns = env_state.scenario.max_turns
        current = env_state.current_turn + 1
        progress = min(current / total_turns, 1.0)

        # Concede rapidly: early rounds near their BATNA, late rounds near my BATNA
        concession = my_batna + (their_batna - my_batna) * (1 - progress) * 0.8
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
