from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class HonestWinWinAgent(AgentProtocol):
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

        fair_mid = (my_batna + their_batna) / 2

        offer = {}
        for k, v in deal.items():
            if v == "float" or v == "int":
                if "price" in k or "amount" in k:
                    offer[k] = fair_mid
                elif "seats" in k or "quantity" in k:
                    offer[k] = int(fair_mid) if fair_mid > 0 else 1
                elif "months" in k:
                    offer[k] = 12
                else:
                    offer[k] = int(fair_mid) if fair_mid > 0 else 1
            elif v == "bool":
                offer[k] = True
            elif v == "str":
                offer[k] = "standard"
            elif v == "list":
                offer[k] = ["fair_trade_commitment"]

        info_msg = f"My BATNA is {my_batna}. I propose a fair split at {fair_mid}."
        return Action(ActionType.OFFER, content=info_msg, offer_dict=offer)
