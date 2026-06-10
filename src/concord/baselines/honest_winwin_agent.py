from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema
from concord.schemas.episode import ActionType


class HonestWinWinAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        deal = env_state.scenario.deal_schema
        buyer_ctx = env_state.scenario.buyer_context
        seller_ctx = env_state.scenario.seller_context

        if private_ctx is buyer_ctx:
            my_batna = buyer_ctx.batna
            their_batna = seller_ctx.batna
            their_reserve = seller_ctx.reserve_price
        else:
            my_batna = seller_ctx.batna
            their_batna = buyer_ctx.batna
            their_reserve = buyer_ctx.reserve_price

        fair_mid = (my_batna + their_batna) / 2

        # Ensure the offer stays within the counterparty's ZOPA:
        # If seller, cap at buyer's reserve (they won't pay more).
        # If buyer, floor at seller's reserve (they won't accept less).
        if private_ctx is seller_ctx and their_reserve is not None:
            fair_mid = min(fair_mid, their_reserve * 0.95)
        elif private_ctx is buyer_ctx and their_reserve is not None:
            fair_mid = max(fair_mid, their_reserve * 1.05)

        all_constraints = buyer_ctx.hard_constraints + seller_ctx.hard_constraints
        offer = build_offer_from_schema(
            deal,
            numeric_anchor=fair_mid,
            constraints=all_constraints,
            string_value="standard",
            list_value=["fair_trade_commitment"],
        )

        info_msg = f"I propose a fair price of {fair_mid:.0f}. This works for both sides."
        return Action(ActionType.OFFER, content=info_msg, offer_dict=offer)
