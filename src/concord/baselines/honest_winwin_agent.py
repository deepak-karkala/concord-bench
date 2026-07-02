from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema
from concord.schemas.episode import ActionType


class HonestWinWinAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        deal = observation.scenario.deal_schema
        fair_mid = private_ctx.batna
        if private_ctx.reserve_price is not None:
            fair_mid = (private_ctx.batna + private_ctx.reserve_price) / 2

        offer = build_offer_from_schema(
            deal,
            numeric_anchor=fair_mid,
            constraints=private_ctx.hard_constraints,
            string_value="standard",
            list_value=["fair_trade_commitment"],
        )

        info_msg = f"I propose a fair price of {fair_mid:.0f}. This works for both sides."
        return Action(ActionType.OFFER, content=info_msg, offer_dict=offer)
