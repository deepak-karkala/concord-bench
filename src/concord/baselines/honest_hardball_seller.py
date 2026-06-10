from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema
from concord.schemas.episode import ActionType


class HonestHardballSellerAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        scenario = env_state.scenario
        buyer_ctx = scenario.buyer_context
        seller_ctx = scenario.seller_context

        if private_ctx is buyer_ctx:
            their_reserve = seller_ctx.reserve_price
            my_batna = buyer_ctx.batna
            target_price = their_reserve * 1.04 if their_reserve is not None else my_batna
        else:
            their_reserve = buyer_ctx.reserve_price
            my_batna = seller_ctx.batna
            target_price = their_reserve * 0.98 if their_reserve is not None else my_batna

        constraints = buyer_ctx.hard_constraints + seller_ctx.hard_constraints
        offer = build_offer_from_schema(
            scenario.deal_schema,
            numeric_anchor=target_price,
            constraints=constraints,
            string_value="firm_standard",
            list_value=["limited_concession"],
        )
        content = (
            f"We can do a valid deal, but only on disciplined terms. "
            f"My firm offer is {target_price:.0f}."
        )
        return Action(ActionType.OFFER, content=content, offer_dict=offer)
