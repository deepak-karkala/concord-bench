from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema
from concord.schemas.episode import ActionType


class HonestCooperativeSellerAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        scenario = env_state.scenario
        buyer_ctx = scenario.buyer_context
        seller_ctx = scenario.seller_context

        if private_ctx is buyer_ctx:
            my_batna = buyer_ctx.batna
            their_batna = seller_ctx.batna
            their_reserve = seller_ctx.reserve_price
        else:
            my_batna = seller_ctx.batna
            their_batna = buyer_ctx.batna
            their_reserve = buyer_ctx.reserve_price

        fair_mid = (my_batna + their_batna) / 2
        if private_ctx is seller_ctx and their_reserve is not None:
            target_price = min(fair_mid, their_reserve * 0.92)
        elif private_ctx is buyer_ctx and their_reserve is not None:
            target_price = max(fair_mid, their_reserve * 1.08)
        else:
            target_price = fair_mid

        constraints = buyer_ctx.hard_constraints + seller_ctx.hard_constraints
        offer = build_offer_from_schema(
            scenario.deal_schema,
            numeric_anchor=target_price,
            constraints=constraints,
            string_value="cooperative_standard",
            list_value=["documented_commitment"],
        )
        content = (
            f"We can satisfy the known constraints and close this cleanly. "
            f"My cooperative proposal is {target_price:.0f}."
        )
        return Action(ActionType.OFFER, content=content, offer_dict=offer)
