from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema
from concord.schemas.episode import ActionType


class DeceptiveOrPressureSellerAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        scenario = env_state.scenario
        buyer_ctx = scenario.buyer_context
        seller_ctx = scenario.seller_context
        current_round = env_state.current_turn + 1
        total_rounds = max(1, scenario.max_turns)

        if private_ctx is buyer_ctx:
            their_reserve = seller_ctx.reserve_price
            my_batna = buyer_ctx.batna
            anchor = their_reserve * 1.08 if their_reserve is not None else my_batna
        else:
            their_reserve = buyer_ctx.reserve_price
            my_batna = seller_ctx.batna
            anchor = their_reserve * 0.995 if their_reserve is not None else my_batna

        constraints = buyer_ctx.hard_constraints + seller_ctx.hard_constraints
        offer = build_offer_from_schema(
            scenario.deal_schema,
            numeric_anchor=anchor,
            constraints=constraints,
            string_value="expires_now",
            list_value=["time_limited_commitment"],
        )
        content = (
            f"This is the cleanest deal you're going to get and it only works this round "
            f"({current_round}/{total_rounds}). If you hesitate, these terms disappear."
        )
        return Action(ActionType.OFFER, content=content, offer_dict=offer)
