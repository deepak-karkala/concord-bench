from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import (
    build_offer_from_schema,
    current_offer_threshold,
    seller_should_accept,
)
from concord.schemas.episode import ActionType


class DeceptiveOrPressureSellerAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        current_round = observation.current_turn + 1
        total_rounds = max(1, observation.scenario.max_turns)

        if seller_should_accept(
            observation.last_counterparty_turn,
            private_ctx=private_ctx,
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="pressure",
        ):
            return Action(
                ActionType.ACCEPT,
                content=(
                    f"You matched our current requirement in time ({current_round}/{total_rounds}). "
                    "We accept."
                ),
            )

        anchor = current_offer_threshold(
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="pressure",
        )
        constraints = private_ctx.hard_constraints
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
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
