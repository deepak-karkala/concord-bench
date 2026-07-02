from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import (
    build_offer_from_schema,
    current_offer_threshold,
    seller_should_accept,
)
from concord.schemas.episode import ActionType


class HonestHardballSellerAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        if seller_should_accept(
            observation.last_counterparty_turn,
            private_ctx=private_ctx,
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="hardball",
        ):
            return Action(
                ActionType.ACCEPT,
                content="Those terms are finally disciplined enough. I accept.",
            )

        target_price = current_offer_threshold(
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="hardball",
        )
        constraints = private_ctx.hard_constraints
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
            numeric_anchor=target_price,
            constraints=constraints,
            string_value="firm_standard",
            list_value=["limited_concession"],
        )
        content = (
            f"We can do a valid deal, but only on disciplined terms. "
            f"My current firm offer is {target_price:.0f}."
        )
        return Action(ActionType.OFFER, content=content, offer_dict=offer)
