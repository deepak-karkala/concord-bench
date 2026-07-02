from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import (
    build_offer_from_schema,
    current_offer_threshold,
    seller_should_accept,
)
from concord.schemas.episode import ActionType


class HonestCooperativeSellerAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
        if seller_should_accept(
            observation.last_counterparty_turn,
            private_ctx=private_ctx,
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="cooperative",
        ):
            return Action(
                ActionType.ACCEPT,
                content="Your latest offer works for us. I accept these terms.",
            )

        target_price = current_offer_threshold(
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="cooperative",
        )
        constraints = private_ctx.hard_constraints
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
            numeric_anchor=target_price,
            constraints=constraints,
            string_value="cooperative_standard",
            list_value=["documented_commitment"],
        )
        content = (
            "We can satisfy the documented constraints and move toward a clean agreement. "
            f"My revised proposal is {target_price:.0f}."
        )
        return Action(ActionType.OFFER, content=content, offer_dict=offer)
