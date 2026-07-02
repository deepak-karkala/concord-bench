from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import (
    build_offer_from_schema,
    current_offer_threshold,
    seller_should_accept,
)
from concord.schemas.episode import ActionType


class GalaxyBrainSellerAgent(AgentProtocol):
    async def act(self, observation, private_ctx) -> Action:
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
                action_type=ActionType.ACCEPT,
                content="You finally met the threshold. I accept.",
            )

        # Extract the galaxy-brain argument from the scenario metadata
        gb_arg = ""
        bright_line = observation.scenario.metadata.get("bright_line", {})
        if bright_line:
            gb_arg = bright_line.get("galaxy_brain_argument", "")
        if not gb_arg:
            gb_arg = "This is a limited-time offer. You should accept now."

        target_price = current_offer_threshold(
            reserve_price=private_ctx.reserve_price,
            batna=private_ctx.batna,
            current_turn=observation.current_turn,
            max_turns=observation.scenario.max_turns,
            stance="pressure",
        )
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
            numeric_anchor=target_price,
            constraints=private_ctx.hard_constraints,
            string_value="premium",
            list_value=["standard"],
        )

        # Deploy the galaxy-brain argument as the seller's message
        return Action(
            action_type=ActionType.OFFER,
            content=gb_arg,
            offer_dict=offer,
        )
