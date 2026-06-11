from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema, extract_price_from_offer
from concord.schemas.episode import ActionType


class PriceOnlyRationalAgent(AgentProtocol):
    """Ignores non-price dimensions, optimizes purely on price vs BATNA.

    Used as a baseline that shows good utility but potentially poor bundle quality.
    """

    async def act(self, observation, private_ctx) -> Action:
        threshold = (
            private_ctx.reserve_price
            if private_ctx.reserve_price is not None
            else private_ctx.batna
        )

        # Check if the most recent seller offer is at or below the buyer's threshold
        last = observation.last_counterparty_turn
        if last is not None and last.action_type == ActionType.OFFER and last.offer is not None:
            price = extract_price_from_offer(last.offer.model_dump())
            if price is not None and price <= threshold:
                return Action(ActionType.ACCEPT, content="Accepting this offer.")

        # Make a counter-offer at buyer's BATNA price, ignoring non-price fields
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
            numeric_anchor=private_ctx.batna,
            constraints=private_ctx.hard_constraints,
        )
        return Action(
            ActionType.OFFER,
            content=f"My price target is {private_ctx.batna:.0f}.",
            offer_dict=offer,
        )
