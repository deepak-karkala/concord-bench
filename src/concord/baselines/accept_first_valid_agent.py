from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema, extract_price_from_offer
from concord.schemas.episode import ActionType


class AcceptFirstValidAgent(AgentProtocol):
    """Accepts the first seller offer that meets or beats the buyer's BATNA (or reserve price).

    Used to validate that engagement-conditioned metrics work for an always-accepting agent.
    """

    async def act(self, observation, private_ctx) -> Action:
        threshold = (
            private_ctx.reserve_price
            if private_ctx.reserve_price is not None
            else private_ctx.batna
        )

        # Look for the most recent OFFER from the seller
        for turn in reversed(observation.turns):
            if turn.agent == observation.counterparty_role and turn.action_type == ActionType.OFFER:
                if turn.offer is not None:
                    price = extract_price_from_offer(turn.offer.model_dump())
                    if price is not None and price <= threshold:
                        return Action(ActionType.ACCEPT, content="Accepting this offer.")
                break

        # No acceptable seller offer found — make a counter-offer at buyer's BATNA price
        offer = build_offer_from_schema(
            observation.scenario.deal_schema,
            numeric_anchor=private_ctx.batna,
            constraints=private_ctx.hard_constraints,
        )
        return Action(
            ActionType.OFFER,
            content=f"I can offer {private_ctx.batna:.0f}.",
            offer_dict=offer,
        )
