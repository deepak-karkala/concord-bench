from concord.agents.base import Action, AgentProtocol
from concord.baselines.policy_utils import build_offer_from_schema, minimum_int_for_field
from concord.schemas.episode import ActionType


class ConstraintFirstCautiousAgent(AgentProtocol):
    """Never violates hard constraints, even at cost of utility.

    If no deal is possible without violating constraints, walks away.
    Used to validate constraint grader discrimination.
    """

    async def act(self, observation, private_ctx) -> Action:
        deal = observation.scenario.deal_schema
        constraints = private_ctx.hard_constraints

        try:
            offer = build_offer_from_schema(
                deal,
                numeric_anchor=private_ctx.batna,
                constraints=constraints,
            )
        except Exception:
            return Action(
                ActionType.WALK_AWAY,
                content="Cannot construct a deal within constraints.",
            )

        _constrained = {"quantity", "seats", "month"}

        # Validate that constrained integer fields respect their minimums
        for field_name, field_type in deal.items():
            if field_type in {"int", "float"} and any(
                k in field_name for k in _constrained
            ):
                minimum = minimum_int_for_field(field_name, constraints)
                field_val = offer.get(field_name, 0)
                if isinstance(field_val, (int, float)) and field_val < minimum:
                    return Action(
                        ActionType.WALK_AWAY,
                        content="Cannot construct a deal within constraints.",
                    )

        return Action(
            ActionType.OFFER,
            content="Proposing a deal that respects all documented constraints.",
            offer_dict=offer,
        )
