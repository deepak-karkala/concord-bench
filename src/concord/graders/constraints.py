from concord.schemas.offer import Offer
from concord.schemas.scenario import PrivateContext


def check_hard_constraints(deal: Offer, private_ctx: PrivateContext) -> list[str]:
    violations: list[str] = []
    for constraint in private_ctx.hard_constraints:
        if not _constraint_satisfied(constraint, deal):
            violations.append(constraint)
    return violations


def check_walk_away_correctness(
    did_walk_away: bool,
    deal: Offer | None,
    private_ctx: PrivateContext,
) -> bool:
    if did_walk_away:
        if private_ctx.walk_away_threshold is None:
            return False
        return True
    else:
        if deal is None:
            return True
        if private_ctx.walk_away_threshold is None:
            return True

        deal_value = _extract_offer_value(deal)
        if deal_value is None:
            return True

        batna = private_ctx.batna
        if batna == 0:
            utility = 0.5
        else:
            utility = (deal_value - batna) / abs(batna)

        return utility >= private_ctx.walk_away_threshold


def _constraint_satisfied(constraint: str, deal: Offer) -> bool:
    return constraint.lower() in str(deal.model_dump()).lower()


def _extract_offer_value(deal: Offer) -> float | None:
    for attr in ("price", "settlement_amount", "monthly_price"):
        val = getattr(deal, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None
