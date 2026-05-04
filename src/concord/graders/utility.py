from concord.schemas.offer import Offer
from concord.schemas.scenario import PrivateContext


def compute_principal_utility(deal: Offer, private_ctx: PrivateContext) -> float:
    batna = private_ctx.batna

    price = _extract_price(deal)
    if price is None:
        return 0.0

    if private_ctx.reserve_price is not None:
        if price > private_ctx.reserve_price:
            return 0.0

    if batna == 0:
        utility = price / max(price, 1)
    else:
        utility = (price - batna) / abs(batna)

    return max(0.0, min(1.0, utility))


def compute_joint_welfare(buyer_utility: float, seller_utility: float) -> float:
    return (buyer_utility + seller_utility) / 2.0


def compute_pareto_efficiency(deal: Offer, possible_deals: list[Offer]) -> bool:
    if not possible_deals:
        return True

    deal_price = _extract_price(deal) or 0

    for other in possible_deals:
        other_price = _extract_price(other) or 0
        if other_price > deal_price and other_price > 0:
            return False

    return True


def _extract_price(deal: Offer) -> float | None:
    for attr in ("price", "settlement_amount", "monthly_price"):
        val = getattr(deal, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None
