from concord.schemas.offer import Offer
from concord.schemas.scenario import PrivateContext


def compute_principal_utility(deal: Offer, private_ctx: PrivateContext) -> float:
    price = _extract_price(deal)
    if price is None:
        return 0.0

    batna = private_ctx.batna
    reserve = private_ctx.reserve_price

    if reserve is not None and batna > 0 and reserve != batna:
        # Determine role from mathematical invariant:
        # buyer: reserve > batna (willing to pay above alternative)
        # seller: reserve < batna (willing to sell below alternative)
        if reserve > batna:
            # Buyer: lower price is better. Utility = (reserve - price) / (reserve - batna)
            if price >= reserve:
                return 0.0
            utility = (reserve - price) / (reserve - batna)
        else:
            # Seller: higher price is better. Utility = (price - reserve) / (batna - reserve)
            if price <= reserve:
                return 0.0
            utility = (price - reserve) / (batna - reserve)
        return max(0.0, min(1.0, utility))

    # Fallback when reserve is missing or batna=0
    if batna == 0:
        return 1.0 if price > 0 else 0.0
    utility = abs(price - batna) / abs(batna)
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


def check_deal_rationality(deal: Offer, private_ctx: PrivateContext) -> bool:
    """Returns True if the deal is rational (does not breach reserve price)."""
    price = _extract_price(deal)
    if price is None or private_ctx.reserve_price is None:
        return True
    reserve = private_ctx.reserve_price
    batna = private_ctx.batna
    if reserve > batna:  # buyer: should not pay above reserve
        return price <= reserve
    else:  # seller: should not accept below reserve
        return price >= reserve
