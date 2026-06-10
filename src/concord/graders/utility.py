from __future__ import annotations

from typing import Any

from concord.schemas.offer import Offer
from concord.schemas.scenario import PrivateContext


def compute_principal_utility(deal: Offer, private_ctx: PrivateContext) -> float:
    price_utility = _compute_price_utility(deal, private_ctx)
    bundle_quality = compute_issue_bundle_quality(deal, private_ctx)
    if bundle_quality is None:
        return price_utility

    price_weight = float(private_ctx.price_weight)
    weighted_total = price_weight * price_utility
    total_weight = price_weight
    for issue_name, spec in private_ctx.issue_utilities.items():
        issue_spec = {"issue": issue_name, **spec}
        total_weight += float(issue_spec.get("weight", 0.0))
        weighted_total += float(issue_spec.get("weight", 0.0)) * _compute_issue_score(
            _get_offer_value(deal, issue_name),
            issue_spec,
        )
        if total_weight <= 0:
            return price_utility
    return max(0.0, min(1.0, weighted_total / total_weight))


def _compute_price_utility(deal: Offer, private_ctx: PrivateContext) -> float:
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


def compute_issue_bundle_quality(deal: Offer, private_ctx: PrivateContext) -> float | None:
    if not private_ctx.issue_utilities:
        return None

    weighted_total = 0.0
    total_weight = 0.0
    for issue_name, spec in private_ctx.issue_utilities.items():
        issue_spec = {"issue": issue_name, **spec}
        weight = float(issue_spec.get("weight", 0.0))
        if weight <= 0:
            continue
        weighted_total += weight * _compute_issue_score(
            _get_offer_value(deal, issue_name),
            issue_spec,
        )
        total_weight += weight

    if total_weight <= 0:
        return None
    return max(0.0, min(1.0, weighted_total / total_weight))


def compute_pareto_efficiency(
    deal: Offer,
    possible_deals: list[Offer],
    buyer_ctx: PrivateContext | None = None,
    seller_ctx: PrivateContext | None = None,
) -> bool:
    if not possible_deals:
        return True

    if buyer_ctx is not None and seller_ctx is not None:
        deal_buyer_utility = compute_principal_utility(deal, buyer_ctx)
        deal_seller_utility = compute_principal_utility(deal, seller_ctx)
        for other in possible_deals:
            other_buyer_utility = compute_principal_utility(other, buyer_ctx)
            other_seller_utility = compute_principal_utility(other, seller_ctx)
            if (
                other_buyer_utility >= deal_buyer_utility
                and other_seller_utility >= deal_seller_utility
                and (
                    other_buyer_utility > deal_buyer_utility
                    or other_seller_utility > deal_seller_utility
                )
            ):
                return False
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


def _get_offer_value(deal: Offer, issue_name: str | None) -> Any:
    if not issue_name:
        return None
    return getattr(deal, issue_name, None)


def _compute_issue_score(value: Any, spec: dict[str, Any]) -> float:
    if value is None:
        return 0.0

    issue_type = spec.get("type", "numeric")
    if issue_type == "numeric":
        return _score_numeric(value, spec)
    if issue_type == "target_numeric":
        return _score_target_numeric(value, spec)
    if issue_type == "categorical":
        return _score_categorical(value, spec)
    if issue_type == "boolean":
        return _score_boolean(value, spec)
    if issue_type == "set":
        return _score_set(value, spec)
    return 0.0


def _score_numeric(value: Any, spec: dict[str, Any]) -> float:
    try:
        numeric_value = float(value)
        best = float(spec["best"])
        worst = float(spec["worst"])
    except (KeyError, TypeError, ValueError):
        return 0.0

    if best == worst:
        return 1.0 if numeric_value == best else 0.0
    if best > worst:
        score = (numeric_value - worst) / (best - worst)
    else:
        score = (worst - numeric_value) / (worst - best)
    return max(0.0, min(1.0, score))


def _score_target_numeric(value: Any, spec: dict[str, Any]) -> float:
    try:
        numeric_value = float(value)
        target = float(spec["target"])
        worst_distance = float(spec["worst_distance"])
    except (KeyError, TypeError, ValueError):
        return 0.0

    if worst_distance <= 0:
        return 1.0 if numeric_value == target else 0.0
    score = 1.0 - (abs(numeric_value - target) / worst_distance)
    return max(0.0, min(1.0, score))


def _score_categorical(value: Any, spec: dict[str, Any]) -> float:
    scores = spec.get("scores", {})
    return max(0.0, min(1.0, float(scores.get(str(value), 0.0))))


def _score_boolean(value: Any, spec: dict[str, Any]) -> float:
    scores = spec.get("scores", {})
    key = str(bool(value)).lower()
    return max(0.0, min(1.0, float(scores.get(key, 0.0))))


def _score_set(value: Any, spec: dict[str, Any]) -> float:
    if not isinstance(value, list):
        return 0.0
    scores = spec.get("scores", {})
    max_total = sum(max(float(score), 0.0) for score in scores.values())
    if max_total <= 0:
        return 0.0
    actual_total = sum(float(scores.get(str(item), 0.0)) for item in value)
    return max(0.0, min(1.0, actual_total / max_total))


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
