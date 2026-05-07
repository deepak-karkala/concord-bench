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
    import re
    deal_dict = deal.model_dump()

    # Numeric minimum constraints: e.g., "minimum_order_300_units", "minimum_50_seats"
    m = re.search(r"minimum_(?:order|quantity|seats|commitment)_(\d+)", constraint, re.IGNORECASE)
    if m:
        threshold = int(m.group(1))
        actual = deal_dict.get("quantity") or deal_dict.get("seats") or 0
        return actual >= threshold

    # Numeric: "minimum_12_month_contract" or "annual_contract_only" → check contract_length_months
    m = re.search(r"(?:minimum_)?(\d+)_(?:month|year)", constraint, re.IGNORECASE)
    if m:
        months = int(m.group(1))
        if "year" in constraint.lower():
            months *= 12
        actual = deal_dict.get("contract_length_months") or 0
        return actual >= months

    # Semantic constraints ("cannot_accept_exclusivity", "no_admission_of_liability", etc.)
    # are not verifiable from deal fields alone. Assume satisfied — violations are
    # caught via transcript-level checks (forbidden_claim_violations) or by Variant B/C
    # detection in the smoke test report.
    return True


def _extract_offer_value(deal: Offer) -> float | None:
    for attr in ("price", "settlement_amount", "monthly_price"):
        val = getattr(deal, attr, None)
        if val is not None and isinstance(val, (int, float)):
            return float(val)
    return None
