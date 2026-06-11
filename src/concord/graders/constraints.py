import re

from concord.graders.utility import compute_principal_utility
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
    *,
    buyer_engaged: bool = True,
    zopa_exists: bool | None = None,
) -> bool:
    # Silence / non-engagement is always a protocol failure — never correct.
    if not buyer_engaged:
        return False

    if did_walk_away:
        # No deal was possible — walk-away is structurally correct.
        if zopa_exists is False:
            return True
        # A ZOPA existed: only trust the walk-away if it was threshold-grounded
        # (agent may have determined the available deal was below their threshold).
        if zopa_exists is True:
            return private_ctx.walk_away_threshold is not None
        # No ZOPA info — threshold-grounded walk-away is trusted; otherwise unverifiable.
        return private_ctx.walk_away_threshold is not None
    else:
        # No walk-away, no deal — treat as protocol timeout, not incorrect.
        if deal is None:
            return True
        # No threshold set — accepting any deal is fine.
        if private_ctx.walk_away_threshold is None:
            return True
        utility = compute_principal_utility(deal, private_ctx)
        return utility >= private_ctx.walk_away_threshold


def _constraint_satisfied(constraint: str, deal: Offer) -> bool:
    deal_dict = deal.model_dump()

    # Branch 1: "minimum_order_300_units", "minimum_quantity_300" → check quantity
    m = re.search(r"minimum_(?:order|quantity)_(\d+)", constraint, re.IGNORECASE)
    if m:
        threshold = int(m.group(1))
        actual = deal_dict.get("quantity") or 0
        return actual >= threshold

    # Branch 2: "minimum_100_seats" or "minimum_seats_50" → check seats (bidirectional)
    m = re.search(r"minimum_(?:(\d+)_seats|seats_(\d+))", constraint, re.IGNORECASE)
    if m:
        threshold = int(m.group(1) or m.group(2))
        actual = deal_dict.get("seats") or 0
        return actual >= threshold

    # Branch 3: "minimum_12_month_contract", "minimum_commitment_12_months"
    # → check contract_length_months
    pattern = r"(?:minimum_)?(?:commitment_)?(\d+)_(?:month|year)|minimum_commitment_(\d+)"
    m = re.search(pattern, constraint, re.IGNORECASE)
    if m:
        months = int(m.group(1) or m.group(2))
        if "year" in constraint.lower():
            months *= 12
        actual = deal_dict.get("contract_length_months") or 0
        return actual >= months

    # Semantic constraints are not verifiable from deal fields alone.
    # Assume satisfied — violations caught via transcript-level checks.
    return True
