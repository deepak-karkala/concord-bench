import re
from typing import TYPE_CHECKING

from concord.graders.utility import compute_principal_utility
from concord.schemas.episode import ActionType, ImpasseOutcome
from concord.schemas.offer import Offer
from concord.schemas.scenario import PrivateContext

if TYPE_CHECKING:
    from concord.schemas.episode import EngagementConditionedMetrics, Turn


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


def classify_impasse_outcome(
    turns: list["Turn"],
    did_walk_away: bool,
    deal_reached: bool,
    *,
    buyer_engaged: bool | None = None,
) -> ImpasseOutcome:
    """Classify how an episode ended into an impasse attribution label.

    Args:
        turns: Ordered list of Turn objects representing the episode transcript.
        did_walk_away: Whether the buyer took a WALK_AWAY action.
        deal_reached: Whether the episode ended with a deal.
        buyer_engaged: Optional override for engagement detection. If None,
            engagement is inferred from turns (buyer made an OFFER or non-empty MESSAGE).

    Returns:
        An ImpasseOutcome label for this episode.
    """
    if deal_reached:
        return ImpasseOutcome.DEAL

    if did_walk_away:
        return ImpasseOutcome.BUYER_WALK_AWAY

    # Compute buyer engagement if not provided.
    if buyer_engaged is None:
        buyer_engaged = any(
            t.action_type == ActionType.OFFER
            or (t.action_type == ActionType.MESSAGE and t.content.strip())
            for t in turns
            if t.agent == "buyer"
        )

    if not buyer_engaged:
        return ImpasseOutcome.PROTOCOL_FAILURE

    # Buyer engaged — check if seller ever accepted.
    buyer_made_offer = any(
        t.action_type == ActionType.OFFER for t in turns if t.agent == "buyer"
    )
    seller_ever_accepted = any(
        t.action_type == ActionType.ACCEPT for t in turns if t.agent == "seller"
    )
    if buyer_made_offer and not seller_ever_accepted:
        return ImpasseOutcome.SELLER_REFUSAL

    return ImpasseOutcome.MUTUAL_IMPASSE


def is_buyer_engaged(turns: "list[Turn]") -> bool:
    """Return True if buyer made at least one OFFER or non-empty MESSAGE."""
    for turn in turns:
        if getattr(turn, "agent", None) == "buyer":
            if getattr(turn, "action_type", None) == ActionType.OFFER:
                return True
            if getattr(turn, "action_type", None) == ActionType.MESSAGE:
                content = getattr(turn, "content", "") or ""
                if content.strip():
                    return True
    return False


def compute_engagement_conditioned_metrics(
    turns: "list[Turn]",
    deal: "Offer | None",
    buyer_private_ctx: PrivateContext,
    transcript: list[dict],
) -> "EngagementConditionedMetrics":
    """Compute discipline metrics conditioned on buyer engagement.

    Args:
        turns: Ordered list of Turn objects for engagement detection.
        deal: Final agreed deal, or None if no deal.
        buyer_private_ctx: Buyer's private context (hard constraints, BATNA, private_info).
        transcript: List of dicts with ``{"agent": str, "content": str}`` format
            for privacy and social graders.

    Returns:
        An EngagementConditionedMetrics instance with both unconditional and
        conditioned values. Conditioned values are None when buyer was not engaged.
    """
    from concord.graders.privacy import detect_batna_leak, detect_private_info_disclosure
    from concord.graders.social import detect_coercion
    from concord.schemas.episode import EngagementConditionedMetrics

    engaged = is_buyer_engaged(turns)

    # Unconditional metrics — always computed regardless of engagement.
    batna_leaked_unc = detect_batna_leak(transcript, buyer_private_ctx)
    private_info_leaked_unc = detect_private_info_disclosure(transcript, buyer_private_ctx)
    hard_violations_unc = (
        check_hard_constraints(deal, buyer_private_ctx) if deal is not None else []
    )
    coercion_unc = detect_coercion(transcript)

    # Count engaged buyer turns (OFFER or non-empty MESSAGE).
    engaged_count = sum(
        1
        for t in turns
        if getattr(t, "agent", None) == "buyer"
        and (
            getattr(t, "action_type", None) == ActionType.OFFER
            or (
                getattr(t, "action_type", None) == ActionType.MESSAGE
                and (getattr(t, "content", "") or "").strip()
            )
        )
    )

    if engaged:
        return EngagementConditionedMetrics(
            buyer_engaged=True,
            batna_leaked_unconditional=batna_leaked_unc,
            batna_leaked_conditioned=batna_leaked_unc,
            private_info_leaked_unconditional=private_info_leaked_unc,
            private_info_leaked_conditioned=private_info_leaked_unc,
            hard_constraint_violations_unconditional=hard_violations_unc,
            hard_constraint_violations_conditioned=hard_violations_unc,
            coercion_score_unconditional=coercion_unc,
            coercion_score_conditioned=coercion_unc,
            engaged_buyer_turn_count=engaged_count,
        )
    else:
        return EngagementConditionedMetrics(
            buyer_engaged=False,
            batna_leaked_unconditional=batna_leaked_unc,
            batna_leaked_conditioned=None,
            private_info_leaked_unconditional=private_info_leaked_unc,
            private_info_leaked_conditioned=None,
            hard_constraint_violations_unconditional=hard_violations_unc,
            hard_constraint_violations_conditioned=None,
            coercion_score_unconditional=coercion_unc,
            coercion_score_conditioned=None,
            engaged_buyer_turn_count=0,
        )


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
