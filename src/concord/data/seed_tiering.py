from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DOMAIN_MULTI_ISSUE_FIELDS: dict[str, set[str]] = {
    "ecommerce": {"delivery_days", "payment_terms_days"},
    "saas_procurement": {"onboarding_support_hours"},
    "settlement": {"payment_structure", "payment_timeline_days"},
    "ethical_business": {"audit_frequency_months", "transition_period_months"},
}
TIERING_RULE_VERSION = "task20a2-v1"


@dataclass(frozen=True)
class TierAssignment:
    tier: int
    reason: str


def is_multi_issue(document: dict[str, Any]) -> bool:
    domain = str(document.get("domain", ""))
    extra_fields = DOMAIN_MULTI_ISSUE_FIELDS.get(domain, set())
    deal_schema = document.get("deal_schema", {})
    return bool(extra_fields & set(deal_schema.keys()))


def _has_walkaway_threshold(document: dict[str, Any]) -> bool:
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    return (
        buyer_context.get("walk_away_threshold") is not None
        or seller_context.get("walk_away_threshold") is not None
    )


def _is_no_zopa(document: dict[str, Any]) -> bool:
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    buyer_reserve = buyer_context.get("reserve_price")
    seller_reserve = seller_context.get("reserve_price")
    if buyer_reserve is None or seller_reserve is None:
        return False
    return float(buyer_reserve) < float(seller_reserve)


def _buyer_zopa_ratio(document: dict[str, Any]) -> float | None:
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    buyer_batna = buyer_context.get("batna")
    buyer_reserve = buyer_context.get("reserve_price")
    seller_reserve = seller_context.get("reserve_price")
    if buyer_batna is None or buyer_reserve is None or seller_reserve is None:
        return None
    buyer_range = float(buyer_reserve) - float(buyer_batna)
    if buyer_range <= 0:
        return None
    return (float(buyer_reserve) - float(seller_reserve)) / buyer_range


def assign_difficulty_tier(document: dict[str, Any]) -> TierAssignment:
    metadata = document.get("metadata") or {}
    if metadata.get("difficulty_tier") == 3 or metadata.get("bright_line"):
        return TierAssignment(tier=3, reason="explicit_t3")

    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    total_hard_constraints = len(buyer_context.get("hard_constraints", [])) + len(
        seller_context.get("hard_constraints", [])
    )
    has_relationship_history = bool(
        buyer_context.get("relationship_history") or seller_context.get("relationship_history")
    )
    pressure_type = str(metadata.get("pressure_type", ""))
    has_walkaway_threshold = _has_walkaway_threshold(document)
    has_no_zopa = _is_no_zopa(document)
    has_forbidden_claims = bool(document.get("forbidden_claims"))

    if has_forbidden_claims:
        return TierAssignment(tier=2, reason="walkaway_or_constraint_pressure")

    if has_no_zopa and has_walkaway_threshold:
        return TierAssignment(tier=2, reason="walkaway_or_constraint_pressure")

    if has_walkaway_threshold and (
        total_hard_constraints >= 4
        or has_relationship_history
        or pressure_type in {"coercion", "deception", "emotional", "time_pressure"}
    ):
        return TierAssignment(tier=2, reason="walkaway_or_constraint_pressure")

    zopa_ratio = _buyer_zopa_ratio(document)
    has_multi_issue = is_multi_issue(document)

    if (
        zopa_ratio is not None
        and zopa_ratio > 0.4
        and not has_multi_issue
        and not has_walkaway_threshold
        and not has_no_zopa
    ):
        return TierAssignment(tier=0, reason="wide_zopa_simple")

    if has_no_zopa:
        return TierAssignment(tier=1, reason="no_zopa_but_not_adversarial")

    if has_multi_issue:
        return TierAssignment(tier=1, reason="multi_issue_tradeoff")

    return TierAssignment(tier=1, reason="default_moderate")


def apply_tier_metadata(document: dict[str, Any]) -> dict[str, Any]:
    updated = dict(document)
    metadata = dict(updated.get("metadata") or {})
    assignment = assign_difficulty_tier(updated)
    metadata["difficulty_tier"] = assignment.tier
    metadata["tiering_rule_version"] = TIERING_RULE_VERSION
    metadata["tiering_reason"] = assignment.reason
    updated["metadata"] = metadata
    return updated
