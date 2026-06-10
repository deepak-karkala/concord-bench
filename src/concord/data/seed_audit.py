from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEED_DIR = Path(__file__).parent / "seed_yamls"
MULTI_ISSUE_FIELDS = {
    "delivery_days",
    "payment_terms_days",
    "contract_length_months",
    "onboarding_support_hours",
    "payment_structure",
    "payment_timeline_days",
    "audit_frequency_months",
    "transition_period_months",
}


def _load_seed_documents(seed_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(seed_dir.rglob("*.yaml")):
        with path.open() as f:
            documents.append(yaml.safe_load(f))
    return documents


def _is_no_zopa(document: dict[str, Any]) -> bool:
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    buyer_reserve = buyer_context.get("reserve_price")
    seller_reserve = seller_context.get("reserve_price")
    if buyer_reserve is None or seller_reserve is None:
        return False
    return float(buyer_reserve) < float(seller_reserve)


def _is_multi_issue(document: dict[str, Any]) -> bool:
    deal_schema = document.get("deal_schema", {})
    return bool(MULTI_ISSUE_FIELDS & set(deal_schema.keys()))


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


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


def _classify_zopa_status(document: dict[str, Any]) -> str:
    if _is_no_zopa(document):
        return "none"
    ratio = _buyer_zopa_ratio(document)
    if ratio is not None and ratio > 0.4:
        return "wide"
    return "narrow_or_moderate"


def _classify_issue_structure(document: dict[str, Any]) -> str:
    return "multi_issue" if _is_multi_issue(document) else "single_issue"


def _classify_pressure_type(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {})
    if metadata.get("bright_line"):
        return "bright_line"
    pressure_type = str(metadata.get("pressure_type", "")).strip()
    return pressure_type or "none"


def _classify_constraint_types(document: dict[str, Any]) -> list[str]:
    categories: set[str] = set()
    metadata = document.get("metadata", {})
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    if buyer_context.get("hard_constraints") or seller_context.get("hard_constraints"):
        categories.add("hard_constraints")
    if buyer_context.get("walk_away_threshold") is not None or seller_context.get("walk_away_threshold") is not None:
        categories.add("walkaway")
    if document.get("forbidden_claims") or metadata.get("bright_line"):
        categories.add("bright_line")
    if buyer_context.get("private_info") or seller_context.get("private_info"):
        categories.add("privacy")
    return sorted(categories) or ["none"]


def _is_structurally_valid_no_zopa(document: dict[str, Any]) -> bool:
    if not _is_no_zopa(document):
        return False
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    buyer_batna = buyer_context.get("batna")
    buyer_reserve = buyer_context.get("reserve_price")
    seller_batna = seller_context.get("batna")
    seller_reserve = seller_context.get("reserve_price")
    if None in (buyer_batna, buyer_reserve, seller_batna, seller_reserve):
        return False
    return (
        float(buyer_reserve) < float(seller_reserve)
        and float(buyer_reserve) != float(buyer_batna)
        and float(seller_reserve) != float(seller_batna)
    )


def _is_walkaway_positive_control(document: dict[str, Any]) -> bool:
    if _is_no_zopa(document):
        return False
    buyer_context = document.get("buyer_context", {})
    seller_context = document.get("seller_context", {})
    return (
        buyer_context.get("walk_away_threshold") is not None
        and seller_context.get("walk_away_threshold") is not None
    )


def _is_t0_restatement_risk(document: dict[str, Any]) -> bool:
    metadata = document.get("metadata", {})
    if metadata.get("difficulty_tier") != 0:
        return False
    description = str(document.get("scenario_description", "")).lower()
    risky_phrases = (
        "both sides want to close quickly",
        "fair price on a familiar order",
        "straightforward reorder",
        "quick close expected",
    )
    return any(phrase in description for phrase in risky_phrases)


def audit_seed_corpus(seed_dir: Path | None = None) -> dict[str, Any]:
    if seed_dir is None:
        seed_dir = DEFAULT_SEED_DIR

    documents = _load_seed_documents(seed_dir)
    domain_counts: Counter[str] = Counter()
    effective_tier_counts: Counter[str] = Counter()
    explicit_tier_counts: Counter[str] = Counter()
    pressure_type_counts: Counter[str] = Counter()
    by_domain_and_effective_tier: dict[str, Counter[str]] = defaultdict(Counter)
    no_zopa_count = 0
    multi_issue_count = 0
    missing_difficulty_tier_count = 0

    for document in documents:
        domain = str(document.get("domain", "unknown"))
        metadata = document.get("metadata", {})
        raw_tier = metadata.get("difficulty_tier")
        effective_tier = "1" if raw_tier is None else str(raw_tier)
        pressure_type = str(metadata.get("pressure_type", "missing"))

        domain_counts[domain] += 1
        effective_tier_counts[effective_tier] += 1
        by_domain_and_effective_tier[domain][effective_tier] += 1
        pressure_type_counts[pressure_type] += 1

        if raw_tier is None:
            missing_difficulty_tier_count += 1
        else:
            explicit_tier_counts[str(raw_tier)] += 1

        if _is_no_zopa(document):
            no_zopa_count += 1
        if _is_multi_issue(document):
            multi_issue_count += 1

    likely_cause = "All seeds have explicit difficulty_tier metadata."
    if missing_difficulty_tier_count:
        likely_cause = (
            "Some seeds are missing difficulty_tier metadata; untagged seeds currently default to T1."
        )

    findings = {
        "t0_absent": effective_tier_counts.get("0", 0) == 0,
        "t2_absent": effective_tier_counts.get("2", 0) == 0,
        "likely_cause": likely_cause,
    }

    return {
        "seed_dir": str(seed_dir),
        "total_seeds": len(documents),
        "domain_counts": _sorted_counter(domain_counts),
        "effective_tier_counts": _sorted_counter(effective_tier_counts),
        "explicit_tier_counts": _sorted_counter(explicit_tier_counts),
        "missing_difficulty_tier_count": missing_difficulty_tier_count,
        "pressure_type_counts": _sorted_counter(pressure_type_counts),
        "no_zopa_count": no_zopa_count,
        "multi_issue_count": multi_issue_count,
        "by_domain_and_effective_tier": {
            domain: _sorted_counter(tiers)
            for domain, tiers in sorted(by_domain_and_effective_tier.items())
        },
        "findings": findings,
    }


def audit_no_zopa_slice(seed_dir: Path | None = None) -> dict[str, Any]:
    if seed_dir is None:
        seed_dir = DEFAULT_SEED_DIR

    documents = _load_seed_documents(seed_dir)
    no_zopa_by_domain: Counter[str] = Counter()
    positive_control_by_domain: Counter[str] = Counter()
    invalid_no_zopa_ids: list[str] = []
    no_zopa_count = 0
    structurally_valid_no_zopa_count = 0
    positive_control_count = 0

    for document in documents:
        domain = str(document.get("domain", "unknown"))
        scenario_id = str(document.get("id", "unknown"))

        if _is_no_zopa(document):
            no_zopa_count += 1
            no_zopa_by_domain[domain] += 1
            if _is_structurally_valid_no_zopa(document):
                structurally_valid_no_zopa_count += 1
            else:
                invalid_no_zopa_ids.append(scenario_id)

        if _is_walkaway_positive_control(document):
            positive_control_count += 1
            positive_control_by_domain[domain] += 1

    minimum_domain_coverage_met = all(
        no_zopa_by_domain.get(domain, 0) >= 5
        for domain in ("ecommerce", "ethical_business", "saas_procurement", "settlement")
    )

    return {
        "seed_dir": str(seed_dir),
        "no_zopa_count": no_zopa_count,
        "structurally_valid_no_zopa_count": structurally_valid_no_zopa_count,
        "invalid_no_zopa_ids": sorted(invalid_no_zopa_ids),
        "no_zopa_by_domain": _sorted_counter(no_zopa_by_domain),
        "positive_control_count": positive_control_count,
        "positive_control_by_domain": _sorted_counter(positive_control_by_domain),
        "findings": {
            "target_range_met": 40 <= no_zopa_count <= 60,
            "minimum_domain_coverage_met": minimum_domain_coverage_met,
            "all_no_zopa_structurally_valid": (
                structurally_valid_no_zopa_count == no_zopa_count
            ),
        },
    }


def audit_t0_slice(seed_dir: Path | None = None) -> dict[str, Any]:
    if seed_dir is None:
        seed_dir = DEFAULT_SEED_DIR

    documents = _load_seed_documents(seed_dir)
    t0_by_domain: Counter[str] = Counter()
    restatement_risk_ids: list[str] = []

    for document in documents:
        if document.get("metadata", {}).get("difficulty_tier") != 0:
            continue
        domain = str(document.get("domain", "unknown"))
        scenario_id = str(document.get("id", "unknown"))
        t0_by_domain[domain] += 1
        if _is_t0_restatement_risk(document):
            restatement_risk_ids.append(scenario_id)

    active_domains = sum(1 for count in t0_by_domain.values() if count > 0)
    return {
        "seed_dir": str(seed_dir),
        "t0_count": sum(t0_by_domain.values()),
        "t0_by_domain": _sorted_counter(t0_by_domain),
        "restatement_risk_ids": sorted(restatement_risk_ids),
        "findings": {
            "multi_domain_t0_met": active_domains >= 4,
            "all_t0s_low_restatement_risk": not restatement_risk_ids,
        },
    }


def audit_tier_semantics(seed_dir: Path | None = None) -> dict[str, Any]:
    if seed_dir is None:
        seed_dir = DEFAULT_SEED_DIR

    documents = _load_seed_documents(seed_dir)
    zopa_status_counts: Counter[str] = Counter()
    issue_structure_counts: Counter[str] = Counter()
    pressure_type_counts: Counter[str] = Counter()
    constraint_type_counts: Counter[str] = Counter()
    culture_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    tier_profiles: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "zopa_status": Counter(),
            "issue_structure": Counter(),
            "pressure_type": Counter(),
            "constraint_type": Counter(),
            "culture": Counter(),
        }
    )

    for document in documents:
        metadata = document.get("metadata", {})
        tier = str(metadata.get("difficulty_tier", "unknown"))
        zopa_status = _classify_zopa_status(document)
        issue_structure = _classify_issue_structure(document)
        pressure_type = _classify_pressure_type(document)
        culture = str(document.get("culture", "unknown"))
        constraint_types = _classify_constraint_types(document)

        tier_counts[tier] += 1
        zopa_status_counts[zopa_status] += 1
        issue_structure_counts[issue_structure] += 1
        pressure_type_counts[pressure_type] += 1
        culture_counts[culture] += 1
        tier_profiles[tier]["zopa_status"][zopa_status] += 1
        tier_profiles[tier]["issue_structure"][issue_structure] += 1
        tier_profiles[tier]["pressure_type"][pressure_type] += 1
        tier_profiles[tier]["culture"][culture] += 1

        for constraint_type in constraint_types:
            constraint_type_counts[constraint_type] += 1
            tier_profiles[tier]["constraint_type"][constraint_type] += 1

    return {
        "seed_dir": str(seed_dir),
        "factor_dimensions": {
            "difficulty_label_counts": _sorted_counter(tier_counts),
            "zopa_status_counts": _sorted_counter(zopa_status_counts),
            "issue_structure_counts": _sorted_counter(issue_structure_counts),
            "pressure_type_counts": _sorted_counter(pressure_type_counts),
            "constraint_type_counts": _sorted_counter(constraint_type_counts),
            "culture_counts": _sorted_counter(culture_counts),
            "seller_policy_dimension": {
                "status": "runtime_controlled",
                "available_policies": [
                    "honest",
                    "honest_cooperative",
                    "honest_hardball",
                    "deceptive_or_pressure",
                ],
            },
        },
        "tier_profiles": {
            tier: {
                dimension: _sorted_counter(counter)
                for dimension, counter in sorted(profile.items())
            }
            for tier, profile in sorted(tier_profiles.items())
        },
        "findings": {
            "tiers_are_provisional_operational_labels": True,
            "empirical_calibration_required_after_scale": True,
            "difficulty_should_not_be_inferred_from_tier_label_alone": True,
        },
    }


def audit_multi_issue_utility_models(seed_dir: Path | None = None) -> dict[str, Any]:
    if seed_dir is None:
        seed_dir = DEFAULT_SEED_DIR

    documents = _load_seed_documents(seed_dir)
    multi_issue_count = 0
    with_explicit_models = 0
    missing_issue_utility_ids: list[str] = []
    by_domain: Counter[str] = Counter()

    for document in documents:
        if not _is_multi_issue(document):
            continue
        multi_issue_count += 1
        domain = str(document.get("domain", "unknown"))
        scenario_id = str(document.get("id", "unknown"))
        by_domain[domain] += 1
        buyer_model = document.get("buyer_context", {}).get("issue_utilities", {})
        seller_model = document.get("seller_context", {}).get("issue_utilities", {})
        if buyer_model and seller_model:
            with_explicit_models += 1
        else:
            missing_issue_utility_ids.append(scenario_id)

    return {
        "seed_dir": str(seed_dir),
        "multi_issue_count": multi_issue_count,
        "with_explicit_issue_utility_models": with_explicit_models,
        "missing_issue_utility_ids": sorted(missing_issue_utility_ids),
        "multi_issue_by_domain": _sorted_counter(by_domain),
        "findings": {
            "all_multi_issue_seeds_have_explicit_models": (
                multi_issue_count == with_explicit_models
            ),
        },
    }
