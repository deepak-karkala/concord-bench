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
