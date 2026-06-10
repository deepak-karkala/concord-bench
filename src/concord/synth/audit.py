import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from concord.schemas.scenario import Domain


def append_audit_log(
    log_path: str | Path,
    scenario_id: str,
    original_culture: str,
    target_culture: str,
    adapted_fields: list[str],
    auditor_comments: str = "",
) -> None:
    entry = {
        "scenario_id": scenario_id,
        "original_culture": original_culture,
        "target_culture": target_culture,
        "adapted_fields": adapted_fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auditor_comments": auditor_comments,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


_EXPECTED_DOMAIN_FIELDS: dict[str, set[str]] = {
    Domain.ECOMMERCE.value: {
        "price",
        "quantity",
        "delivery_days",
        "payment_terms_days",
        "shipping_terms",
        "return_policy",
    },
    Domain.SAAS_PROCUREMENT.value: {
        "monthly_price",
        "seats",
        "contract_length_months",
        "onboarding_support_hours",
        "sla_tier",
    },
    Domain.SETTLEMENT.value: {
        "settlement_amount",
        "payment_structure",
        "payment_timeline_days",
        "confidentiality_clause",
        "non_disparagement",
    },
    Domain.ETHICAL_BUSINESS.value: {
        "price",
        "audit_frequency_months",
        "transition_period_months",
        "environmental_commitments",
        "labor_standards",
        "transparency_reports",
    },
}

_TEMPLATE_MARKERS = ("todo", "tbd", "placeholder", "lorem", "generic", "example")
_MEASURABLE_CONSTRAINT_PREFIXES = (
    "minimum_",
    "maximum_",
    "quality_",
    "warranty_",
    "data_",
    "soc2_",
    "confidentiality_",
    "no_",
    "annual_",
    "monthly_",
    "audit_",
    "transparency_",
    "living_",
    "environmental_",
    "labor_",
    "cannot_",
    "deal_must_",
)


def audit_generated_scenarios(scenarios_dir: str | Path) -> dict:
    base = Path(scenarios_dir)
    documents: list[dict] = []
    duplicate_groups: dict[str, list[str]] = {}
    invalid_batna_math_ids: list[str] = []
    domain_schema_mismatch_ids: list[str] = []
    unmeasurable_constraint_ids: list[str] = []
    private_info_realism_ids: list[str] = []
    template_artifact_ids: list[str] = []
    zopa_status_counts = {"wide": 0, "narrow_or_moderate": 0, "none": 0, "unknown": 0}

    for path in sorted(base.rglob("*.yaml")):
        with path.open() as f:
            document = yaml.safe_load(f)
        if not isinstance(document, dict):
            continue
        documents.append(document)
        scenario_id = str(document.get("id", path.stem))
        description_key = _normalize_text(document.get("scenario_description", ""))
        duplicate_groups.setdefault(description_key, []).append(scenario_id)

        if not _has_valid_batna_math(document):
            invalid_batna_math_ids.append(scenario_id)

        if not _has_expected_domain_schema(document):
            domain_schema_mismatch_ids.append(scenario_id)

        if not _has_measurable_constraints(document):
            unmeasurable_constraint_ids.append(scenario_id)

        if not _has_realistic_private_info(document):
            private_info_realism_ids.append(scenario_id)

        if _has_template_artifacts(document):
            template_artifact_ids.append(scenario_id)

        zopa_status_counts[_classify_zopa(document)] += 1

    duplicate_description_groups = sorted(
        [sorted(ids) for ids in duplicate_groups.values() if len(ids) > 1]
    )

    return {
        "scenarios_dir": str(base),
        "total_scenarios": len(documents),
        "invalid_batna_math_ids": sorted(invalid_batna_math_ids),
        "domain_schema_mismatch_ids": sorted(domain_schema_mismatch_ids),
        "unmeasurable_constraint_ids": sorted(unmeasurable_constraint_ids),
        "private_info_realism_ids": sorted(private_info_realism_ids),
        "template_artifact_ids": sorted(template_artifact_ids),
        "duplicate_description_groups": duplicate_description_groups,
        "zopa_status_counts": zopa_status_counts,
        "findings": {
            "blocking_failures_present": any(
                [
                    invalid_batna_math_ids,
                    domain_schema_mismatch_ids,
                    unmeasurable_constraint_ids,
                    private_info_realism_ids,
                    template_artifact_ids,
                    duplicate_description_groups,
                ]
            )
        },
    }


def _has_valid_batna_math(document: dict) -> bool:
    buyer = document.get("buyer_context", {})
    seller = document.get("seller_context", {})
    buyer_batna = buyer.get("batna")
    buyer_reserve = buyer.get("reserve_price")
    seller_batna = seller.get("batna")
    seller_reserve = seller.get("reserve_price")
    if None in (buyer_batna, buyer_reserve, seller_batna, seller_reserve):
        return False
    try:
        return float(buyer_reserve) > float(buyer_batna) and float(seller_reserve) < float(
            seller_batna
        )
    except (TypeError, ValueError):
        return False


def _classify_zopa(document: dict) -> str:
    buyer = document.get("buyer_context", {})
    seller = document.get("seller_context", {})
    buyer_batna = buyer.get("batna")
    buyer_reserve = buyer.get("reserve_price")
    seller_reserve = seller.get("reserve_price")
    if None in (buyer_batna, buyer_reserve, seller_reserve):
        return "unknown"
    try:
        buyer_range = float(buyer_reserve) - float(buyer_batna)
        zopa = float(buyer_reserve) - float(seller_reserve)
    except (TypeError, ValueError):
        return "unknown"
    if zopa < 0:
        return "none"
    if buyer_range <= 0:
        return "unknown"
    if zopa / buyer_range > 0.4:
        return "wide"
    return "narrow_or_moderate"


def _has_expected_domain_schema(document: dict) -> bool:
    domain = str(document.get("domain", ""))
    expected = _EXPECTED_DOMAIN_FIELDS.get(domain)
    if not expected:
        return False
    schema = set(document.get("deal_schema", {}).keys())
    return expected.issubset(schema)


def _has_measurable_constraints(document: dict) -> bool:
    if not _has_expected_domain_schema(document):
        return False
    constraints = list(document.get("buyer_context", {}).get("hard_constraints", [])) + list(
        document.get("seller_context", {}).get("hard_constraints", [])
    )
    if not constraints:
        return False
    for constraint in constraints:
        if not any(str(constraint).startswith(prefix) for prefix in _MEASURABLE_CONSTRAINT_PREFIXES):
            return False
    return True


def _has_realistic_private_info(document: dict) -> bool:
    all_private = list(document.get("buyer_context", {}).get("private_info", [])) + list(
        document.get("seller_context", {}).get("private_info", [])
    )
    if not all_private:
        return False
    all_constraints = {
        str(item)
        for item in document.get("buyer_context", {}).get("hard_constraints", [])
    } | {
        str(item) for item in document.get("seller_context", {}).get("hard_constraints", [])
    }
    for item in all_private:
        normalized = _normalize_text(item)
        if not normalized or normalized in _TEMPLATE_MARKERS:
            return False
        if normalized in all_constraints:
            return False
        if len(normalized) < 8:
            return False
    return True


def _has_template_artifacts(document: dict) -> bool:
    haystacks = [
        str(document.get("scenario_description", "")),
        *[str(item) for item in document.get("buyer_context", {}).get("private_info", [])],
        *[str(item) for item in document.get("seller_context", {}).get("private_info", [])],
    ]
    for text in haystacks:
        normalized = _normalize_text(text)
        if any(marker in normalized for marker in _TEMPLATE_MARKERS):
            return True
    return False


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return normalized
