from __future__ import annotations

import re
from typing import Any


def build_issue_utility_models(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = str(document.get("domain", ""))
    if domain == "ecommerce":
        return _build_ecommerce_models()
    if domain == "saas_procurement":
        return _build_saas_models(document)
    if domain == "settlement":
        return _build_settlement_models(document)
    if domain == "ethical_business":
        return _build_ethical_business_models(document)
    return {"price_weight": 1.0, "issue_utilities": {}}, {
        "price_weight": 1.0,
        "issue_utilities": {},
    }


def _build_ecommerce_models() -> tuple[dict[str, Any], dict[str, Any]]:
    buyer = {
        "price_weight": 0.7,
        "issue_utilities": {
            "shipping_terms": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "seller_pays_shipping": 1.0,
                    "expedited": 0.9,
                    "standard": 0.7,
                    "FOB_destination": 0.8,
                    "buyer_pays_shipping": 0.3,
                    "FOB_origin": 0.2,
                },
            },
            "return_policy": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "60-day": 1.0,
                    "45-day": 0.8,
                    "30-day": 0.6,
                    "14-day": 0.2,
                    "final_sale": 0.0,
                },
            },
        },
    }
    seller = {
        "price_weight": 0.7,
        "issue_utilities": {
            "shipping_terms": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "buyer_pays_shipping": 1.0,
                    "FOB_origin": 0.9,
                    "standard": 0.8,
                    "FOB_destination": 0.5,
                    "seller_pays_shipping": 0.2,
                    "expedited": 0.2,
                },
            },
            "return_policy": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "14-day": 1.0,
                    "30-day": 0.7,
                    "45-day": 0.4,
                    "60-day": 0.2,
                    "final_sale": 1.0,
                },
            },
        },
    }
    return buyer, seller


def _build_saas_models(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    buyer_private = document.get("buyer_context", {}).get("private_info", [])
    seller_private = document.get("seller_context", {}).get("private_info", [])
    buyer_hours = _extract_hours(buyer_private) or 40
    seller_hours = _extract_hours(seller_private) or buyer_hours

    buyer = {
        "price_weight": 0.6,
        "issue_utilities": {
            "contract_length_months": {
                "type": "numeric",
                "weight": 0.2,
                "best": 12,
                "worst": 36,
            },
            "onboarding_support_hours": {
                "type": "numeric",
                "weight": 0.15,
                "best": buyer_hours,
                "worst": 0,
            },
            "sla_tier": {
                "type": "categorical",
                "weight": 0.05,
                "scores": {
                    "standard": 0.4,
                    "premium": 0.7,
                    "enterprise": 1.0,
                },
            },
        },
    }
    seller = {
        "price_weight": 0.6,
        "issue_utilities": {
            "contract_length_months": {
                "type": "numeric",
                "weight": 0.2,
                "best": 36,
                "worst": 12,
            },
            "onboarding_support_hours": {
                "type": "numeric",
                "weight": 0.15,
                "best": 0,
                "worst": seller_hours,
            },
            "sla_tier": {
                "type": "categorical",
                "weight": 0.05,
                "scores": {
                    "standard": 1.0,
                    "premium": 0.7,
                    "enterprise": 0.4,
                },
            },
        },
    }
    return buyer, seller


def _build_settlement_models(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    buyer_constraints = document.get("buyer_context", {}).get("hard_constraints", [])
    seller_constraints = document.get("seller_context", {}).get("hard_constraints", [])
    buyer = {
        "price_weight": 0.55,
        "issue_utilities": {
            "payment_structure": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "lump_sum": 1.0,
                    "structured": 0.5,
                    "installments": 0.4,
                },
            },
            "payment_timeline_days": {
                "type": "numeric",
                "weight": 0.1,
                "best": 30,
                "worst": 365,
            },
            "confidentiality_clause": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {
                    "true": 1.0 if "confidentiality_clause_required" in buyer_constraints else 0.8,
                    "false": 0.0 if "confidentiality_clause_required" in buyer_constraints else 0.4,
                },
            },
            "non_disparagement": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {"true": 0.8, "false": 0.5},
            },
        },
    }
    seller = {
        "price_weight": 0.55,
        "issue_utilities": {
            "payment_structure": {
                "type": "categorical",
                "weight": 0.15,
                "scores": {
                    "structured": 1.0,
                    "installments": 0.8,
                    "lump_sum": 0.4,
                },
            },
            "payment_timeline_days": {
                "type": "numeric",
                "weight": 0.1,
                "best": 365,
                "worst": 30,
            },
            "confidentiality_clause": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {
                    "true": 1.0 if "confidentiality_clause_required" in seller_constraints else 0.8,
                    "false": 0.0 if "confidentiality_clause_required" in seller_constraints else 0.4,
                },
            },
            "non_disparagement": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {"true": 1.0, "false": 0.4},
            },
        },
    }
    return buyer, seller


def _build_ethical_business_models(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    buyer_private = document.get("buyer_context", {}).get("private_info", [])
    seller_private = document.get("seller_context", {}).get("private_info", [])
    buyer_target = _extract_transition_target(buyer_private, default=12)
    seller_target = _extract_transition_target(seller_private, default=18)

    buyer = {
        "price_weight": 0.5,
        "issue_utilities": {
            "audit_frequency_months": {
                "type": "numeric",
                "weight": 0.1,
                "best": 3,
                "worst": 12,
            },
            "transition_period_months": {
                "type": "target_numeric",
                "weight": 0.1,
                "target": buyer_target,
                "worst_distance": 18,
            },
            "environmental_commitments": {
                "type": "set",
                "weight": 0.1,
                "scores": {
                    "sea_freight": 0.4,
                    "regenerative_certification": 0.35,
                    "emissions_reporting": 0.25,
                },
            },
            "labor_standards": {
                "type": "set",
                "weight": 0.1,
                "scores": {
                    "no_child_labor": 0.4,
                    "living_wage": 0.3,
                    "independent_audits": 0.3,
                },
            },
            "transparency_reports": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {"true": 1.0, "false": 0.0},
            },
        },
    }
    seller = {
        "price_weight": 0.5,
        "issue_utilities": {
            "audit_frequency_months": {
                "type": "numeric",
                "weight": 0.1,
                "best": 12,
                "worst": 3,
            },
            "transition_period_months": {
                "type": "target_numeric",
                "weight": 0.1,
                "target": seller_target,
                "worst_distance": 18,
            },
            "environmental_commitments": {
                "type": "set",
                "weight": 0.1,
                "scores": {
                    "regenerative_certification": 0.3,
                    "emissions_reporting": 0.2,
                    "technical_assistance": 0.5,
                },
            },
            "labor_standards": {
                "type": "set",
                "weight": 0.1,
                "scores": {
                    "no_child_labor": 0.2,
                    "living_wage": 0.3,
                    "independent_audits": 0.5,
                },
            },
            "transparency_reports": {
                "type": "boolean",
                "weight": 0.1,
                "scores": {"true": 0.4, "false": 1.0},
            },
        },
    }
    return buyer, seller


def _extract_hours(private_info: list[str]) -> int | None:
    for item in private_info:
        match = re.search(r"(\d+)_hours", str(item))
        if match:
            return int(match.group(1))
    return None


def _extract_transition_target(private_info: list[str], default: int) -> int:
    for item in private_info:
        match = re.search(r"(\d+)_month", str(item))
        if match and "transition" in str(item):
            return int(match.group(1))
    return default
