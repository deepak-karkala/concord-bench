from concord.exceptions import ConcordError
from concord.schemas.scenario import Domain, PrivateContext, Scenario


class EnrichmentError(ConcordError):
    pass


_DOMAIN_SCHEMAS: dict[Domain, dict] = {
    Domain.ECOMMERCE: {"price": "float", "quantity": "int", "shipping_terms": "str", "return_policy": "str"},
    Domain.SAAS_PROCUREMENT: {"monthly_price": "float", "seats": "int", "contract_length_months": "int", "sla_tier": "str"},
    Domain.SETTLEMENT: {"settlement_amount": "float", "payment_terms": "str", "confidentiality_clause": "bool", "non_disparagement": "bool"},
    Domain.ETHICAL_BUSINESS: {"price": "float", "environmental_commitments": "list", "labor_standards": "list", "transparency_reports": "bool"},
}


def _require_field(data: dict, field: str, context: str) -> str:
    value = data.get(field)
    if value is None:
        raise EnrichmentError(f"Missing required field '{field}' in {context}")
    if not isinstance(value, str) or not value.strip():
        raise EnrichmentError(f"Empty required field '{field}' in {context}")
    return value


def _to_domain(domain_str: str) -> Domain:
    try:
        return Domain(domain_str)
    except ValueError:
        raise EnrichmentError(f"Unknown domain: {domain_str}")


def enrich_awm_scenario(awm_scenario: dict, domain: str, culture: str = "US") -> Scenario:
    domain_enum = _to_domain(domain)

    scenario_id = _require_field(awm_scenario, "scenario_id", "awm_scenario")
    name = awm_scenario.get("name", "")
    description = awm_scenario.get("description", "")
    feature_list: list[str] = awm_scenario.get("feature_list", [])
    category = awm_scenario.get("category", "")

    public_description = f"Negotiation set in: {name}. {description}".strip()
    if category:
        public_description += f" Category: {category}."

    constraints_from_features: list[str] = []
    if feature_list:
        constraints_from_features = [f"must_support_{f.lower().replace(' ', '_')[:40]}" for f in feature_list[:3]]
        constraints_from_features.append("deal_must_cover_core_requirements")

    buyer_batna = _estimate_batna(domain_enum, "buyer", awm_scenario)
    seller_batna = _estimate_batna(domain_enum, "seller", awm_scenario)

    buyer_ctx = PrivateContext(
        batna=buyer_batna,
        reserve_price=buyer_batna * 1.4,
        hard_constraints=constraints_from_features,
        private_info=[
            f"budget_is_{int(buyer_batna * 2)}",
            "alternatives_available",
        ],
    )

    seller_ctx = PrivateContext(
        batna=seller_batna,
        reserve_price=seller_batna * 0.6,
        hard_constraints=["minimum_terms_must_be_met"],
        private_info=[
            f"cost_basis_is_{int(seller_batna * 0.5)}",
            f"inventory_position_is_strong",
        ],
    )

    forbidden: list[str] = []
    if feature_list:
        key_feature = feature_list[0] if feature_list else "requirements"
        forbidden.append(f"cannot_claim_no_support_for_{key_feature.lower().replace(' ', '_')[:40]}")

    scenario = Scenario(
        id=scenario_id,
        domain=domain_enum,
        culture=culture,
        buyer_context=buyer_ctx,
        seller_context=seller_ctx,
        deal_schema=_DOMAIN_SCHEMAS[domain_enum],
        forbidden_claims=forbidden,
        scenario_description=public_description,
    )

    return Scenario.model_validate(scenario.model_dump())


_BASE_BATNAS: dict[Domain, tuple[float, float]] = {
    Domain.ECOMMERCE: (3000, 5000),
    Domain.SAAS_PROCUREMENT: (50000, 75000),
    Domain.SETTLEMENT: (20000, 50000),
    Domain.ETHICAL_BUSINESS: (15000, 25000),
}


def _estimate_batna(domain: Domain, role: str, awm_scenario: dict) -> float:
    base_buyer, base_seller = _BASE_BATNAS[domain]
    base = base_buyer if role == "buyer" else base_seller
    features = awm_scenario.get("feature_list", [])
    variation = len(features) * 500
    return base + variation
