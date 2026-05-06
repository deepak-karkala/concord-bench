from __future__ import annotations

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
            "inventory_position_is_strong",
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


_NARRATIVE_PROMPT = """Rewrite the scenario_description for this negotiation scenario to make it vivid and realistic. Write 2-3 sentences that tell a business story.

Include:
- Who each party is (company type, size, context)
- Why this deal matters to each party right now
- What tension or pressure makes this negotiation non-trivial
- Any relevant backstory (prior relationship, past deal, market context)

Do NOT reveal BATNA values or private information. Keep it from a neutral observer's perspective.

CURRENT DESCRIPTION: {current_description}
DOMAIN: {domain}
BUYER PRIVATE INFO (use for context, do not reveal): {buyer_private_info}
SELLER PRIVATE INFO (use for context, do not reveal): {seller_private_info}

Return only the new scenario_description text (2-3 sentences, no YAML wrapper)."""


async def add_narrative_description(scenario: Scenario, model: str = "deepseek-v4-pro") -> Scenario:
    """Rewrite scenario_description to a rich narrative paragraph via LLM call."""
    prompt = _NARRATIVE_PROMPT.format(
        current_description=scenario.scenario_description,
        domain=scenario.domain,
        buyer_private_info=scenario.buyer_context.private_info,
        seller_private_info=scenario.seller_context.private_info,
    )
    new_description = await _call_llm(prompt, model=model)
    return scenario.model_copy(update={"scenario_description": new_description.strip()})


async def _call_llm(prompt: str, model: str) -> str:
    """Call an LLM and return the text response. Supports Anthropic, OpenAI, and DeepSeek models."""
    import os

    if "claude" in model.lower():
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
    elif "deepseek" in model.lower():
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY"))
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
