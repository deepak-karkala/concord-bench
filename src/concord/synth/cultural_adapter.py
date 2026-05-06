from __future__ import annotations

import yaml

from concord.exceptions import ConcordError
from concord.schemas.culture import CULTURAL_PROFILES, Culture, CulturalProfile
from concord.schemas.scenario import Scenario


class CulturalAdapterError(ConcordError):
    pass


_CULTURE_PROFILES: dict[str, str] = {
    "JP": """- Consensus-driven (nemawashi); individual negotiators rarely have final authority
- Indirect communication; "that would be difficult" means no
- Relationships (kankei) precede business; trust-building before deal-making
- Contracts are frameworks; renegotiation expected as context changes
- Group harmony (wa); concessions that cause face-loss poison the relationship
- Silence is meaningful and deliberate""",

    "IN": """- Negotiation is expected to be extended; first offer is never accepted
- Price haggling is normal; opening offers are anchors
- Relationships and deal-making happen simultaneously (faster than Japan)
- Flexibility on terms is high; creative structures welcome
- Last-minute additions are common ("Indian last-minute")
- Decision-makers may not be in the room""",

    "BR": """- Relationship first, business second (jeitinho — finding creative workarounds)
- Personal warmth and social bonding are prerequisites
- Expressive communication; emotion and enthusiasm are appropriate
- Creative problem-solving celebrated; workarounds are not bad faith
- Commitments made in good faith may shift; renegotiation is normal""",

    "MENA": """- Wasta (connections/influence) — who introduced the parties matters
- Hospitality is obligatory; refusing tea signals bad faith
- Important decisions happen in informal majlis settings
- Islam-informed business ethics: no riba (interest), halal compliance
- Honor and face-saving are paramount; public confrontation destroys deals
- Verbal commitments carry significant weight; handshake = binding""",
}

_ADAPTATION_INSTRUCTIONS: dict[str, str] = {
    "JP": """1. Rewrite scenario_description: mention prior relationship-building phase; current meeting is a formal session after internal consensus (ringi).
2. Add to buyer private_info: internal_ringi_approval_required_for_deal_above_X and relationship_with_counterpart_spans_5_years.
3. Add to seller private_info: direct_rejection_will_damage_relationship and have_hosted_buyer_at_factory_visit_last_year.
4. Reframe hard_constraints from direct demands to relationship language.
5. Add relationship_history: exchanged_gifts_at_last_meeting_signaling_commitment.
6. Add forbidden_claim: cannot_directly_say_no_or_reject_offer_bluntly.""",

    "IN": """1. Rewrite scenario_description: ongoing 3-4 week negotiation; non-linear progress.
2. Add to buyer private_info: senior_decision_maker_not_present_today and expects_at_least_3_rounds_of_counter_offers.
3. Add to seller private_info: has_entertained_buyer_for_dinner and flexible_on_payment_terms_if_price_is_met.
4. Add late-introduction element: planning_to_add_extended_warranty_request_late.
5. Add relationship_history: met_at_trade_show_in_mumbai_established_rapport.""",

    "BR": """1. Rewrite scenario_description: negotiators have personal connection from São Paulo industry event; meeting starts with 30 min personal conversation.
2. Add to buyer private_info: decision_maker_is_personal_friend_of_counterpart and flexible_on_payment_terms_if_relationship_is_strong.
3. Add to seller private_info: hosting_buyer_for_churrasco_dinner_after_meeting and regulatory_workaround_available_for_import_restriction.
4. Add hard_constraint (buyer): deal_must_be_approved_by_board_junta.
5. Add relationship_history: played_soccer_together_at_industry_event.""",

    "MENA": """1. Rewrite scenario_description: meeting arranged through trusted intermediary (wasta); formal majlis setting.
2. Add to buyer private_info: introduced_by_mutual_contact_at_saudi_aramco and payment_via_islamic_finance_instrument_required.
3. Add to seller private_info: senior_partner_will_make_final_call_outside_meeting and must_not_publicly_disagree_with_buyer_in_front_of_their_team.
4. Add hard_constraint: deal_cannot_involve_interest_payments (riba prohibition).
5. Add relationship_history: introduced_at_world_future_energy_summit_abu_dhabi.""",
}

_ADAPTATION_PROMPT = """You are adapting a business negotiation scenario to reflect authentic {culture_name} negotiation norms and communication styles.

BASE SCENARIO:
{scenario_yaml}

CULTURAL PROFILE — {culture_name}:
{culture_profile}

ADAPTATION INSTRUCTIONS:
{adaptation_instructions}

Return a modified scenario YAML with ONLY these fields changed:
scenario_description, private_info strings, hard_constraints wording, relationship_history context.

Do NOT change: id, domain, batna values, reserve_price, deal_schema structure, forbidden_claims.

Return valid YAML only. Do not include markdown code fences."""


def adapt_for_culture(scenario: Scenario, target_culture: Culture) -> Scenario:
    """Adapt a scenario for a target culture using deterministic Hofstede-based transformation.

    For LLM-based adaptation (higher quality), use adapt_for_culture_llm().
    """
    if target_culture not in CULTURAL_PROFILES:
        raise CulturalAdapterError(f"Unknown culture: {target_culture}")

    if target_culture.value == "US":
        adapted = scenario.model_copy(deep=True)
        adapted.culture = "US"
        return adapted

    profile = CULTURAL_PROFILES[target_culture]
    adapted = scenario.model_copy(deep=True)
    adapted.culture = target_culture.value

    adapted.scenario_description = _adapt_description(scenario.scenario_description, profile)
    adapted.buyer_context.private_info = _adapt_private_info(scenario.buyer_context.private_info, profile)
    adapted.seller_context.private_info = _adapt_private_info(scenario.seller_context.private_info, profile)
    adapted.buyer_context.relationship_history = _adapt_relationship_history(
        scenario.buyer_context.relationship_history, profile
    )
    adapted.seller_context.relationship_history = _adapt_relationship_history(
        scenario.seller_context.relationship_history, profile
    )

    return adapted


async def adapt_for_culture_llm(
    scenario: Scenario,
    target_culture: Culture,
    model: str = "deepseek-v4-pro",
) -> Scenario:
    """Adapt a scenario for a target culture via LLM call for higher authenticity."""
    if target_culture.value == "US":
        adapted = scenario.model_copy(deep=True)
        adapted.culture = "US"
        return adapted

    cult_code = target_culture.value
    if cult_code not in _CULTURE_PROFILES:
        return adapt_for_culture(scenario, target_culture)

    scenario_yaml = yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False, default_flow_style=False)
    prompt = _ADAPTATION_PROMPT.format(
        culture_name=cult_code,
        scenario_yaml=scenario_yaml,
        culture_profile=_CULTURE_PROFILES[cult_code],
        adaptation_instructions=_ADAPTATION_INSTRUCTIONS[cult_code],
    )

    try:
        from concord.synth.enrichment import _call_llm
        response_yaml = await _call_llm(prompt, model=model)
        data = yaml.safe_load(response_yaml)
        if not isinstance(data, dict):
            raise CulturalAdapterError("LLM returned non-dict YAML")
        data["culture"] = cult_code
        return Scenario.model_validate(data)
    except Exception:
        return adapt_for_culture(scenario, target_culture)


def _adapt_description(desc: str, profile: CulturalProfile) -> str:
    style_marker = f"[Culture: {profile.communication_style}]"
    norms = "; ".join(profile.negotiation_norms[:2]) if profile.negotiation_norms else ""
    return f"{desc} {style_marker} Cultural norms: {norms}".strip()


def _adapt_private_info(info: list[str], profile: CulturalProfile) -> list[str]:
    adapted = list(info)
    if profile.acceptable_tactics:
        tactic_sample = profile.acceptable_tactics[0] if profile.acceptable_tactics else ""
        adapted.append(f"culture_aware_tactic: {tactic_sample}")
    adapted.append(f"culture_power_distance: {profile.power_distance}")
    return adapted


def _adapt_relationship_history(history: list[str], profile: CulturalProfile) -> list[str]:
    adapted = list(history)
    if profile.negotiation_norms:
        norm_sample = profile.negotiation_norms[0] if profile.negotiation_norms else ""
        adapted.append(f"cultural_norm: {norm_sample}")
    return adapted
