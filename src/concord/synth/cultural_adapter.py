from concord.exceptions import ConcordError
from concord.schemas.culture import CULTURAL_PROFILES, Culture, CulturalProfile
from concord.schemas.scenario import Scenario


class CulturalAdapterError(ConcordError):
    pass


def adapt_for_culture(scenario: Scenario, target_culture: Culture) -> Scenario:
    if target_culture not in CULTURAL_PROFILES:
        raise CulturalAdapterError(f"Unknown culture: {target_culture}")

    profile = CULTURAL_PROFILES[target_culture]
    adapted = scenario.model_copy(deep=True)
    adapted.culture = target_culture.value

    adapted.scenario_description = _adapt_description(
        scenario.scenario_description, profile
    )
    adapted.buyer_context.private_info = _adapt_private_info(
        scenario.buyer_context.private_info, profile
    )
    adapted.seller_context.private_info = _adapt_private_info(
        scenario.seller_context.private_info, profile
    )
    adapted.buyer_context.relationship_history = _adapt_relationship_history(
        scenario.buyer_context.relationship_history, profile
    )
    adapted.seller_context.relationship_history = _adapt_relationship_history(
        scenario.seller_context.relationship_history, profile
    )

    return adapted


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
