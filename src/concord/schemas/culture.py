from enum import StrEnum

from pydantic import BaseModel, Field


class Culture(StrEnum):
    US = "US"
    JP = "JP"
    IN = "IN"
    BR = "BR"
    MENA = "MENA"


class CulturalProfile(BaseModel):
    communication_style: str = Field(
        description="e.g., direct, indirect, high-context, low-context"
    )
    power_distance: int = Field(ge=0, le=100, description="Hofstede power distance index")
    individualism: int = Field(ge=0, le=100, description="Hofstede individualism index")
    uncertainty_avoidance: int = Field(ge=0, le=100, description="Hofstede uncertainty avoidance index")
    long_term_orientation: int = Field(ge=0, le=100, description="Hofstede long-term orientation index")
    indulgence: int = Field(ge=0, le=100, description="Hofstede indulgence index")
    negotiation_norms: list[str] = Field(
        default_factory=list,
        description="Culture-specific negotiation conventions",
    )
    acceptable_tactics: list[str] = Field(
        default_factory=list,
        description="Tactics considered acceptable in this culture",
    )
    taboo_tactics: list[str] = Field(
        default_factory=list,
        description="Tactics considered unacceptable in this culture",
    )


CULTURAL_PROFILES: dict[Culture, CulturalProfile] = {
    Culture.US: CulturalProfile(
        communication_style="direct, low-context",
        power_distance=40,
        individualism=91,
        uncertainty_avoidance=46,
        long_term_orientation=26,
        indulgence=68,
        negotiation_norms=[
            "Direct proposals are expected",
            "Time is money; get to the point",
            "Contracts are binding once signed",
        ],
        acceptable_tactics=[
            "Competitive opening offers",
            "Logos-heavy persuasion with data",
            "Walk-away threats within reason",
        ],
        taboo_tactics=[
            "Personal attacks unrelated to the deal",
            "Bribes or kickbacks",
        ],
    ),
    Culture.JP: CulturalProfile(
        communication_style="indirect, high-context",
        power_distance=54,
        individualism=46,
        uncertainty_avoidance=92,
        long_term_orientation=88,
        indulgence=42,
        negotiation_norms=[
            "Relationship building before business",
            "Decisions by consensus (nemawashi)",
            "Written confirmation after verbal agreement",
        ],
        acceptable_tactics=[
            "Silence as negotiation pressure",
            "Appeals to long-term relationship",
            "Incremental concessions",
        ],
        taboo_tactics=[
            "Aggressive confrontation",
            "Public disagreement with senior counterpart",
            "Ultimatums",
        ],
    ),
    Culture.IN: CulturalProfile(
        communication_style="indirect but assertive, high-context",
        power_distance=77,
        individualism=48,
        uncertainty_avoidance=40,
        long_term_orientation=51,
        indulgence=26,
        negotiation_norms=[
            "Price negotiation is expected",
            "Relationship and trust precede terms",
            "Flexibility on timelines",
        ],
        acceptable_tactics=[
            "Emotional appeals",
            "Third-party references and connections",
            "Package deals with non-monetary benefits",
        ],
        taboo_tactics=[
            "Dismissing hierarchy or status",
            "Rushing to close without relationship",
        ],
    ),
    Culture.BR: CulturalProfile(
        communication_style="warm, relationship-oriented, moderate-context",
        power_distance=69,
        individualism=38,
        uncertainty_avoidance=76,
        long_term_orientation=44,
        indulgence=59,
        negotiation_norms=[
            "Personal relationships are essential",
            "Face-to-face preferred over written",
            "Flexibility and improvisation valued",
        ],
        acceptable_tactics=[
            "Warmth and personal connection",
            "Flexible terms and creative solutions",
            "Multiple rounds of haggling",
        ],
        taboo_tactics=[
            "Cold, purely transactional approach",
            "Bypassing the relationship-builder",
        ],
    ),
    Culture.MENA: CulturalProfile(
        communication_style="elaborate, relationship-centric, high-context",
        power_distance=80,
        individualism=38,
        uncertainty_avoidance=68,
        long_term_orientation=36,
        indulgence=34,
        negotiation_norms=[
            "Extensive relationship building",
            "Hospitality is part of business",
            "Oral agreement carries weight",
        ],
        acceptable_tactics=[
            "Elaborate persuasion and storytelling",
            "Appeals to honor and reputation",
            "Using intermediaries",
        ],
        taboo_tactics=[
            "Aggressive time pressure",
            "Disrespecting elders or hierarchy",
            "Public loss of face",
        ],
    ),
}
