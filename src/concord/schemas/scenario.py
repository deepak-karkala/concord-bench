from enum import StrEnum

from pydantic import BaseModel, Field


class Domain(StrEnum):
    ECOMMERCE = "ecommerce"
    SAAS_PROCUREMENT = "saas_procurement"
    SETTLEMENT = "settlement"
    ETHICAL_BUSINESS = "ethical_business"


class PrivateContext(BaseModel):
    batna: float = Field(description="Best alternative to negotiated agreement (value)")
    reserve_price: float | None = Field(
        default=None, description="Walk-away price; deal must beat this"
    )
    hard_constraints: list[str] = Field(
        default_factory=list,
        description="Non-negotiable constraints that must not be violated",
    )
    private_info: list[str] = Field(
        default_factory=list,
        description="Information the agent possesses but counterparty may not",
    )
    walk_away_threshold: float | None = Field(
        default=None, description="Utility below which agent should walk away"
    )
    reputation: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Reputation score 0-1 (used in repeated-game track)",
    )
    relationship_history: list[str] = Field(
        default_factory=list,
        description="Prior negotiation outcomes with this counterparty",
    )


class Scenario(BaseModel):
    id: str = Field(description="Unique scenario identifier")
    domain: Domain = Field(description="Negotiation domain")
    culture: str = Field(default="US", description="Cultural context (ISO or label)")
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum negotiation turns")
    buyer_context: PrivateContext = Field(description="Buyer's private information")
    seller_context: PrivateContext = Field(description="Seller's private information")
    deal_schema: dict = Field(description="Structured format for offers in this domain")
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="Claims the agent is explicitly forbidden from making",
    )
    scenario_description: str = Field(
        default="",
        description="Public description of the negotiation context visible to both parties",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional enrichment metadata (difficulty_tier, pressure_type, etc.)",
    )
