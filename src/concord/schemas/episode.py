from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from concord.schemas.offer import Offer


class ActionType(StrEnum):
    MESSAGE = "message"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    WALK_AWAY = "walk_away"
    ESCALATE = "escalate"


class ImpasseOutcome(StrEnum):
    DEAL = "deal"                          # episode ended with a deal
    BUYER_WALK_AWAY = "buyer_walk_away"    # buyer explicitly walked away
    SELLER_REFUSAL = "seller_refusal"      # seller refused after valid buyer offer(s)
    PROTOCOL_FAILURE = "protocol_failure"  # no meaningful buyer action (silence, zero offers)
    MUTUAL_IMPASSE = "mutual_impasse"      # both sides negotiated but couldn't agree
    TIMEOUT = "timeout"                    # max turns reached without deal or walk-away


class MetricStatus(StrEnum):
    HEADLINE_SAFE = "headline_safe"    # validated, supports strong claims
    SECONDARY = "secondary"            # useful but more caveated
    EXPLORATORY = "exploratory"        # research-only, not for strong claims
    UNSUPPORTED = "unsupported"        # structurally inert or invalid currently


METRIC_STATUS_REGISTRY: dict[str, MetricStatus] = {
    # Headline-safe: strongest current surfaces
    "principal_utility": MetricStatus.HEADLINE_SAFE,
    "deal_rate": MetricStatus.HEADLINE_SAFE,
    "joint_welfare": MetricStatus.SECONDARY,
    "bundle_quality": MetricStatus.SECONDARY,
    "pareto_efficient": MetricStatus.SECONDARY,

    # Walk-away: rebuilt in Phase 2 but not yet human-validated — secondary
    "walk_away_correct": MetricStatus.SECONDARY,

    # Constraints: regex-only, disclosed in CONSTRAINT_GRADER_COVERAGE — secondary
    "hard_constraint_violations": MetricStatus.SECONDARY,

    # Privacy: verbatim matching only, misses paraphrases — exploratory
    "privacy_leak": MetricStatus.EXPLORATORY,
    "batna_leaked": MetricStatus.EXPLORATORY,
    "private_info_leaked": MetricStatus.EXPLORATORY,

    # Pressure/coercion: regex heuristic, structurally flat — exploratory
    "coercion_score": MetricStatus.EXPLORATORY,

    # Cultural: US-only corpus, always 0.0 for US culture — unsupported
    "cultural_sensitivity_score": MetricStatus.UNSUPPORTED,

    # Engagement-conditioned: Phase 2 addition — secondary (conditioned variants)
    "engagement_metrics": MetricStatus.SECONDARY,

    # Impasse attribution: Phase 2 addition — secondary
    "impasse_outcome": MetricStatus.SECONDARY,

    # Other
    "turns_to_deal": MetricStatus.SECONDARY,
    "irrational_deal": MetricStatus.SECONDARY,
    "false_claims": MetricStatus.SECONDARY,
    "forbidden_claim_violations": MetricStatus.SECONDARY,
    "acceptance_reasoning_aligned": MetricStatus.EXPLORATORY,
    "relationship_preservation_score": MetricStatus.EXPLORATORY,
}


class Turn(BaseModel):
    agent: str = Field(description="Agent identifier (buyer or seller)")
    action_type: ActionType = Field(description="Type of action taken")
    content: str = Field(default="", description="Natural language message content")
    offer: Offer | None = Field(default=None, description="Structured offer, if action_type is offer")
    metadata: dict = Field(default_factory=dict, description="Turn-level protocol/runtime metadata")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )


class EngagementConditionedMetrics(BaseModel):
    """Discipline metrics conditioned on buyer engagement.

    For each metric: unconditional value is always reported.
    Conditioned value is None when buyer was not engaged (protocol failure).
    """

    buyer_engaged: bool = Field(description="Whether buyer made meaningful offers or messages")

    # Privacy discipline
    batna_leaked_unconditional: bool = Field(
        description="BATNA leaked in transcript (unconditional)"
    )
    batna_leaked_conditioned: bool | None = Field(
        default=None,
        description="BATNA leaked (None if buyer not engaged — metric not meaningful)",
    )
    private_info_leaked_unconditional: list[str] = Field(default_factory=list)
    private_info_leaked_conditioned: list[str] | None = Field(default=None)

    # Constraint discipline
    hard_constraint_violations_unconditional: list[str] = Field(default_factory=list)
    hard_constraint_violations_conditioned: list[str] | None = Field(default=None)

    # Pressure/coercion discipline
    coercion_score_unconditional: float | None = Field(default=None, ge=0, le=1)
    coercion_score_conditioned: float | None = Field(default=None, ge=0, le=1)

    # Denominator info
    engaged_buyer_turn_count: int = Field(
        default=0, description="Number of turns where buyer was engaged"
    )


class GradeReport(BaseModel):
    principal_utility: float | None = None
    joint_welfare: float | None = None
    bundle_quality: float | None = None
    pareto_efficient: bool | None = None
    walk_away_correct: bool | None = None
    hard_constraint_violations: list[str] = Field(default_factory=list)
    false_claims: list[str] = Field(default_factory=list)
    forbidden_claim_violations: list[str] = Field(default_factory=list)
    privacy_leak: bool = False
    batna_leaked: bool = False
    private_info_leaked: list[str] = Field(default_factory=list)
    coercion_score: float | None = Field(default=None, ge=0, le=1)
    cultural_sensitivity_score: float | None = Field(default=None, ge=0, le=1)
    relationship_preservation_score: float | None = Field(default=None, ge=0, le=1)
    turns_to_deal: int | None = None
    irrational_deal: bool = False
    acceptance_reasoning_aligned: bool | None = None
    impasse_outcome: ImpasseOutcome | None = Field(
        default=None,
        description="Episode-level impasse attribution: how the episode ended",
    )
    engagement_metrics: "EngagementConditionedMetrics | None" = None


class EpisodeLog(BaseModel):
    scenario_id: str = Field(description="Reference to the scenario played")
    turns: list[Turn] = Field(default_factory=list, description="Ordered turn-by-turn transcript")
    deal: Offer | None = Field(default=None, description="Final agreed deal, or None if no deal")
    grades: GradeReport = Field(default_factory=GradeReport, description="Grader scores")
    metadata: dict = Field(
        default_factory=dict,
        description="Run metadata: model IDs, prompt hash, seed, version, etc.",
    )

    @property
    def buyer_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.agent == "buyer"]

    @property
    def seller_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.agent == "seller"]

    @property
    def terminal(self) -> bool:
        if not self.turns:
            return False
        last = self.turns[-1]
        return last.action_type in (ActionType.ACCEPT, ActionType.WALK_AWAY)


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    confidence: float = 0.95


class DimensionScore(BaseModel):
    mean: float
    ci95: ConfidenceInterval | None = None
    n_episodes: int = 0


class ModelCard(BaseModel):
    model_id: str = Field(description="Model identifier with snapshot date")
    concord_version: str = Field(description="Concord version used for evaluation")
    outcome: dict[str, DimensionScore] = Field(default_factory=dict)
    constraints: dict[str, DimensionScore] = Field(default_factory=dict)
    social: dict[str, DimensionScore] = Field(default_factory=dict)
    robustness: dict[str, DimensionScore] = Field(default_factory=dict)
    total_episodes: int = 0
