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


class Turn(BaseModel):
    agent: str = Field(description="Agent identifier (buyer or seller)")
    action_type: ActionType = Field(description="Type of action taken")
    content: str = Field(default="", description="Natural language message content")
    offer: Offer | None = Field(default=None, description="Structured offer, if action_type is offer")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )


class GradeReport(BaseModel):
    principal_utility: float | None = None
    joint_welfare: float | None = None
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
