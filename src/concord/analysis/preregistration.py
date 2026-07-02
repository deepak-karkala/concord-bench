from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from concord.exceptions import ConcordError


class PreRegistrationViolationError(ConcordError):
    pass


class PreRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    registered_at: str
    status: str
    description: str
    primary_metrics: list[str]
    secondary_metrics: list[str]
    exploratory_metrics: list[str]
    unsupported_metrics: list[str]
    minimum_sample_sizes: dict[str, int] = Field(default_factory=dict)
    required_repeated_runs: int
    allowed_comparison_families: list[str] = Field(default_factory=list)
    cross_family_comparison_requires: list[str] = Field(default_factory=list)
    allowed_strong_claims: list[str] = Field(default_factory=list)
    disallowed_claims: list[str] = Field(default_factory=list)
    acceptance_thresholds: dict[str, float] = Field(default_factory=dict)

    def all_known_metrics(self) -> set[str]:
        return (
            set(self.primary_metrics)
            | set(self.secondary_metrics)
            | set(self.exploratory_metrics)
            | set(self.unsupported_metrics)
        )

    def category_for_metric(self, metric_name: str) -> str | None:
        if metric_name in self.primary_metrics:
            return "primary"
        if metric_name in self.secondary_metrics:
            return "secondary"
        if metric_name in self.exploratory_metrics:
            return "exploratory"
        if metric_name in self.unsupported_metrics:
            return "unsupported"
        return None


def load_preregistration(path: Path) -> PreRegistration:
    data = json.loads(path.read_text())
    return PreRegistration.model_validate(data)


def validate_metric_status(metric_name: str, preregistration: PreRegistration) -> None:
    category = preregistration.category_for_metric(metric_name)
    if category is None:
        raise PreRegistrationViolationError(
            f"Metric '{metric_name}' is not in pre-registration. Add it explicitly before using it in claims."
        )
    if category == "unsupported":
        raise PreRegistrationViolationError(
            f"Metric '{metric_name}' is listed as unsupported in the pre-registration and must not appear in claims."
        )


def validate_report_against_preregistration(
    *,
    metric_claim_tiers: dict[str, list[str]],
    slice_counts: dict[str, int],
    preregistration: PreRegistration,
) -> None:
    for tier_name, metric_names in metric_claim_tiers.items():
        if tier_name not in {"primary", "secondary", "exploratory"}:
            continue
        for metric_name in metric_names:
            category = preregistration.category_for_metric(metric_name)
            if category is None:
                raise PreRegistrationViolationError(
                    f"Metric '{metric_name}' appears in report tier '{tier_name}' but is not in the pre-registration."
                )
            if category == "unsupported":
                raise PreRegistrationViolationError(
                    f"Metric '{metric_name}' appears in report tier '{tier_name}' but is unsupported in the pre-registration."
                )
            if category != tier_name:
                raise PreRegistrationViolationError(
                    f"Metric '{metric_name}' appears in report tier '{tier_name}' but is registered as '{category}'."
                )

    for key, minimum in preregistration.minimum_sample_sizes.items():
        actual = slice_counts.get(key)
        if actual is None:
            continue
        if actual < minimum:
            raise PreRegistrationViolationError(
                f"Slice '{key}' has n={actual}, below the preregistered minimum of {minimum}."
            )
