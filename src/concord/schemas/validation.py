from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from concord.schemas.offer import Offer
from concord.schemas.scenario import Domain, PrivateContext


class ValidationLabels(BaseModel):
    deal_quality: float | None = Field(default=None, ge=0, le=1)
    has_false_claims: bool | None = None
    coercion_level: float | None = Field(default=None, ge=0, le=1)
    cultural_appropriateness: float | None = Field(default=None, ge=0, le=1)
    privacy_leak: bool | None = None
    walk_away_correct: bool | None = None
    hard_constraint_violation: bool | None = None


class ValidationContext(BaseModel):
    buyer_private_context: PrivateContext | None = None
    seller_private_context: PrivateContext | None = None
    deal: Offer | None = None
    buyer_engaged: bool | None = None
    zopa_exists: bool | None = None
    forbidden_claims: list[str] | None = None
    annotation_notes: list[str] = Field(default_factory=list)
    annotation_source: str | None = None


class ValidationRecord(BaseModel):
    transcript_id: str
    domain: Domain
    culture: str
    transcript: list[dict[str, Any]]
    labels: ValidationLabels = Field(default_factory=ValidationLabels)
    validation_context: ValidationContext | None = None


class ValidationAnnotationOverlay(BaseModel):
    transcript_id: str
    labels: ValidationLabels = Field(default_factory=ValidationLabels)
    validation_context: ValidationContext | None = None


def load_annotation_overlays(path: Path | None) -> dict[str, ValidationAnnotationOverlay]:
    if path is None:
        return {}

    overlays: dict[str, ValidationAnnotationOverlay] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        overlay = ValidationAnnotationOverlay.model_validate_json(line)
        overlays[overlay.transcript_id] = overlay
    return overlays


def load_validation_record(
    payload: dict[str, Any],
    *,
    overlay: ValidationAnnotationOverlay | None = None,
) -> dict[str, Any]:
    merged = json.loads(json.dumps(payload))
    if overlay is not None:
        merged_labels = dict(merged.get("labels") or {})
        merged_labels.update(overlay.labels.model_dump(exclude_none=True))
        merged["labels"] = merged_labels
        if overlay.validation_context is not None:
            merged_validation_context = dict(merged.get("validation_context") or {})
            merged_validation_context.update(
                overlay.validation_context.model_dump(
                    exclude_none=True,
                    mode="json",
                )
            )
            merged["validation_context"] = merged_validation_context

    record = ValidationRecord.model_validate(merged)
    return record.model_dump(exclude_none=True, mode="json")
