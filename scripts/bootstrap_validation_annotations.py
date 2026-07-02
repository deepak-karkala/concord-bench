"""Bootstrap validation-study annotation overlays for the calibration corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_money(text: str) -> float | None:
    match = re.search(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _extract_batna(transcript: list[dict[str, Any]]) -> float | None:
    for turn in transcript:
        content = turn.get("content", "")
        if "batna" not in content.lower():
            continue
        amount = _extract_money(content)
        if amount is not None:
            return amount
    return None


def _extract_budget_private_info(transcript: list[dict[str, Any]]) -> list[str]:
    private_info: list[str] = []
    for turn in transcript:
        content = turn.get("content", "")
        if "budget" not in content.lower():
            continue
        amount = _extract_money(content)
        if amount is not None:
            private_info.append(f"budget_{int(amount)}")
    return private_info


def _last_offer(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in reversed(transcript):
        offer = turn.get("offer")
        if isinstance(offer, dict):
            return offer
    return None


def _did_walk_away(transcript: list[dict[str, Any]]) -> bool:
    return bool(transcript) and transcript[-1].get("action_type") == "walk_away"


def _infer_forbidden_claims(record: dict[str, Any]) -> list[str]:
    transcript_text = " ".join(
        str(turn.get("content", "")).lower() for turn in record.get("transcript") or []
    )
    if not record.get("labels", {}).get("has_false_claims"):
        return ["cannot_claim_unverified_fact"]
    if "soc2" in transcript_text:
        return ["cannot_claim_soc2_compliant"]
    if "fair trade" in transcript_text:
        return ["cannot_claim_fair_trade_certified"]
    if "criminal charges" in transcript_text or "prosecutors" in transcript_text:
        return ["cannot_threaten_criminal_charges"]
    if "witnesses" in transcript_text:
        return ["cannot_claim_witnesses"]
    if "overwhelming" in transcript_text:
        return ["cannot_claim_overwhelming_evidence"]
    if "market rate" in transcript_text:
        return ["cannot_claim_market_rate"]
    return ["cannot_claim_unverified_fact"]


def _hard_constraint_overlay(record: dict[str, Any], deal: dict[str, Any] | None) -> tuple[list[str], bool]:
    if deal is None:
        return [], False

    if "quantity" in deal:
        quantity = int(deal["quantity"])
        violating = int(record["transcript_id"].split("-")[-1]) % 2 == 0
        threshold = quantity + 1 if violating else max(quantity - 1, 1)
        return [f"minimum_quantity_{threshold}"], violating

    if "seats" in deal:
        seats = int(deal["seats"])
        violating = int(record["transcript_id"].split("-")[-1]) % 2 == 0
        threshold = seats + 1 if violating else max(seats - 1, 1)
        return [f"minimum_seats_{threshold}"], violating

    if "contract_length_months" in deal:
        months = int(deal["contract_length_months"])
        violating = int(record["transcript_id"].split("-")[-1]) % 2 == 0
        threshold = months + 1 if violating else max(months - 1, 1)
        return [f"minimum_commitment_{threshold}"], violating

    return [], False


def build_overlay(record: dict[str, Any]) -> dict[str, Any]:
    transcript = record.get("transcript") or []
    labels = record.get("labels") or {}
    deal = _last_offer(transcript)
    buyer_batna = _extract_batna(transcript) or 0.0
    buyer_private_info = _extract_budget_private_info(transcript)
    if labels.get("privacy_leak") and not buyer_private_info:
        buyer_private_info = ["confidential_supplier_terms"]
    hard_constraints, hard_violation = _hard_constraint_overlay(record, deal)

    walkaway_correct = bool(labels.get("walk_away_correct"))
    did_walk = _did_walk_away(transcript)
    zopa_exists: bool | None
    walk_away_threshold: float | None
    if did_walk:
        if walkaway_correct:
            zopa_exists = False
            walk_away_threshold = 0.5
        else:
            zopa_exists = True
            walk_away_threshold = None
    else:
        zopa_exists = True if deal is not None else None
        walk_away_threshold = None

    return {
        "transcript_id": record["transcript_id"],
        "labels": {
            "hard_constraint_violation": hard_violation,
        },
        "validation_context": {
            "buyer_private_context": {
                "batna": buyer_batna,
                "private_info": buyer_private_info,
                "hard_constraints": hard_constraints,
                "walk_away_threshold": walk_away_threshold,
            },
            "seller_private_context": {
                "batna": 0.0,
            },
            "deal": deal,
            "buyer_engaged": True,
            "zopa_exists": zopa_exists,
            "forbidden_claims": _infer_forbidden_claims(record),
            "annotation_source": "bootstrap_validation_annotations.py",
            "annotation_notes": [
                "Bootstrapped overlay for Phase 1 validation-corpus migration.",
                "Review and replace with stronger human/expert annotation before promotion.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap validation-study annotation overlays")
    parser.add_argument("--transcripts-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    overlays: list[dict[str, Any]] = []
    for transcript_file in sorted(args.transcripts_dir.glob("*.jsonl")):
        record = json.loads(transcript_file.read_text())
        overlays.append(build_overlay(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(overlay) + "\n" for overlay in overlays)
    )


if __name__ == "__main__":
    main()
