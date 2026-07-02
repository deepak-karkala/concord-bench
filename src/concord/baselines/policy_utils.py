from __future__ import annotations

import re
from typing import Any

from concord.graders.constraints import assess_hard_constraints
from concord.schemas.episode import ActionType, Turn
from concord.schemas.scenario import PrivateContext


def build_offer_from_schema(
    deal_schema: dict[str, Any],
    *,
    numeric_anchor: float,
    constraints: list[str],
    string_value: str = "standard",
    list_value: list[str] | None = None,
) -> dict[str, Any]:
    offer: dict[str, Any] = {}
    for field_name, field_type in deal_schema.items():
        if field_type in {"float", "int"}:
            if "price" in field_name or "amount" in field_name:
                offer[field_name] = numeric_anchor
            elif "seats" in field_name or "quantity" in field_name:
                offer[field_name] = minimum_int_for_field(field_name, constraints)
            elif "months" in field_name:
                offer[field_name] = minimum_int_for_field(field_name, constraints)
            else:
                offer[field_name] = 1 if field_type == "int" else float(max(1, numeric_anchor))
        elif field_type == "bool":
            offer[field_name] = True
        elif field_type == "str":
            offer[field_name] = string_value
        elif field_type == "list":
            offer[field_name] = list_value or ["standard_commitment"]
    return offer


def minimum_int_for_field(field_name: str, constraints: list[str]) -> int:
    minima = [1]
    for constraint in constraints:
        if "quantity" in field_name:
            match = re.search(r"minimum_(?:order|quantity)_(\d+)", constraint, re.IGNORECASE)
            if match:
                minima.append(int(match.group(1)))
        elif "seats" in field_name:
            match = re.search(r"minimum_(?:(\d+)_seats|seats_(\d+))", constraint, re.IGNORECASE)
            if match:
                minima.append(int(match.group(1) or match.group(2)))
        elif "month" in field_name:
            match = re.search(
                r"(?:minimum_)?(?:commitment_)?(\d+)_(?:month|year)|minimum_commitment_(\d+)",
                constraint,
                re.IGNORECASE,
            )
            if match:
                months = int(match.group(1) or match.group(2))
                if "year" in constraint.lower():
                    months *= 12
                minima.append(months)
    return max(minima)


def extract_price_from_offer(offer_dict: dict[str, Any] | None) -> float | None:
    if not offer_dict:
        return None
    for attr in ("price", "settlement_amount", "monthly_price"):
        value = offer_dict.get(attr)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def current_offer_threshold(
    *,
    reserve_price: float | None,
    batna: float,
    current_turn: int,
    max_turns: int,
    stance: str,
) -> float:
    floor = float(reserve_price if reserve_price is not None else batna)
    aspiration = max(float(batna), floor)
    gap = max(0.0, aspiration - floor)
    total_rounds = max(1, max_turns)
    progress = min(current_turn / total_rounds, 1.0)

    if stance == "cooperative":
        start_share = 0.55
        end_share = 0.10
    elif stance == "hardball":
        start_share = 0.95
        end_share = 0.35
    else:
        start_share = 1.10
        end_share = 0.55

    concession_share = max(end_share, start_share - (start_share - end_share) * progress)
    return floor + gap * concession_share


def seller_should_accept(
    last_counterparty_turn: Turn | None,
    *,
    private_ctx: PrivateContext,
    reserve_price: float | None,
    batna: float,
    current_turn: int,
    max_turns: int,
    stance: str,
) -> bool:
    if last_counterparty_turn is None or last_counterparty_turn.offer is None:
        return False
    if last_counterparty_turn.action_type != ActionType.OFFER:
        return False

    offered_price = extract_price_from_offer(last_counterparty_turn.offer.model_dump())
    if offered_price is None:
        return False

    floor = float(reserve_price if reserve_price is not None else batna)
    if offered_price < floor:
        return False

    violations, unverifiable = assess_hard_constraints(last_counterparty_turn.offer, private_ctx)
    if violations or unverifiable:
        return False

    threshold = current_offer_threshold(
        reserve_price=reserve_price,
        batna=batna,
        current_turn=current_turn,
        max_turns=max_turns,
        stance=stance,
    )
    return offered_price >= threshold
