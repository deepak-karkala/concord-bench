from __future__ import annotations

import re
from typing import Any


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
