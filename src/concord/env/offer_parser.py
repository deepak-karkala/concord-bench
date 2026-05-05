import json
import re

from pydantic import TypeAdapter, ValidationError

from concord.exceptions import ConcordError
from concord.schemas.offer import Offer

offer_adapter = TypeAdapter(Offer)


class OfferParseError(ConcordError):
    pass


def parse_offer_json(raw: str, domain: str) -> Offer:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise OfferParseError(f"Failed to parse offer as JSON: {e}") from e

    if not isinstance(data, dict):
        raise OfferParseError(f"Offer must be a JSON object, got {type(data).__name__}")

    if "domain" not in data:
        data["domain"] = domain

    try:
        return offer_adapter.validate_python(data)
    except ValidationError as e:
        raise OfferParseError(f"Offer validation failed: {e}") from e


def parse_offer_regex(raw: str, domain: str) -> Offer:
    domain_literal_map = {
        "ecommerce": "ecommerce",
        "saas_procurement": "saas_procurement",
        "settlement": "settlement",
        "ethical_business": "ethical_business",
    }
    domain_value = domain_literal_map.get(domain, domain)
    data: dict = {"domain": domain_value}

    price_match = re.search(r"(?:price|amount|settlement_amount)\s*(?:is|:|=)\s*\$?([\d,.]+)", raw, re.IGNORECASE)
    if price_match:
        key = "price"
        if domain == "settlement":
            key = "settlement_amount"
        elif domain == "saas_procurement":
            key = "monthly_price"
        try:
            data[key] = float(price_match.group(1).replace(",", ""))
        except ValueError:
            pass

    quantity_match = re.search(r"(?:quantity|units|seats)\s*(?:is|:|=|\s+)\s*(\d+)", raw, re.IGNORECASE)
    if quantity_match:
        key = "quantity" if domain in ("ecommerce", "ethical_business") else "seats"
        data[key] = int(quantity_match.group(1))

    contract_match = re.search(
        r"(?:contract(?:\s+length)?|months?)\s*(?:is|:|=|\s+)\s*(\d+)"
        r"|(\d+)\s+months?",
        raw, re.IGNORECASE,
    )
    if contract_match:
        months_val = contract_match.group(1) or contract_match.group(2)
        data["contract_length_months"] = int(months_val)

    shipping_match = re.search(r"(?:shipping)\s*(?:is|:|=)\s*(.+)", raw, re.IGNORECASE)
    if shipping_match:
        data["shipping_terms"] = shipping_match.group(1).strip().rstrip(". ") or "standard"

    try:
        return offer_adapter.validate_python(data)
    except ValidationError as e:
        raise OfferParseError(f"Offer validation failed after regex extraction: {e}") from e


def parse_offer(raw: str, domain: str, use_constrained: bool = False) -> Offer:
    if use_constrained:
        return parse_offer_json(raw, domain)

    try:
        return parse_offer_json(raw, domain)
    except OfferParseError:
        return parse_offer_regex(raw, domain)
