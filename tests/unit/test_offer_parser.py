import pytest

from concord.env.offer_parser import OfferParseError, parse_offer, parse_offer_json, parse_offer_regex
from concord.schemas.offer import EcommerceOffer, SaaSProcurementOffer, SettlementOffer


class TestParseOfferJSON:
    def test_valid_ecommerce_json(self):
        raw = '{"domain": "ecommerce", "price": 150.0, "quantity": 100}'
        offer = parse_offer_json(raw, "ecommerce")
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 150.0

    def test_valid_settlement_json(self):
        raw = '{"domain": "settlement", "settlement_amount": 75000, "confidentiality_clause": true}'
        offer = parse_offer_json(raw, "settlement")
        assert isinstance(offer, SettlementOffer)
        assert offer.settlement_amount == 75000

    def test_missing_domain_in_json(self):
        raw = '{"price": 200, "quantity": 300}'
        offer = parse_offer_json(raw, "ecommerce")
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 200

    def test_invalid_json(self):
        with pytest.raises(OfferParseError, match="Failed to parse offer as JSON"):
            parse_offer_json("not json at all", "ecommerce")

    def test_json_array_not_object(self):
        with pytest.raises(OfferParseError, match="must be a JSON object"):
            parse_offer_json('[1, 2, 3]', "ecommerce")

    def test_valid_json_wrong_domain_fields(self):
        with pytest.raises(OfferParseError, match="validation failed"):
            parse_offer_json('{"domain": "ecommerce", "settlement_amount": 50000}', "ecommerce")


class TestParseOfferRegex:
    def test_ecommerce_price_and_quantity(self):
        offer = parse_offer_regex("price is $199.99 and quantity is 500", "ecommerce")
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 199.99
        assert offer.quantity == 500

    def test_settlement_amount(self):
        offer = parse_offer_regex("settlement_amount is 75000 with confidentiality", "settlement")
        assert isinstance(offer, SettlementOffer)
        assert offer.settlement_amount == 75000

    def test_saas_monthly_price_and_seats(self):
        offer = parse_offer_regex(
            "monthly price: $29.99 per seat with seats: 200 for 12 months",
            "saas_procurement",
        )
        assert isinstance(offer, SaaSProcurementOffer)
        assert offer.monthly_price == 29.99
        assert offer.seats == 200

    def test_unparseable_regex_raises(self):
        with pytest.raises(OfferParseError, match="validation failed after regex"):
            parse_offer_regex("no numbers or anything useful here at all", "ecommerce")


class TestParseOffer:
    def test_json_fallback_to_regex(self):
        offer = parse_offer("price is $149.99 and quantity: 250 units", "ecommerce")
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 149.99
        assert offer.quantity == 250

    def test_json_first_malformed(self):
        offer = parse_offer("{broken json} but price is $80.50 quantity 100", "ecommerce")
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 80.50

    def test_settlement_regex(self):
        offer = parse_offer("We propose settlement_amount: $100000 with mutual_release", "settlement")
        assert isinstance(offer, SettlementOffer)
        assert offer.settlement_amount == 100000
