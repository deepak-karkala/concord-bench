import pytest
from pydantic import TypeAdapter, ValidationError

from concord.schemas.culture import CULTURAL_PROFILES, CulturalProfile, Culture
from concord.schemas.episode import ActionType, ConfidenceInterval, DimensionScore, EpisodeLog, GradeReport, ModelCard, Turn
from concord.schemas.offer import (
    EcommerceOffer,
    EthicalBusinessOffer,
    Offer,
    SaaSProcurementOffer,
    SettlementOffer,
)
from concord.schemas.scenario import Domain, PrivateContext, Scenario


class TestCulture:
    def test_culture_enum_values(self):
        assert len(Culture) == 5
        assert Culture.US == "US"
        assert Culture.JP == "JP"
        assert Culture.IN == "IN"
        assert Culture.BR == "BR"
        assert Culture.MENA == "MENA"

    def test_all_profiles_defined(self):
        for culture in Culture:
            assert culture in CULTURAL_PROFILES, f"Missing profile for {culture}"

    def test_cultural_profile_valid(self):
        profile = CulturalProfile(
            communication_style="direct",
            power_distance=40,
            individualism=91,
            uncertainty_avoidance=46,
            long_term_orientation=26,
            indulgence=68,
            negotiation_norms=["Get to the point"],
            acceptable_tactics=["Data-driven"],
            taboo_tactics=["Bribes"],
        )
        assert profile.communication_style == "direct"
        assert 0 <= profile.power_distance <= 100

    def test_cultural_profile_invalid_range(self):
        with pytest.raises(ValidationError):
            CulturalProfile(
                communication_style="direct",
                power_distance=150,
                individualism=50,
                uncertainty_avoidance=50,
                long_term_orientation=50,
                indulgence=50,
            )

    def test_round_trip(self):
        profile = CULTURAL_PROFILES[Culture.US]
        data = profile.model_dump()
        restored = CulturalProfile.model_validate(data)
        assert restored == profile


class TestScenario:
    def test_domain_enum(self):
        assert len(Domain) == 4
        assert Domain.ECOMMERCE == "ecommerce"
        assert Domain.SAAS_PROCUREMENT == "saas_procurement"
        assert Domain.SETTLEMENT == "settlement"
        assert Domain.ETHICAL_BUSINESS == "ethical_business"

    def test_minimal_scenario(self):
        s = Scenario(
            id="test-001",
            domain=Domain.ECOMMERCE,
            culture="US",
            buyer_context=PrivateContext(batna=100),
            seller_context=PrivateContext(batna=200),
            deal_schema={"price": "float", "quantity": "int"},
        )
        assert s.id == "test-001"
        assert s.max_turns == 10
        assert s.culture == "US"

    def test_scenario_defaults(self):
        s = Scenario(
            id="test-002",
            domain=Domain.ECOMMERCE,
            buyer_context=PrivateContext(batna=50),
            seller_context=PrivateContext(batna=60),
            deal_schema={},
        )
        assert s.culture == "US"
        assert s.max_turns == 10
        assert s.forbidden_claims == []
        assert s.scenario_description == ""

    def test_scenario_validation_error(self):
        with pytest.raises(ValidationError):
            Scenario(
                domain=Domain.ECOMMERCE,
                buyer_context=PrivateContext(batna=100),
                seller_context=PrivateContext(batna=200),
                deal_schema={},
            )

    def test_private_context_full(self):
        ctx = PrivateContext(
            batna=5000.0,
            reserve_price=7500.0,
            hard_constraints=["no_nda_breach"],
            private_info=["budget_is_12000"],
            walk_away_threshold=0.6,
            reputation=0.85,
            relationship_history=["prior_deal_q3_2025"],
        )
        assert ctx.batna == 5000.0
        assert ctx.reserve_price == 7500.0
        assert "budget_is_12000" in ctx.private_info

    def test_round_trip(self):
        s = Scenario(
            id="round-trip-test",
            domain=Domain.SETTLEMENT,
            culture="JP",
            max_turns=8,
            buyer_context=PrivateContext(
                batna=10000,
                reserve_price=5000,
                hard_constraints=["confidentiality"],
                private_info=["evidence_X"],
            ),
            seller_context=PrivateContext(
                batna=0,
                reserve_price=20000,
            ),
            deal_schema={"settlement_amount": "float", "confidentiality_clause": "bool"},
            forbidden_claims=["cannot_claim_no_evidence"],
            scenario_description="Patent infringement settlement negotiation",
        )
        data = s.model_dump()
        restored = Scenario.model_validate(data)
        assert restored == s


class TestOffer:
    def _validate_offer(self, data: dict) -> Offer:
        return TypeAdapter(Offer).validate_python(data)

    def test_ecommerce_offer(self):
        data = {"domain": "ecommerce", "price": 150.0, "quantity": 100, "shipping_terms": "express"}
        offer = self._validate_offer(data)
        assert isinstance(offer, EcommerceOffer)
        assert offer.price == 150.0
        assert offer.quantity == 100

    def test_saas_offer(self):
        data = {
            "domain": "saas_procurement",
            "monthly_price": 29.99,
            "seats": 50,
            "contract_length_months": 12,
        }
        offer = self._validate_offer(data)
        assert isinstance(offer, SaaSProcurementOffer)
        assert offer.monthly_price == 29.99

    def test_settlement_offer(self):
        data = {
            "domain": "settlement",
            "settlement_amount": 50000.0,
            "payment_terms": "structured",
            "confidentiality_clause": True,
        }
        offer = self._validate_offer(data)
        assert isinstance(offer, SettlementOffer)
        assert offer.settlement_amount == 50000.0
        assert offer.confidentiality_clause is True

    def test_ethical_business_offer(self):
        data = {
            "domain": "ethical_business",
            "price": 25000.0,
            "environmental_commitments": ["carbon_neutral_by_2030"],
            "transparency_reports": True,
        }
        offer = self._validate_offer(data)
        assert isinstance(offer, EthicalBusinessOffer)
        assert "carbon_neutral_by_2030" in offer.environmental_commitments

    def test_missing_domain_raises(self):
        with pytest.raises(ValidationError):
            self._validate_offer({"price": 100, "quantity": 10})

    def test_unknown_domain_raises(self):
        with pytest.raises(ValidationError):
            self._validate_offer({"domain": "unknown", "price": 100})

    def test_round_trip_ecommerce(self):
        data = {"domain": "ecommerce", "price": 200.0, "quantity": 500, "shipping_terms": "standard"}
        offer = self._validate_offer(data)
        restored = TypeAdapter(Offer).validate_python(offer.model_dump())
        assert restored == offer

    def test_round_trip_settlement(self):
        data = {
            "domain": "settlement",
            "settlement_amount": 75000.0,
            "payment_terms": "lump_sum",
            "confidentiality_clause": True,
            "non_disparagement": True,
        }
        offer = self._validate_offer(data)
        restored = TypeAdapter(Offer).validate_python(offer.model_dump())
        assert restored == offer


class TestEpisode:
    def test_turn_creation(self):
        t = Turn(agent="buyer", action_type=ActionType.MESSAGE, content="Let's negotiate")
        assert t.agent == "buyer"
        assert t.action_type == ActionType.MESSAGE
        assert t.content == "Let's negotiate"
        assert t.offer is None
        assert t.timestamp is not None

    def test_turn_with_offer(self):
        offer = EcommerceOffer(price=150.0, quantity=100)
        t = Turn(agent="seller", action_type=ActionType.OFFER, content="Here is my offer", offer=offer)
        assert t.action_type == ActionType.OFFER
        assert isinstance(t.offer, EcommerceOffer)
        assert t.offer.price == 150.0

    def test_episode_log_creation(self):
        t1 = Turn(agent="buyer", action_type=ActionType.MESSAGE, content="Hello")
        ep = EpisodeLog(scenario_id="test-1", turns=[t1])
        assert ep.scenario_id == "test-1"
        assert len(ep.turns) == 1
        assert ep.deal is None
        assert ep.terminal is False

    def test_episode_terminal_accept(self):
        offer = EcommerceOffer(price=99.99, quantity=10)
        t1 = Turn(agent="buyer", action_type=ActionType.OFFER, content="Offer", offer=offer)
        t2 = Turn(agent="seller", action_type=ActionType.ACCEPT, content="Accepted")
        ep = EpisodeLog(scenario_id="test-2", turns=[t1, t2], deal=offer)
        assert ep.terminal is True

    def test_episode_terminal_walk_away(self):
        t1 = Turn(agent="buyer", action_type=ActionType.WALK_AWAY, content="No deal")
        ep = EpisodeLog(scenario_id="test-3", turns=[t1])
        assert ep.terminal is True
        assert ep.deal is None

    def test_buyer_seller_turns(self):
        t1 = Turn(agent="buyer", action_type=ActionType.MESSAGE, content="Hi")
        t2 = Turn(agent="seller", action_type=ActionType.MESSAGE, content="Hello")
        ep = EpisodeLog(scenario_id="test-4", turns=[t1, t2])
        assert len(ep.buyer_turns) == 1
        assert len(ep.seller_turns) == 1

    def test_round_trip(self):
        t = Turn(agent="buyer", action_type=ActionType.MESSAGE, content="test")
        ep = EpisodeLog(
            scenario_id="round-trip-test",
            turns=[t],
            metadata={"model_id": "test-model", "seed": 42},
        )
        data = ep.model_dump()
        restored = EpisodeLog.model_validate(data)
        assert restored.scenario_id == ep.scenario_id
        assert restored.metadata["seed"] == 42

    def test_model_card(self):
        mc = ModelCard(
            model_id="claude-opus-4-7",
            concord_version="0.1.0",
            outcome={
                "principal_utility": DimensionScore(mean=0.72, n_episodes=3000),
                "joint_welfare": DimensionScore(mean=0.65, n_episodes=3000),
            },
            constraints={
                "hard_constraint_violations": DimensionScore(mean=0.03, n_episodes=3000),
            },
            total_episodes=3000,
        )
        assert mc.model_id == "claude-opus-4-7"
        assert mc.outcome["principal_utility"].mean == 0.72
        assert mc.total_episodes == 3000

    def test_confidence_interval(self):
        ci = ConfidenceInterval(lower=0.68, upper=0.76)
        assert ci.lower == 0.68
        assert ci.upper == 0.76
        assert ci.confidence == 0.95

    def test_episode_round_trip_full(self):
        offer = SettlementOffer(settlement_amount=50000, payment_terms="lump_sum")
        t1 = Turn(agent="buyer", action_type=ActionType.MESSAGE, content="Hi")
        t2 = Turn(agent="seller", action_type=ActionType.OFFER, content="Offer", offer=offer)
        t3 = Turn(agent="buyer", action_type=ActionType.ACCEPT, content="Deal", offer=offer)
        ep = EpisodeLog(
            scenario_id="full-test",
            turns=[t1, t2, t3],
            deal=offer,
            grades=GradeReport(principal_utility=0.8, walk_away_correct=True),
            metadata={"model_id": "claude-opus-4-7", "seed": 42, "prompt_hash": "abc123"},
        )
        data = ep.model_dump()
        restored = EpisodeLog.model_validate(data)
        assert restored.deal == offer
        assert restored.grades.principal_utility == 0.8
        assert restored.metadata["prompt_hash"] == "abc123"
