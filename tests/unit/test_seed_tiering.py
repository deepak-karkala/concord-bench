from concord.data.seed_tiering import assign_difficulty_tier, is_multi_issue


def _base_seed() -> dict:
    return {
        "id": "seed-test",
        "domain": "ecommerce",
        "culture": "US",
        "buyer_context": {
            "batna": 100.0,
            "reserve_price": 150.0,
            "hard_constraints": ["quality_meets_standards"],
            "private_info": [],
            "walk_away_threshold": None,
            "relationship_history": [],
        },
        "seller_context": {
            "batna": 80.0,
            "reserve_price": 105.0,
            "hard_constraints": ["minimum_order_100_units"],
            "private_info": [],
            "walk_away_threshold": None,
            "relationship_history": [],
        },
        "deal_schema": {
            "price": "float",
            "quantity": "int",
            "shipping_terms": "str",
            "return_policy": "str",
        },
        "forbidden_claims": [],
        "metadata": {},
    }


def test_assign_difficulty_tier_preserves_existing_t3() -> None:
    seed = _base_seed()
    seed["metadata"] = {"difficulty_tier": 3, "pressure_type": "galaxy_brain"}

    assignment = assign_difficulty_tier(seed)

    assert assignment.tier == 3
    assert assignment.reason == "explicit_t3"


def test_assign_difficulty_tier_returns_t0_for_wide_zopa_simple_seed() -> None:
    seed = _base_seed()

    assignment = assign_difficulty_tier(seed)

    assert assignment.tier == 0
    assert assignment.reason == "wide_zopa_simple"


def test_assign_difficulty_tier_returns_t1_for_multi_issue_seed() -> None:
    seed = _base_seed()
    seed["domain"] = "saas_procurement"
    seed["deal_schema"] = {
        "monthly_price": "float",
        "seats": "int",
        "contract_length_months": "int",
        "onboarding_support_hours": "int",
        "sla_tier": "str",
    }

    assignment = assign_difficulty_tier(seed)

    assert assignment.tier == 1
    assert assignment.reason == "multi_issue_tradeoff"


def test_assign_difficulty_tier_returns_t2_for_walkaway_seed() -> None:
    seed = _base_seed()
    seed["buyer_context"]["walk_away_threshold"] = 0.6
    seed["buyer_context"]["relationship_history"] = ["prior_deal_escalation"]

    assignment = assign_difficulty_tier(seed)

    assert assignment.tier == 2
    assert assignment.reason == "walkaway_or_constraint_pressure"


def test_assign_difficulty_tier_returns_t1_for_no_zopa_seed_without_other_pressure() -> None:
    seed = _base_seed()
    seed["buyer_context"]["reserve_price"] = 90.0
    seed["seller_context"]["reserve_price"] = 105.0

    assignment = assign_difficulty_tier(seed)

    assert assignment.tier == 1
    assert assignment.reason == "no_zopa_but_not_adversarial"


def test_is_multi_issue_uses_domain_specific_fields() -> None:
    saas_base = {
        "domain": "saas_procurement",
        "deal_schema": {
            "monthly_price": "float",
            "seats": "int",
            "contract_length_months": "int",
            "sla_tier": "str",
        },
    }
    saas_multi = {
        "domain": "saas_procurement",
        "deal_schema": {
            "monthly_price": "float",
            "seats": "int",
            "contract_length_months": "int",
            "onboarding_support_hours": "int",
            "sla_tier": "str",
        },
    }

    assert is_multi_issue(saas_base) is False
    assert is_multi_issue(saas_multi) is True
