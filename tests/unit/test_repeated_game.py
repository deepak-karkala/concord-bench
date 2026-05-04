import pytest

from concord.schemas.scenario import Domain, PrivateContext, Scenario
from concord.synth.repeated_game import generate_repeated_sequence


@pytest.fixture
def base_scenario():
    return Scenario(
        id="rg-test",
        domain=Domain.ECOMMERCE,
        culture="US",
        max_turns=10,
        buyer_context=PrivateContext(
            batna=3000.0,
            reserve_price=8000.0,
            walk_away_threshold=0.5,
            private_info=["budget_10000"],
            hard_constraints=["delivery_14_days"],
        ),
        seller_context=PrivateContext(
            batna=5000.0,
            reserve_price=4000.0,
            walk_away_threshold=None,
            private_info=["cost_2000"],
            hard_constraints=["min_100_units"],
        ),
        deal_schema={"price": "float", "quantity": "int"},
        scenario_description="Base repeated game scenario.",
    )


class TestRepeatedGame:
    def test_generates_correct_number_of_rounds(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        assert len(rounds) == 5

    def test_round_ids_are_unique(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=3)
        ids = [r.id for r in rounds]
        assert len(set(ids)) == 3
        assert ids[0].endswith("-r1")
        assert ids[1].endswith("-r2")
        assert ids[2].endswith("-r3")

    def test_round_1_no_history(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        r1 = rounds[0]
        assert r1.buyer_context.reputation is None
        assert r1.seller_context.reputation is None
        assert r1.buyer_context.relationship_history == []
        assert r1.seller_context.relationship_history == []

    def test_rounds_2_through_5_have_reputation(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        for i in range(1, 5):
            r = rounds[i]
            assert r.buyer_context.reputation is not None
            assert r.seller_context.reputation is not None

    def test_reputation_progresses(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        for i in range(1, 4):
            curr = rounds[i].buyer_context.reputation
            next_rep = rounds[i + 1].buyer_context.reputation
            assert curr != next_rep

    def test_relationship_history_grows(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        for i in range(5):
            expected = max(0, i)
            assert len(rounds[i].buyer_context.relationship_history) == expected

    def test_round_5_endgame_incentives(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        r5 = rounds[4]
        assert any("endgame" in p for p in r5.buyer_context.private_info)
        assert any("endgame" in p for p in r5.seller_context.private_info)
        assert r5.buyer_context.walk_away_threshold > 0.5
        assert r5.buyer_context.reserve_price < 8000.0

    def test_deal_schema_preserved(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        for r in rounds:
            assert r.deal_schema == base_scenario.deal_schema

    def test_batna_preserved(self, base_scenario):
        rounds = generate_repeated_sequence(base_scenario, num_rounds=5)
        for r in rounds:
            assert r.buyer_context.batna == 3000.0
            assert r.seller_context.batna == 5000.0

    def test_original_unchanged(self, base_scenario):
        generate_repeated_sequence(base_scenario, num_rounds=3)
        assert base_scenario.buyer_context.reputation is None
        assert base_scenario.buyer_context.relationship_history == []
