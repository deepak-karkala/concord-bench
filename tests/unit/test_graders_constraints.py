from concord.graders.constraints import check_hard_constraints, check_walk_away_correctness
from concord.schemas.offer import EcommerceOffer
from concord.schemas.scenario import PrivateContext


class TestCheckHardConstraints:
    def test_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10, shipping_terms="express")
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_5_units"])
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_violation_found(self):
        deal = EcommerceOffer(price=100.0, quantity=5)
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_10_units"])
        violations = check_hard_constraints(deal, ctx)
        assert len(violations) == 1

    def test_empty_constraints_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        ctx = PrivateContext(batna=50.0)
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_multiple_constraints(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_10_units", "minimum_order_50_units"])
        violations = check_hard_constraints(deal, ctx)
        assert violations == ["minimum_order_50_units"]


class TestWalkAwayCorrectness:
    def test_walk_away_with_threshold_correct(self):
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx) is True

    def test_walk_away_without_threshold_incorrect(self):
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(True, None, ctx) is False

    def test_no_walk_away_no_threshold_correct(self):
        deal = EcommerceOffer(price=200.0, quantity=10)
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_good_deal_above_threshold_correct(self):
        deal = EcommerceOffer(price=500.0, quantity=10)
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.3)
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_bad_deal_below_threshold_incorrect(self):
        deal = EcommerceOffer(price=101.0, quantity=10)
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, deal, ctx) is False

    def test_seller_uses_role_aware_utility_for_threshold(self):
        deal = EcommerceOffer(price=120.0, quantity=10)
        ctx = PrivateContext(
            batna=100.0,
            reserve_price=80.0,
            walk_away_threshold=0.5,
        )
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_no_deal_no_walk_away_correct(self):
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, None, ctx) is True

    def test_silent_buyer_walk_away_is_protocol_failure(self):
        # silent/non-engaging buyer should never be correct walk-away
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx, buyer_engaged=False) is False

    def test_silent_buyer_no_deal_is_protocol_failure(self):
        # silent buyer with no deal should not be scored as correct
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, None, ctx, buyer_engaged=False) is False

    def test_no_zopa_walk_away_is_correct(self):
        # if ZOPA doesn't exist, walk-away is always correct
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx, zopa_exists=False) is True

    def test_walk_away_without_threshold_or_zopa_info_is_incorrect(self):
        # walking away with no threshold and no ZOPA info = can't confirm correct
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(True, None, ctx) is False
