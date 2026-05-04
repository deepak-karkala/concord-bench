import pytest

from concord.graders.constraints import check_hard_constraints, check_walk_away_correctness
from concord.schemas.offer import EcommerceOffer
from concord.schemas.scenario import PrivateContext


class TestCheckHardConstraints:
    def test_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10, shipping_terms="express")
        ctx = PrivateContext(batna=50.0, hard_constraints=["express"])
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_violation_found(self):
        deal = EcommerceOffer(price=100.0, quantity=10, shipping_terms="standard")
        ctx = PrivateContext(batna=50.0, hard_constraints=["express"])
        violations = check_hard_constraints(deal, ctx)
        assert len(violations) == 1

    def test_empty_constraints_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        ctx = PrivateContext(batna=50.0)
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_multiple_constraints(self):
        deal = EcommerceOffer(price=100.0, quantity=10, shipping_terms="express")
        ctx = PrivateContext(batna=50.0, hard_constraints=["express", "missing"])
        violations = check_hard_constraints(deal, ctx)
        assert violations == ["missing"]


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

    def test_no_deal_no_walk_away_correct(self):
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, None, ctx) is True
