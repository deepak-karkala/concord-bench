import pytest

from concord.graders.utility import (
    compute_joint_welfare,
    compute_pareto_efficiency,
    compute_principal_utility,
)
from concord.schemas.offer import EcommerceOffer, SaaSProcurementOffer, SettlementOffer
from concord.schemas.scenario import PrivateContext


class TestComputePrincipalUtility:
    def test_utility_above_batna(self):
        deal = EcommerceOffer(price=5000.0, quantity=100)
        ctx = PrivateContext(batna=3000.0)
        utility = compute_principal_utility(deal, ctx)
        assert 0 < utility < 1

    def test_utility_below_batna_zero(self):
        deal = EcommerceOffer(price=2000.0, quantity=100)
        ctx = PrivateContext(batna=3000.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility == 0.0

    def test_utility_exceeds_reserve_price_zero(self):
        deal = EcommerceOffer(price=9000.0, quantity=100)
        ctx = PrivateContext(batna=3000.0, reserve_price=8000.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility == 0.0

    def test_utility_capped_at_one(self):
        deal = EcommerceOffer(price=10000.0, quantity=100)
        ctx = PrivateContext(batna=100.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility == 1.0

    def test_utility_settlement(self):
        deal = SettlementOffer(settlement_amount=75000.0)
        ctx = PrivateContext(batna=50000.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility == pytest.approx(0.5)

    def test_utility_saas_monthly(self):
        deal = SaaSProcurementOffer(monthly_price=40.0, seats=100, contract_length_months=12)
        ctx = PrivateContext(batna=30.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility > 0

    def test_batna_zero_handled(self):
        deal = EcommerceOffer(price=100.0, quantity=1)
        ctx = PrivateContext(batna=0.0)
        utility = compute_principal_utility(deal, ctx)
        assert utility >= 0.0


class TestJointWelfare:
    def test_equal_utilities(self):
        assert compute_joint_welfare(0.5, 0.5) == 0.5

    def test_unequal_utilities(self):
        assert compute_joint_welfare(0.8, 0.2) == 0.5

    def test_zero_both(self):
        assert compute_joint_welfare(0.0, 0.0) == 0.0


class TestParetoEfficiency:
    def test_no_possible_deals(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        assert compute_pareto_efficiency(deal, []) is True

    def test_pareto_efficient(self):
        deal = EcommerceOffer(price=500.0, quantity=100)
        others = [
            EcommerceOffer(price=400.0, quantity=100),
            EcommerceOffer(price=300.0, quantity=100),
        ]
        assert compute_pareto_efficiency(deal, others) is True

    def test_not_pareto_efficient(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        others = [
            EcommerceOffer(price=200.0, quantity=10),
            EcommerceOffer(price=50.0, quantity=10),
        ]
        assert compute_pareto_efficiency(deal, others) is False
