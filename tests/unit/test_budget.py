from concord.runners.budget import DailyBudget


class TestDailyBudgetUnit:
    def test_initial_remaining(self):
        budget = DailyBudget(daily_limit=200.0)
        assert budget.remaining() == 200.0

    def test_exact_limit_boundary(self):
        budget = DailyBudget(daily_limit=100.0)
        assert budget.can_spend(100.0) is True
        budget.spend(100.0)
        assert budget.can_spend(0.01) is False

    def test_reset_on_new_day(self):
        budget = DailyBudget(daily_limit=10.0)
        budget.spend(8.0)
        budget._day_key = "2000-01-01"
        assert budget.remaining() == 10.0
