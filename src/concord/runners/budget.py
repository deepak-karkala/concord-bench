import time


class DailyBudget:
    def __init__(self, daily_limit: float = 100.0):
        self.daily_limit = daily_limit
        self.spent_today: float = 0.0
        self._day_key = self._current_day()

    @staticmethod
    def _current_day() -> str:
        return time.strftime("%Y-%m-%d")

    def _reset_if_new_day(self):
        today = self._current_day()
        if today != self._day_key:
            self.spent_today = 0.0
            self._day_key = today

    def can_spend(self, amount: float) -> bool:
        self._reset_if_new_day()
        return (self.spent_today + amount) <= self.daily_limit

    def spend(self, amount: float) -> None:
        self._reset_if_new_day()
        self.spent_today += amount

    def remaining(self) -> float:
        self._reset_if_new_day()
        return max(0.0, self.daily_limit - self.spent_today)
