import asyncio

import pytest

from concord.runners.budget import DailyBudget
from concord.runners.cache import CacheLLMCalls
from concord.runners.run_batch import run_batch
from concord.schemas.scenario import Domain, PrivateContext, Scenario


@pytest.fixture
def ecom_scenario():
    return Scenario(
        id="cache-batch-test",
        domain=Domain.ECOMMERCE,
        buyer_context=PrivateContext(batna=3000),
        seller_context=PrivateContext(batna=5000),
        deal_schema={"price": "float", "quantity": "int"},
    )


class TestCacheLLMCalls:
    def test_put_and_get(self):
        cache = CacheLLMCalls()
        cache.put("gpt-5.2", "hash-abc", 0.7, 42, {"content": "hello", "prompt_tokens": 10, "completion_tokens": 5})
        result = cache.get("gpt-5.2", "hash-abc", 0.7, 42)
        assert result is not None
        assert result["content"] == "hello"
        cache.close()

    def test_cache_miss(self):
        cache = CacheLLMCalls()
        result = cache.get("gpt-5.2", "nonexistent-hash", 0.7, 42)
        assert result is None
        cache.close()

    def test_different_temperature_different_cache(self):
        cache = CacheLLMCalls()
        cache.put("gpt-5.2", "hash-t", 0.5, 42, {"content": "cold"})
        result = cache.get("gpt-5.2", "hash-t", 0.7, 42)
        assert result is None
        cache.close()

    def test_different_seed_different_cache(self):
        cache = CacheLLMCalls()
        cache.put("gpt-5.2", "hash-s", 0.7, 1, {"content": "seed1"})
        result = cache.get("gpt-5.2", "hash-s", 0.7, 2)
        assert result is None
        cache.close()


class TestDailyBudget:
    def test_can_spend_within_limit(self):
        budget = DailyBudget(daily_limit=50.0)
        assert budget.can_spend(30.0) is True

    def test_cannot_spend_exceeds_limit(self):
        budget = DailyBudget(daily_limit=50.0)
        budget.spend(40.0)
        assert budget.can_spend(20.0) is False

    def test_remaining_decreases(self):
        budget = DailyBudget(daily_limit=100.0)
        budget.spend(30.0)
        assert budget.remaining() == 70.0

    def test_spend_updates_remaining(self):
        budget = DailyBudget(daily_limit=50.0)
        budget.spend(10.0)
        budget.spend(15.0)
        assert budget.remaining() == 25.0


class TestRunBatch:
    def test_runs_multiple_episodes(self, ecom_scenario):
        scenarios = [ecom_scenario.model_copy(update={"id": f"batch-{i}"}) for i in range(3)]
        results = asyncio.run(
            run_batch(scenarios, buyer_model="greedy", seller_model="honest", seeds=[42, 43, 44], concurrency=3)
        )
        assert len(results) == 3
        for r in results:
            assert r.scenario_id.startswith("batch-")

    def test_single_episode(self, ecom_scenario):
        results = asyncio.run(
            run_batch([ecom_scenario], buyer_model="greedy", seller_model="honest", seeds=[42], concurrency=1)
        )
        assert len(results) == 1
