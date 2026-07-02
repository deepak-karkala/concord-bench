import asyncio
import json

import pytest

from concord.runners import run_batch as run_batch_module
from concord.runners.budget import DailyBudget
from concord.runners.cache import CacheLLMCalls
from concord.runners.run_batch import BatchRunResult, run_batch
from concord.schemas.episode import EpisodeLog
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
    def test_uses_safer_default_concurrency_for_api_models(self):
        assert run_batch_module._resolve_concurrency_limit("gpt-test", "honest", None) == 2
        assert run_batch_module._resolve_concurrency_limit("greedy", "honest", None) == 10
        assert run_batch_module._resolve_concurrency_limit("gpt-test", "honest", 4) == 4

    def test_runs_multiple_episodes(self, ecom_scenario):
        scenarios = [ecom_scenario.model_copy(update={"id": f"batch-{i}"}) for i in range(3)]
        result = asyncio.run(
            run_batch(scenarios, buyer_model="greedy", seller_model="honest", seeds=[42, 43, 44], concurrency=3)
        )
        assert len(result.episodes) == 9
        for r in result.episodes:
            assert r.scenario_id.startswith("batch-")

    def test_single_episode(self, ecom_scenario):
        result = asyncio.run(
            run_batch([ecom_scenario], buyer_model="greedy", seller_model="honest", seeds=[42], concurrency=1)
        )
        assert len(result.episodes) == 1

    def test_records_buyer_model_in_dead_letter_failures(self, ecom_scenario, temp_dir, monkeypatch):
        captured_kwargs = {}

        async def _fail_run_episode(*args, **kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("simulated timeout")

        monkeypatch.setattr(run_batch_module, "run_episode", _fail_run_episode)
        output_dir = temp_dir / "batch_output"

        result = asyncio.run(
            run_batch(
                [ecom_scenario],
                buyer_model="gpt-test",
                seller_model="honest",
                seeds=[42],
                concurrency=None,
                agent_timeout=300.0,
                output_dir=output_dir,
            )
        )

        assert result.episodes == []
        assert captured_kwargs["agent_timeout"] == 300.0
        dead_letter_path = output_dir / "_artifacts" / "failed_episodes.jsonl"
        entries = [json.loads(line) for line in dead_letter_path.read_text().splitlines()]
        assert entries == [{
            "scenario_id": ecom_scenario.id,
            "seed": 42,
            "buyer_model": "gpt-test",
            "seller_model": "honest",
            "buyer_backend": "api",
            "seller_backend": "scripted",
            "error": "simulated timeout",
            "agent_timeout_seconds": 300.0,
            "concurrency": 2,
            "retry_policy": {"max_retries": 3, "base_delay_seconds": 1.0},
            "model_panel_manifest_path": None,
            "archived_manifest_path": None,
            "run_id": result.run_id,
        }]
        assert str(result.failure_log_path) == str(dead_letter_path)

    def test_archives_manifest_and_run_metadata(self, ecom_scenario, temp_dir, monkeypatch):
        async def _fake_run_episode(*args, **kwargs):
            return EpisodeLog(
                scenario_id=ecom_scenario.id,
                metadata={"cost_usd": 0.1, "seed": kwargs["seed"]},
            )

        monkeypatch.setattr(run_batch_module, "run_episode", _fake_run_episode)

        manifest_path = temp_dir / "phase1_manifest.json"
        manifest_path.write_text('{"panel":"phase1"}\n')
        output_dir = temp_dir / "batch_output"

        result = asyncio.run(
            run_batch(
                [ecom_scenario],
                buyer_model="gpt-test",
                seller_model="honest",
                seeds=[42],
                concurrency=1,
                output_dir=output_dir,
                model_panel_manifest_path=manifest_path,
            )
        )

        assert len(result.episodes) == 1
        archived_manifest = output_dir / "_artifacts" / "phase1_manifest.json"
        assert archived_manifest.exists()
        run_metadata = json.loads((output_dir / "_artifacts" / "gpt-test_run_metadata.json").read_text())
        assert run_metadata["buyer_model"] == "gpt-test"
        assert run_metadata["seller_model"] == "honest"
        assert run_metadata["model_panel_manifest_path"] == str(manifest_path)
        assert run_metadata["archived_manifest_path"] == str(archived_manifest)
        assert run_metadata["run_id"] == result.run_id
        assert run_metadata["input_seed_count"] == 1
        assert run_metadata["expanded_seed_count"] == 1
