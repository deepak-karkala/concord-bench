import asyncio
import json
from pathlib import Path

from concord.runners.budget import DailyBudget
from concord.runners.run_episode import run_episode
from concord.schemas.episode import EpisodeLog
from concord.schemas.scenario import Scenario

DEAD_LETTER_DIR = Path("outputs/dead_letter")

ESTIMATED_COST_PER_EPISODE = 0.40


async def run_batch(
    scenarios: list[Scenario],
    buyer_model: str = "greedy",
    seller_model: str = "greedy",
    seeds: list[int] | None = None,
    concurrency: int = 10,
    budget_cap: float | None = None,
    stance: str = "default",
) -> list[EpisodeLog]:
    if seeds is None:
        seeds = [42]
    if len(seeds) < len(scenarios):
        seeds = seeds * ((len(scenarios) // len(seeds)) + 1)

    budget = DailyBudget(daily_limit=budget_cap or float("inf"))
    semaphore = asyncio.Semaphore(concurrency)
    results: list[EpisodeLog] = []
    failures: list[dict] = []

    async def _run_one(scenario: Scenario, seed: int) -> None:
        async with semaphore:
            if not budget.can_spend(ESTIMATED_COST_PER_EPISODE):
                failures.append({
                    "scenario_id": scenario.id,
                    "seed": seed,
                    "error": "daily budget cap reached",
                })
                return

            try:
                episode = await run_episode(scenario, buyer_model=buyer_model, seller_model=seller_model, seed=seed, stance=stance)
                actual_cost = episode.metadata.get("cost_usd", ESTIMATED_COST_PER_EPISODE)
                budget.record_spend(actual_cost)
                results.append(episode)
            except Exception as e:
                failures.append({
                    "scenario_id": scenario.id,
                    "seed": seed,
                    "error": str(e),
                })

    tasks = [
        _run_one(scenario, seed)
        for scenario, seed in zip(scenarios, seeds)
    ]
    await asyncio.gather(*tasks)

    if failures:
        DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEAD_LETTER_DIR / "failed_episodes.jsonl", "a") as f:
            for failure in failures:
                f.write(json.dumps(failure) + "\n")

    return results
