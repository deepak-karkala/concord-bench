import asyncio
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from concord.runners.budget import DailyBudget
from concord.runners.run_episode import _is_scripted_model, _effective_agent_timeout, run_episode
from concord.schemas.episode import EpisodeLog
from concord.schemas.scenario import Scenario

DEAD_LETTER_DIR = Path("outputs/dead_letter")

ESTIMATED_COST_PER_EPISODE = 0.40


@dataclass
class BatchRunResult:
    episodes: list[EpisodeLog]
    failures: list[dict]
    run_id: str
    failure_log_path: Path | None
    run_metadata_path: Path | None


def _slug_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _resolve_concurrency_limit(
    buyer_model: str,
    seller_model: str,
    concurrency: int | None,
) -> int:
    if concurrency is not None:
        return concurrency

    if _is_scripted_model(buyer_model) and _is_scripted_model(seller_model):
        return 10

    return 2


async def run_batch(
    scenarios: list[Scenario],
    buyer_model: str = "greedy",
    seller_model: str = "greedy",
    seeds: list[int] | None = None,
    concurrency: int | None = None,
    budget_cap: float | None = None,
    stance: str = "default",
    agent_timeout: float | None = None,
    output_dir: Path | None = None,
    model_panel_manifest_path: Path | None = None,
    dead_letter_path: Path | None = None,
) -> BatchRunResult:
    if seeds is None:
        seeds = [42]
    input_seed_count = len(seeds)
    scenario_seed_pairs = [(scenario, seed) for scenario in scenarios for seed in seeds]
    expanded_seed_count = len(scenario_seed_pairs)

    budget = DailyBudget(daily_limit=budget_cap or float("inf"))
    effective_concurrency = _resolve_concurrency_limit(
        buyer_model=buyer_model,
        seller_model=seller_model,
        concurrency=concurrency,
    )
    effective_timeout = _effective_agent_timeout(buyer_model, agent_timeout)
    semaphore = asyncio.Semaphore(effective_concurrency)
    results: list[EpisodeLog] = []
    failures: list[dict] = []
    archived_manifest_path: str | None = None
    failure_log_output_path: Path | None = None
    run_metadata_path: Path | None = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / "_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        failure_log_output_path = dead_letter_path or (artifacts_dir / "failed_episodes.jsonl")
        if model_panel_manifest_path is not None:
            target_path = artifacts_dir / model_panel_manifest_path.name
            shutil.copyfile(model_panel_manifest_path, target_path)
            archived_manifest_path = str(target_path)
        run_metadata_path = artifacts_dir / f"{_slug_model_id(buyer_model)}_run_metadata.json"
    elif dead_letter_path is not None:
        failure_log_output_path = dead_letter_path
    else:
        failure_log_output_path = DEAD_LETTER_DIR / "failed_episodes.jsonl"

    async def _run_one(scenario: Scenario, seed: int) -> None:
        async with semaphore:
            if not budget.can_spend(ESTIMATED_COST_PER_EPISODE):
                failures.append({
                    "scenario_id": scenario.id,
                    "seed": seed,
                    "buyer_model": buyer_model,
                    "seller_model": seller_model,
                    "buyer_backend": "scripted" if _is_scripted_model(buyer_model) else "api",
                    "seller_backend": "scripted" if _is_scripted_model(seller_model) else "api",
                    "error": "daily budget cap reached",
                    "agent_timeout_seconds": effective_timeout,
                    "concurrency": effective_concurrency,
                    "retry_policy": {"max_retries": 3, "base_delay_seconds": 1.0},
                    "model_panel_manifest_path": str(model_panel_manifest_path) if model_panel_manifest_path else None,
                    "archived_manifest_path": archived_manifest_path,
                    "run_id": run_id,
                })
                return

            try:
                episode = await run_episode(
                    scenario,
                    buyer_model=buyer_model,
                    seller_model=seller_model,
                    seed=seed,
                    stance=stance,
                    agent_timeout=agent_timeout,
                    model_panel_manifest_path=str(model_panel_manifest_path) if model_panel_manifest_path else None,
                )
                actual_cost = episode.metadata.get("cost_usd", ESTIMATED_COST_PER_EPISODE)
                budget.record_spend(actual_cost)
                results.append(episode)
            except Exception as e:
                failures.append({
                    "scenario_id": scenario.id,
                    "seed": seed,
                    "buyer_model": buyer_model,
                    "seller_model": seller_model,
                    "buyer_backend": "scripted" if _is_scripted_model(buyer_model) else "api",
                    "seller_backend": "scripted" if _is_scripted_model(seller_model) else "api",
                    "error": str(e),
                    "agent_timeout_seconds": effective_timeout,
                    "concurrency": effective_concurrency,
                    "retry_policy": {"max_retries": 3, "base_delay_seconds": 1.0},
                    "model_panel_manifest_path": str(model_panel_manifest_path) if model_panel_manifest_path else None,
                    "archived_manifest_path": archived_manifest_path,
                    "run_id": run_id,
                })

    tasks = [_run_one(scenario, seed) for scenario, seed in scenario_seed_pairs]
    await asyncio.gather(*tasks)

    if failures and failure_log_output_path is not None:
        failure_log_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log_output_path, "a") as f:
            for failure in failures:
                f.write(json.dumps(failure) + "\n")

    if run_metadata_path is not None:
        run_metadata = {
            "run_id": run_id,
            "buyer_model": buyer_model,
            "seller_model": seller_model,
            "buyer_backend": "scripted" if _is_scripted_model(buyer_model) else "api",
            "seller_backend": "scripted" if _is_scripted_model(seller_model) else "api",
            "stance": stance,
            "input_seed_count": input_seed_count,
            "expanded_seed_count": expanded_seed_count,
            "repeated_runs_per_scenario": len(seeds),
            "scenario_count": len(scenarios),
            "completed_episode_count": len(results),
            "final_failure_count": len(failures),
            "concurrency": effective_concurrency,
            "agent_timeout_seconds": effective_timeout,
            "retry_policy": {"max_retries": 3, "base_delay_seconds": 1.0},
            "budget_cap": budget_cap,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_panel_manifest_path": str(model_panel_manifest_path) if model_panel_manifest_path else None,
            "archived_manifest_path": archived_manifest_path,
            "failure_log_path": str(failure_log_output_path) if failure_log_output_path else None,
        }
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n")

    return BatchRunResult(
        episodes=results,
        failures=failures,
        run_id=run_id,
        failure_log_path=failure_log_output_path,
        run_metadata_path=run_metadata_path,
    )
