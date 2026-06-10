"""Run a tiny GPT fairness matrix on a fixed cross-domain scenario slice.

This script is intentionally small and reproducible. It compares current Concord
buyer settings against a higher completion-token cap for GPT reasoning models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import yaml

from concord.runners.run_episode import run_episode
from concord.schemas.scenario import Scenario


DEFAULT_SCENARIOS = [
    "concord/outputs/post_remediation_pilot/scenarios/seed-ecommerce-002.yaml",
    "concord/outputs/post_remediation_pilot/scenarios/seed-saas_procurement-021.yaml",
    "concord/outputs/post_remediation_pilot/scenarios/seed-settlement-061.yaml",
    "concord/outputs/post_remediation_pilot/scenarios/seed-ethical_business-023.yaml",
]

DEFAULT_MODELS = [
    "openrouter/openai/gpt-5-nano-2025-08-07",
    "openrouter/openai/gpt-5-mini-2025-08-07",
]

DEFAULT_CONFIGS = [
    {"name": "baseline_1024", "buyer_max_completion_tokens": 1024},
    {"name": "highcap_4096", "buyer_max_completion_tokens": 4096},
]


def _load_scenario(path: Path) -> Scenario:
    with path.open() as f:
        return Scenario.model_validate(yaml.safe_load(f))


def _slug_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _episode_protocol_metrics(episode: dict) -> dict:
    buyer_turns = [
        turn for turn in episode["turns"]
        if turn["agent"] == "buyer"
    ]
    protocol_turns = [
        turn for turn in buyer_turns
        if turn.get("metadata", {}).get("protocol")
    ]
    non_empty = 0
    valid_parse = 0
    requested_offer = 0
    structured_valid = 0
    max_tokens = 0
    for turn in protocol_turns:
        protocol = turn.get("metadata", {}).get("protocol", {})
        if not protocol.get("content_empty", False):
            non_empty += 1
        if protocol.get("action_parse_success"):
            valid_parse += 1
        if protocol.get("requested_offer_action"):
            requested_offer += 1
        if protocol.get("structured_offer_valid"):
            structured_valid += 1
        if protocol.get("max_tokens_reached"):
            max_tokens += 1

    denom = len(protocol_turns) or 1
    return {
        "buyer_turn_count": len(buyer_turns),
        "instrumented_buyer_turn_count": len(protocol_turns),
        "non_empty_buyer_response_rate": non_empty / denom,
        "valid_action_parse_rate": valid_parse / denom,
        "requested_offer_rate": requested_offer / denom,
        "structured_offer_valid_rate": structured_valid / denom,
        "max_tokens_reached_rate": max_tokens / denom,
    }


async def _run_one(
    scenario_path: Path,
    model: str,
    config: dict,
    seller: str,
    seed: int,
    output_dir: Path,
) -> dict:
    scenario = _load_scenario(scenario_path)
    episode = await run_episode(
        scenario,
        buyer_model=model,
        seller_model=seller,
        seed=seed,
        buyer_max_completion_tokens=config["buyer_max_completion_tokens"],
    )
    model_slug = _slug_model(model)
    config_dir = output_dir / config["name"] / model_slug
    config_dir.mkdir(parents=True, exist_ok=True)
    episode_path = config_dir / f"{scenario.id}_{seed}.json"
    payload = episode.model_dump()
    episode_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    protocol = _episode_protocol_metrics(payload)
    return {
        "scenario_id": scenario.id,
        "scenario_path": str(scenario_path),
        "domain": scenario.domain.value,
        "model": model,
        "config": config["name"],
        "buyer_max_completion_tokens": config["buyer_max_completion_tokens"],
        "deal_reached": payload["deal"] is not None,
        "principal_utility": payload["grades"]["principal_utility"],
        "walk_away_correct": payload["grades"]["walk_away_correct"],
        "episode_path": str(episode_path),
        "buyer_runtime": payload["metadata"].get("buyer_runtime", {}),
        "protocol": protocol,
    }


def _aggregate(results: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(row["model"], row["config"])].append(row)

    aggregates = {}
    for key, rows in grouped.items():
        model, config = key
        count = len(rows)
        aggregates[f"{model}::{config}"] = {
            "model": model,
            "config": config,
            "scenario_count": count,
            "deal_rate": sum(1 for row in rows if row["deal_reached"]) / count,
            "principal_utility_mean": sum(row["principal_utility"] for row in rows) / count,
            "walk_away_correct_rate": sum(1 for row in rows if row["walk_away_correct"]) / count,
            "non_empty_buyer_response_rate_mean": sum(
                row["protocol"]["non_empty_buyer_response_rate"] for row in rows
            ) / count,
            "valid_action_parse_rate_mean": sum(
                row["protocol"]["valid_action_parse_rate"] for row in rows
            ) / count,
            "structured_offer_valid_rate_mean": sum(
                row["protocol"]["structured_offer_valid_rate"] for row in rows
            ) / count,
            "max_tokens_reached_rate_mean": sum(
                row["protocol"]["max_tokens_reached_rate"] for row in rows
            ) / count,
        }
    return aggregates


async def _main(args: argparse.Namespace) -> None:
    scenarios = [Path(path) for path in (args.scenarios or DEFAULT_SCENARIOS)]
    models = args.models or DEFAULT_MODELS
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in DEFAULT_CONFIGS:
        for model in models:
            for scenario_path in scenarios:
                results.append(
                    await _run_one(
                        scenario_path=scenario_path,
                        model=model,
                        config=config,
                        seller=args.seller,
                        seed=args.seed,
                        output_dir=output_dir,
                    )
                )

    summary = {
        "seller_model": args.seller,
        "seed": args.seed,
        "scenarios": [str(path) for path in scenarios],
        "models": models,
        "configs": DEFAULT_CONFIGS,
        "results": results,
        "aggregates": _aggregate(results),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny GPT fairness matrix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seller", default="honest_cooperative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--scenarios", nargs="*")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
