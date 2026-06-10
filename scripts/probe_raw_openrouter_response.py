"""Probe a raw OpenRouter chat completion for a single scenario/model pair.

This is a debugging utility for cases where provider-side hidden reasoning,
finish reasons, or visible-content behavior need to be inspected directly.
It is intentionally separate from the benchmark runner so it can capture
raw provider response fields without the normal Concord parsing layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from concord.agents.closed_api_adapter import ClosedAPIAdapter
from concord.env.core import NegotiationEnv
from concord.schemas.scenario import Scenario


def _load_scenario(path: Path) -> Scenario:
    with path.open() as f:
        return Scenario.model_validate(yaml.safe_load(f))


def _extract_reasoning_payload(message) -> dict:
    reasoning = getattr(message, "reasoning", None)
    reasoning_details = getattr(message, "reasoning_details", None)
    payload = {
        "reasoning_text": reasoning,
        "reasoning_text_length": len(reasoning or ""),
    }
    if reasoning_details is not None:
        payload["reasoning_details"] = reasoning_details
    return payload


async def _probe(args: argparse.Namespace) -> dict:
    scenario = _load_scenario(Path(args.scenario))
    env = NegotiationEnv()
    env.reset(scenario, seed=args.seed)

    adapter = ClosedAPIAdapter(
        model_id=args.model,
        timeout=args.timeout,
        system_prompt=args.system_prompt or "",
    )
    system_prompt = adapter.system_prompt
    user_prompt = adapter._build_user_prompt(env.state, scenario.buyer_context)

    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/deepak-karkala/concord-bench",
            "X-Title": "Concord Benchmark",
        },
    )
    try:
        response = await client.chat.completions.create(
            model=args.model.removeprefix("openrouter/"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )
    finally:
        await client.close()

    choice = response.choices[0]
    message = choice.message
    usage = response.usage

    return {
        "scenario_id": scenario.id,
        "model": getattr(response, "model", None),
        "provider": getattr(response, "provider", None),
        "max_completion_tokens": args.max_completion_tokens,
        "finish_reason": getattr(choice, "finish_reason", None),
        "message_content": message.content,
        "message_content_is_none": message.content is None,
        "message_dump": message.model_dump() if hasattr(message, "model_dump") else str(message),
        "reasoning": _extract_reasoning_payload(message),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else str(usage),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a raw OpenRouter response")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--system-prompt", help="Optional override system prompt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = asyncio.run(_probe(args))
    rendered = json.dumps(payload, indent=2, default=str)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
