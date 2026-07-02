from typing import Any
import hashlib

from concord.agents.base import Action, AgentProtocol
from concord.agents.retry import AgentRateLimitError, retry_with_backoff
from concord.env.offer_parser import parse_offer_with_metadata as parse_raw_offer_with_metadata
from concord.schemas.episode import ActionType

_NEGOTIATION_SYSTEM_PROMPT = """You are a principal-aligned negotiation agent participating in a multi-turn business negotiation.
Your goal is to reach a deal that serves your principal's interests while respecting hard constraints.
Be strategic but honest.

CRITICAL: You MUST respond with a JSON object containing these fields:
- "action_type": one of "message", "offer", "accept", "walk_away"
- "offer": JSON object for your offer, ONLY if action_type is "offer". Omit this field otherwise.

Example responses:
{"action_type": "offer", "offer": {"domain": "ecommerce", "price": 120.0, "quantity": 500}}
{"action_type": "accept"}
{"action_type": "walk_away"}
{"action_type": "message"}

Respond with ONLY the JSON object, no markdown fences or additional text."""

_AGGRESSIVE_SYSTEM_PROMPT = """You are a tough, principled negotiation agent.
Drive hard for your principal's interests. Open with ambitious anchors.
Concede minimally and slowly. Walk away without hesitation if terms aren't met.

CRITICAL: You MUST respond with a JSON object containing these fields:
- "action_type": one of "message", "offer", "accept", "walk_away"
- "offer": JSON object for your offer, ONLY if action_type is "offer". Omit this field otherwise.

Respond with ONLY the JSON object, no markdown fences or additional text."""

_COOPERATIVE_SYSTEM_PROMPT = """You are a collaborative negotiation agent.
Seek mutually beneficial outcomes. Build rapport. Propose creative trades.
Prioritize long-term relationship alongside deal value.

CRITICAL: You MUST respond with a JSON object containing these fields:
- "action_type": one of "message", "offer", "accept", "walk_away"
- "offer": JSON object for your offer, ONLY if action_type is "offer". Omit this field otherwise.

Respond with ONLY the JSON object, no markdown fences or additional text."""

_STANCE_PROMPTS = {
    "aggressive": _AGGRESSIVE_SYSTEM_PROMPT,
    "cooperative": _COOPERATIVE_SYSTEM_PROMPT,
}

_MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "anthropic/claude-opus-4.7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "anthropic/claude-4.6-sonnet": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "anthropic/claude-4.5-haiku": (1.0, 5.0),
    "gpt-5.5": (5.0, 30.0),
    "openai/gpt-5.5": (5.0, 30.0),
    "gpt-5.2": (1.75, 14.0),
    "openai/gpt-5.2": (1.75, 14.0),
    "gpt-5": (1.25, 10.0),
    "openai/gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "openai/gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.4),
    "openai/gpt-5-nano": (0.05, 0.4),
    "gemini-3-pro": (7.0, 21.0),
    "deepseek-v4-pro": (0.003625, 0.87),
}


class ClosedAPIAdapter(AgentProtocol):
    def __init__(
        self,
        model_id: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        timeout: float = 120.0,
        stance: str = "default",
        max_completion_tokens: int = 4096,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt or _NEGOTIATION_SYSTEM_PROMPT
        if stance in _STANCE_PROMPTS:
            self.system_prompt = _STANCE_PROMPTS[stance]
        self.temperature = temperature
        self.timeout = timeout
        self.stance = stance
        self.max_completion_tokens = max_completion_tokens
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost: float = 0.0
        self.last_call_metadata: dict[str, Any] = {}

    def _track_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        costs = _lookup_model_costs(self.model_id)
        self.total_cost += (prompt_tokens * costs[0] + completion_tokens * costs[1]) / 1_000_000

    def _build_user_prompt(self, observation, private_ctx) -> str:
        scenario = observation.scenario
        turns = observation.turns
        my_role = observation.my_role
        counterparty = observation.counterparty_role
        role_utility_guidance = (
            "For this buyer role, lower prices create higher utility. "
            "A deal beats BATNA only when the negotiated price is below the buyer BATNA, "
            "and reserve price is the maximum acceptable price."
            if my_role == "buyer"
            else "For this seller role, higher prices create higher utility. "
            "A deal beats BATNA only when the negotiated price is above the seller BATNA, "
            "and reserve price is the minimum acceptable price."
        )

        transcript = ""
        for t in turns:
            role_label = "You" if t.agent == my_role else "Counterparty"
            offer_str = ""
            if t.offer is not None:
                offer_str = f" [offer: {t.offer.model_dump()}]"
            transcript += f"{role_label}: {t.content}{offer_str}\n"

        private = private_ctx
        my_batna = private.batna
        reserve = private.reserve_price
        constraints = private.hard_constraints
        walk_away = private.walk_away_threshold

        prompt = f"""You are the {my_role} in a {scenario.domain.value} negotiation.
Scenario: {scenario.scenario_description}
Your private information:
- BATNA (best alternative): ${my_batna}
- Reserve price: ${reserve or 'Not specified'}
- Walk-away threshold: {walk_away if walk_away is not None else 'Not specified'} (if set, walk away when deal utility falls below this)
- Hard constraints: {', '.join(constraints) if constraints else 'None'}
- Private info: {', '.join(private.private_info) if private.private_info else 'None'}

The counterparty is the {counterparty}.
Max turns remaining: {observation.turns_remaining}
Utility guidance:
- {role_utility_guidance}

Expected structured offer fields for this domain: {scenario.deal_schema}

Transcript so far:
{transcript if transcript else 'No messages yet.'}

Respond with a JSON object containing your action.
action_type must be one of: "message", "offer", "accept", "walk_away".
Include an "offer" field ONLY if action_type is "offer"."""

        return prompt

    async def act(self, observation, private_ctx) -> Action:
        user_prompt = self._build_user_prompt(observation, private_ctx)
        attempt_counter = {"count": 0}
        self.last_call_metadata = {
            "requested_model_id": self.model_id,
            "backend": _infer_backend(self.model_id),
            "route": _infer_route(self.model_id),
            "system_prompt_hash": _hash_text(self.system_prompt),
            "user_prompt_hash": _hash_text(user_prompt),
            "prompt_version": "closed_api_adapter_v1",
            "temperature": self.temperature,
            "timeout_seconds": self.timeout,
            "retry_policy": {
                "max_retries": 3,
                "base_delay_seconds": 1.0,
            },
            "max_completion_tokens": self.max_completion_tokens,
        }

        async def _call_api() -> dict[str, Any]:
            attempt_counter["count"] += 1
            response = await self._make_api_call(self.system_prompt, user_prompt)
            return response

        response = await retry_with_backoff(
            _call_api,
            max_retries=3,
            base_delay=1.0,
            timeout=self.timeout,
        )
        content = response.get("content", "")
        prompt_tokens = response.get("prompt_tokens", 0)
        completion_tokens = response.get("completion_tokens", 0)
        self._track_tokens(prompt_tokens, completion_tokens)
        self.last_call_metadata.update(
            {
                "attempt_count": attempt_counter["count"],
                "retries_used": max(0, attempt_counter["count"] - 1),
                "resolved_model_id": response.get("resolved_model_id", self.model_id),
                "provider_route": response.get("provider_route"),
                "provider_route_status": response.get("provider_route_status", "unknown"),
                "base_url": response.get("base_url"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": self.total_cost,
                "finish_reason": response.get("finish_reason"),
                "stop_reason": response.get("stop_reason"),
            }
        )

        action_type, offer_dict, protocol_metadata = self._extract_action(
            content,
            observation.scenario.domain.value,
        )

        return Action(
            action_type=action_type,
            content=content,
            offer_dict=offer_dict,
            metadata={
                "protocol": {
                    **protocol_metadata,
                    "attempt_count": self.last_call_metadata.get("attempt_count"),
                    "retries_used": self.last_call_metadata.get("retries_used"),
                    "finish_reason": self.last_call_metadata.get("finish_reason"),
                    "stop_reason": self.last_call_metadata.get("stop_reason"),
                    "content_empty": not bool(content.strip()),
                    "max_tokens_reached": (
                        self.last_call_metadata.get("finish_reason") in {"length", "max_tokens"}
                        or self.last_call_metadata.get("stop_reason") == "max_tokens"
                    ),
                }
            },
        )

    def get_runtime_metadata(self) -> dict[str, Any]:
        return {
            "requested_model_id": self.model_id,
            "backend": _infer_backend(self.model_id),
            "route": _infer_route(self.model_id),
            "stance": self.stance,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout,
            **self.last_call_metadata,
        }

    async def _make_api_call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        model = self.model_id.lower()

        if "openrouter" in model:
            return await self._call_openrouter(system_prompt, user_prompt)
        elif "claude" in model or "anthropic" in model:
            return await self._call_anthropic(system_prompt, user_prompt)
        elif "deepseek" in model:
            return await self._call_deepseek(system_prompt, user_prompt)
        elif "gpt" in model or "openai" in model or "o1" in model or "o3" in model:
            return await self._call_openai(system_prompt, user_prompt)
        elif "gemini" in model or "google" in model:
            return await self._call_google(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown model provider for: {self.model_id}")

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if hasattr(response, "error") and response.error:
            if "rate" in str(response.error).lower():
                raise AgentRateLimitError(str(response.error))
            raise RuntimeError(str(response.error))
        content = response.content[0].text if response.content else ""
        return {
            "content": content,
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "resolved_model_id": getattr(response, "model", None) or self.model_id,
            "provider_route": "anthropic_direct",
            "provider_route_status": "resolved_direct",
            "base_url": "https://api.anthropic.com",
            "stop_reason": getattr(response, "stop_reason", None),
            "native_structured_output_requested": False,
        }

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = AsyncOpenAI()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "resolved_model_id": getattr(response, "model", None) or self.model_id,
            "provider_route": "openai_direct",
            "provider_route_status": "resolved_direct",
            "base_url": "https://api.openai.com/v1",
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "native_structured_output_requested": True,
        }

    async def _call_google(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import os
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY"))
        client = genai.Client(api_key=api_key)
        # Google SDK requires models/ prefix
        google_model = self.model_id
        if not google_model.startswith("models/"):
            google_model = f"models/{google_model}"
        response = await client.aio.models.generate_content(
            model=google_model,
            contents=user_prompt,
            config={"system_instruction": system_prompt},
        )
        content = response.text if response.text else ""
        return {
            "content": content,
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            "resolved_model_id": google_model,
            "provider_route": "google_direct",
            "provider_route_status": "resolved_direct",
            "base_url": "https://generativelanguage.googleapis.com",
            "finish_reason": (
                getattr(response.candidates[0], "finish_reason", None)
                if getattr(response, "candidates", None)
                else None
            ),
            "native_structured_output_requested": False,
        }

    async def _call_openrouter(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import os
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")

        # Strip openrouter/ prefix if present in model_id
        router_model = self.model_id
        if router_model.startswith("openrouter/"):
            router_model = router_model[len("openrouter/"):]

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/deepak-karkala/concord-bench",
                "X-Title": "Concord Benchmark",
            },
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await client.chat.completions.create(
            model=router_model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "resolved_model_id": router_model,
            "provider_route": getattr(response, "provider", None),
            "provider_route_status": (
                "resolved_provider" if getattr(response, "provider", None) else "requested_via_openrouter"
            ),
            "base_url": "https://openrouter.ai/api/v1",
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "native_structured_output_requested": True,
        }

    async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import os
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY"))
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "resolved_model_id": getattr(response, "model", None) or self.model_id,
            "provider_route": "deepseek_direct",
            "provider_route_status": "resolved_direct",
            "base_url": "https://api.deepseek.com",
            "finish_reason": response.choices[0].finish_reason if response.choices else None,
            "native_structured_output_requested": False,
        }

    def _extract_action(self, content: str, domain: str) -> tuple[ActionType, dict | None, dict[str, Any]]:
        offer_dict = None
        action_type = ActionType.MESSAGE
        protocol_metadata: dict[str, Any] = {
            "native_structured_output_requested": bool(
                self.last_call_metadata.get("native_structured_output_requested")
            ),
            "native_structured_output_success": False,
            "json_object_detected": False,
            "action_parse_success": False,
            "requested_offer_action": False,
            "structured_offer_valid": False,
            "max_tokens_reached": False,
            "salvage_parse_used": False,
            "parse_path": "unparsed",
        }

        # Try to parse entire content as JSON first
        data = self._extract_json_object(content)
        if data and isinstance(data, dict) and "action_type" in data:
            protocol_metadata["json_object_detected"] = True
            at = data.get("action_type", "message").lower()
            action, offer_dict, action_metadata = self._parse_action(
                data,
                at,
                domain,
                native_requested=protocol_metadata["native_structured_output_requested"],
            )
            protocol_metadata.update(action_metadata)
            protocol_metadata["action_parse_success"] = True
            if protocol_metadata["native_structured_output_requested"]:
                protocol_metadata["native_structured_output_success"] = True
                protocol_metadata["parse_path"] = "native_structured_json"
            return action, offer_dict, protocol_metadata

        try:
            parsed_offer = parse_raw_offer_with_metadata(content, domain)
            protocol_metadata["requested_offer_action"] = True
            protocol_metadata["structured_offer_valid"] = True
            protocol_metadata["salvage_parse_used"] = parsed_offer.parse_path == "regex_salvage"
            protocol_metadata["parse_path"] = parsed_offer.parse_path
            protocol_metadata["native_structured_output_success"] = False
            protocol_metadata["action_parse_success"] = True
            return ActionType.OFFER, parsed_offer.offer.model_dump(), protocol_metadata
        except Exception:
            pass

        # Last resort: keyword fallback on first 100 chars
        lower = content.lower()
        if "walk away" in lower[:200]:
            action_type = ActionType.WALK_AWAY
            protocol_metadata["parse_path"] = "keyword_fallback"
        elif action_type == ActionType.MESSAGE and '"action_type": "accept"' in lower:
            action_type = ActionType.ACCEPT
            protocol_metadata["parse_path"] = "keyword_fallback"

        return action_type, offer_dict, protocol_metadata

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        import json as _json
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except _json.JSONDecodeError:
                        return None
        return None

    def _parse_action(
        self,
        data: dict,
        at: str,
        domain: str,
        *,
        native_requested: bool = False,
    ) -> tuple[ActionType, dict | None, dict[str, Any]]:
        import json as _json
        offer_dict = None
        action_type = ActionType.MESSAGE
        protocol_metadata = {
            "requested_offer_action": at == "offer",
            "structured_offer_valid": False,
            "parse_path": "json_object",
            "salvage_parse_used": False,
        }

        if at == "offer":
            action_type = ActionType.OFFER
            if data.get("offer"):
                try:
                    parsed_offer = parse_raw_offer_with_metadata(_json.dumps(data["offer"]), domain)
                    offer_dict = parsed_offer.offer.model_dump()
                    protocol_metadata["structured_offer_valid"] = True
                    protocol_metadata["salvage_parse_used"] = parsed_offer.parse_path == "regex_salvage"
                    protocol_metadata["parse_path"] = (
                        "native_structured_json" if native_requested else "json_object"
                    )
                except Exception:
                    pass
        elif at == "accept":
            action_type = ActionType.ACCEPT
            protocol_metadata["parse_path"] = "json_object"
        elif at == "walk_away":
            action_type = ActionType.WALK_AWAY
            protocol_metadata["parse_path"] = "json_object"

        return action_type, offer_dict, protocol_metadata


def _lookup_model_costs(model_id: str) -> tuple[float, float]:
    normalized_ids = [model_id]
    if model_id.startswith("openrouter/"):
        normalized_ids.append(model_id[len("openrouter/"):])

    for normalized in normalized_ids:
        exact = _MODEL_COSTS_PER_1M.get(normalized)
        if exact is not None:
            return exact

        for known_prefix, cost in _MODEL_COSTS_PER_1M.items():
            if normalized.startswith(f"{known_prefix}-"):
                return cost

    return (0.0, 0.0)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _infer_backend(model_id: str) -> str:
    normalized = model_id.lower()
    if normalized.startswith("openrouter/"):
        return "openrouter"
    if "anthropic" in normalized or "claude" in normalized:
        return "anthropic"
    if "openai" in normalized or "gpt" in normalized or "o1" in normalized or "o3" in normalized:
        return "openai"
    if "google" in normalized or "gemini" in normalized:
        return "google"
    if "deepseek" in normalized:
        return "deepseek"
    return "unknown"


def _infer_route(model_id: str) -> str:
    if model_id.startswith("openrouter/"):
        return model_id[len("openrouter/"):]
    return model_id
