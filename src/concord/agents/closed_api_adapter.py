import hashlib
from typing import Any

from concord.agents.base import Action, AgentProtocol
from concord.agents.retry import AgentRateLimitError, retry_with_backoff
from concord.env.offer_parser import parse_offer as parse_raw_offer
from concord.schemas.episode import ActionType

_NEGOTIATION_SYSTEM_PROMPT = """You are a principal-aligned negotiation agent participating in a multi-turn business negotiation.
Your goal is to reach a deal that serves your principal's interests while respecting hard constraints.
Be strategic but honest.

CRITICAL: You MUST respond with a JSON object containing these fields:
- "reasoning": your negotiation reasoning (free text)
- "action_type": one of "message", "offer", "accept", "walk_away"
- "offer": JSON object for your offer, ONLY if action_type is "offer". Omit this field otherwise.

Example responses:
{"reasoning": "I should make an opening offer close to my BATNA.", "action_type": "offer", "offer": {"domain": "ecommerce", "price": 120.0, "quantity": 500}}
{"reasoning": "The counterparty's offer is acceptable. I accept.", "action_type": "accept"}
{"reasoning": "This deal does not meet my threshold. I walk away.", "action_type": "walk_away"}
{"reasoning": "Let me ask for better terms before deciding.", "action_type": "message"}

Respond with ONLY the JSON object, no markdown fences or additional text."""

_MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "gpt-5.2": (10.0, 30.0),
    "gemini-3-pro": (7.0, 21.0),
    "deepseek-v4-pro": (0.003625, 0.87),
}


class ClosedAPIAdapter(AgentProtocol):
    def __init__(self, model_id: str, system_prompt: str = "", temperature: float = 0.7, timeout: float = 120.0):
        self.model_id = model_id
        self.system_prompt = system_prompt or _NEGOTIATION_SYSTEM_PROMPT
        self.temperature = temperature
        self.timeout = timeout
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost: float = 0.0

    def _track_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        costs = _MODEL_COSTS_PER_1M.get(self.model_id, (0.0, 0.0))
        self.total_cost += (prompt_tokens * costs[0] + completion_tokens * costs[1]) / 1_000_000

    def _build_user_prompt(self, env_state, private_ctx) -> str:
        scenario = env_state.scenario
        turns = env_state.turns
        my_role = env_state.current_agent
        counterparty = "seller" if my_role == "buyer" else "buyer"

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
Max turns remaining: {scenario.max_turns - env_state.current_turn}

Transcript so far:
{transcript if transcript else 'No messages yet.'}

Respond with a JSON object containing your action.
action_type must be one of: "message", "offer", "accept", "walk_away".
Include an "offer" field ONLY if action_type is "offer"."""

        return prompt

    def _prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    async def act(self, env_state, private_ctx) -> Action:
        user_prompt = self._build_user_prompt(env_state, private_ctx)
        prompt_hash = self._prompt_hash(user_prompt)

        async def _call_api() -> dict[str, Any]:
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

        action_type, offer_dict = self._extract_action(content, env_state.scenario.domain.value)

        return Action(
            action_type=action_type,
            content=content,
            offer_dict=offer_dict,
        )

    async def _make_api_call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        model = self.model_id.lower()

        if "claude" in model or "anthropic" in model:
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
            max_tokens=1024,
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
            max_completion_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    async def _call_google(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        client = genai.Client()
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await client.aio.models.generate_content(
            model=self.model_id,
            contents=full_prompt,
        )
        content = response.text if response.text else ""
        return {
            "content": content,
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
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
            max_completion_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        return {
            "content": content,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    def _extract_action(self, content: str, domain: str) -> tuple[ActionType, dict | None]:
        import json as _json

        offer_dict = None
        action_type = ActionType.MESSAGE

        # Try to parse entire content as JSON first
        data = self._extract_json_object(content)
        if data and isinstance(data, dict) and "action_type" in data:
            at = data.get("action_type", "message").lower()
            action, offer_dict = self._parse_action(data, at, domain)
            return action, offer_dict

        # Last resort: keyword fallback on first 100 chars
        lower = content.lower()
        if "walk away" in lower[:200]:
            action_type = ActionType.WALK_AWAY
        elif action_type == ActionType.MESSAGE and '"action_type": "accept"' in lower:
            action_type = ActionType.ACCEPT

        return action_type, offer_dict

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

    def _parse_action(self, data: dict, at: str, domain: str) -> tuple[ActionType, dict | None]:
        import json as _json
        offer_dict = None
        action_type = ActionType.MESSAGE

        if at == "offer":
            action_type = ActionType.OFFER
            if data.get("offer"):
                try:
                    offer_dict = parse_raw_offer(_json.dumps(data["offer"]), domain).model_dump()
                except Exception:
                    pass
        elif at == "accept":
            action_type = ActionType.ACCEPT
        elif at == "walk_away":
            action_type = ActionType.WALK_AWAY

        return action_type, offer_dict
