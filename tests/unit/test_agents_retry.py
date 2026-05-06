import asyncio

import pytest

from concord.agents.base import Action
from concord.agents.closed_api_adapter import ClosedAPIAdapter
from concord.agents.retry import AgentRetryError, AgentTimeoutError, retry_with_backoff
from concord.schemas.episode import ActionType


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        async def succeed():
            return "ok"

        result = await retry_with_backoff(succeed, max_retries=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_succeed(self):
        attempts = []

        async def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("temporary failure")
            return "recovered"

        result = await retry_with_backoff(flaky, max_retries=5)
        assert result == "recovered"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        async def always_fail():
            raise RuntimeError("persistent error")

        with pytest.raises(AgentRetryError, match="persistent error"):
            await retry_with_backoff(always_fail, max_retries=2)

    @pytest.mark.asyncio
    async def test_timeout(self):
        async def slow():
            await asyncio.sleep(10)
            return "too late"

        with pytest.raises(AgentTimeoutError):
            await retry_with_backoff(slow, max_retries=1, timeout=0.01)

    @pytest.mark.asyncio
    async def test_no_timeout_when_none(self):
        async def quick():
            return "fast"

        result = await retry_with_backoff(quick, timeout=None)
        assert result == "fast"


class TestClosedAPIAdapter:
    def test_init_defaults(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        assert adapter.model_id == "gpt-5.2"
        assert adapter.temperature == 0.7
        assert adapter.timeout == 120.0
        assert adapter.total_prompt_tokens == 0
        assert adapter.total_cost == 0.0

    def test_custom_params(self):
        adapter = ClosedAPIAdapter("claude-opus-4-7", temperature=0.3, timeout=60.0)
        assert adapter.temperature == 0.3
        assert adapter.timeout == 60.0

    def test_cost_tracking(self):
        adapter = ClosedAPIAdapter("claude-opus-4-7")
        adapter._track_tokens(prompt_tokens=1000000, completion_tokens=1000000)
        assert adapter.total_prompt_tokens == 1000000
        assert adapter.total_completion_tokens == 1000000
        assert adapter.total_cost == pytest.approx(90.0)  # 15 + 75

    def test_cost_tracking_zero_cost_unknown_model(self):
        adapter = ClosedAPIAdapter("unknown-model")
        adapter._track_tokens(prompt_tokens=1000, completion_tokens=500)
        assert adapter.total_cost == 0.0

    def test_extract_action_offer(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        content = '{"reasoning": "I should make an offer.", "action_type": "offer", "offer": {"domain": "ecommerce", "price": 150, "quantity": 100}}'
        action_type, offer_dict = adapter._extract_action(content, "ecommerce")
        assert action_type == ActionType.OFFER
        assert offer_dict is not None
        assert offer_dict.get("price") == 150.0
        assert offer_dict is not None
        assert offer_dict.get("domain") == "ecommerce"
        assert offer_dict.get("price") == 150.0

    def test_extract_action_walk_away(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        content = "I cannot reach an acceptable deal. I will walk away."
        action_type, offer_dict = adapter._extract_action(content, "ecommerce")
        assert action_type == ActionType.WALK_AWAY
        assert offer_dict is None

    def test_extract_action_accept(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        content = '{"reasoning": "I accept.", "action_type": "accept"}'
        action_type, offer_dict = adapter._extract_action(content, "ecommerce")
        assert action_type == ActionType.ACCEPT

    def test_extract_action_message_default(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        content = "Let me think about this counter-offer."
        action_type, offer_dict = adapter._extract_action(content, "ecommerce")
        assert action_type == ActionType.MESSAGE
        assert offer_dict is None

    def test_inline_json_detection(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        content = 'Some text {"action_type": "offer", "reasoning": "fair", "offer": {"domain": "settlement", "settlement_amount": 50000, "confidentiality_clause": true}}'
        action_type, offer_dict = adapter._extract_action(content, "settlement")
        assert action_type == ActionType.OFFER
        assert offer_dict is not None
        assert offer_dict.get("settlement_amount") == 50000.0
        assert offer_dict is not None
        assert offer_dict.get("settlement_amount") == 50000.0

    def test_build_user_prompt(self, sample_scenario):
        from concord.env.core import NegotiationEnv
        env = NegotiationEnv()
        env.reset(sample_scenario)
        env.step_message("buyer", "Hello, I want to discuss pricing.")
        adapter = ClosedAPIAdapter("claude-opus-4-7")
        prompt = adapter._build_user_prompt(env.state, sample_scenario.buyer_context)
        assert "ecommerce" in prompt
        assert "Hello, I want to discuss pricing" in prompt
        assert "BATNA" in prompt

    def test_prompt_hash_deterministic(self):
        adapter = ClosedAPIAdapter("gpt-5.2")
        h1 = adapter._prompt_hash("test prompt")
        h2 = adapter._prompt_hash("test prompt")
        assert h1 == h2
        assert len(h1) == 16
