import pytest

from concord.agents.base import Action
from concord.runners.run_episode import run_episode
from concord.schemas.episode import ActionType


class _InvalidOfferAgent:
    async def act(self, env_state, private_ctx):
        return Action(
            action_type=ActionType.OFFER,
            content="broken offer",
            offer_dict={"price": "not-a-number"},
        )


class _WalkAwayAgent:
    async def act(self, env_state, private_ctx):
        return Action(
            action_type=ActionType.WALK_AWAY,
            content="no deal",
        )


class _MessageAgent:
    async def act(self, env_state, private_ctx):
        return Action(
            action_type=ActionType.MESSAGE,
            content="need better terms",
        )


@pytest.mark.asyncio
async def test_run_episode_raises_on_invalid_structured_offer(monkeypatch, sample_scenario):
    def fake_resolve_agent(model: str, stance: str = "default", timeout: float | None = None):
        if model == "broken-buyer":
            return _InvalidOfferAgent()
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    with pytest.raises(Exception):
        await run_episode(sample_scenario, buyer_model="broken-buyer", seller_model="honest")


@pytest.mark.asyncio
async def test_run_episode_scores_seller_walk_away_against_seller_context(monkeypatch, sample_scenario):
    sample_scenario.buyer_context.walk_away_threshold = 0.4
    sample_scenario.seller_context.walk_away_threshold = None

    def fake_resolve_agent(model: str, stance: str = "default", timeout: float | None = None):
        if model == "buyer-message":
            return _MessageAgent()
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    episode = await run_episode(sample_scenario, buyer_model="buyer-message", seller_model="seller-walk")

    assert episode.grades.walk_away_correct is False
