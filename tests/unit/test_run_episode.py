import pytest
from pathlib import Path

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


class _ProtocolAwareMessageAgent:
    async def act(self, env_state, private_ctx):
        if env_state.current_agent == "buyer":
            return Action(
                action_type=ActionType.MESSAGE,
                content="",
                metadata={
                    "protocol": {
                        "json_object_detected": True,
                        "action_parse_success": True,
                        "requested_offer_action": False,
                        "structured_offer_valid": False,
                        "content_empty": True,
                        "retries_used": 2,
                        "max_tokens_reached": True,
                    }
                },
            )
        return Action(action_type=ActionType.WALK_AWAY, content="done")


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

    def fake_resolve_agent(
        model: str,
        stance: str = "default",
        timeout: float | None = None,
        system_prompt: str = "",
        max_completion_tokens: int = 1024,
    ):
        if model == "buyer-message":
            return _MessageAgent()
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    episode = await run_episode(sample_scenario, buyer_model="buyer-message", seller_model="seller-walk")

    assert episode.grades.walk_away_correct is False


@pytest.mark.asyncio
async def test_run_episode_records_reproducibility_metadata(monkeypatch, sample_scenario, temp_dir: Path):
    manifest_path = temp_dir / "phase1_manifest.json"
    manifest_path.write_text('{"panel":"phase1"}\n')

    def fake_resolve_agent(
        model: str,
        stance: str = "default",
        timeout: float | None = None,
        system_prompt: str = "",
        max_completion_tokens: int = 1024,
    ):
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    episode = await run_episode(
        sample_scenario,
        buyer_model="greedy",
        seller_model="honest",
        seed=42,
        model_panel_manifest_path=str(manifest_path),
    )

    assert episode.metadata["prompt_version"] == "scenario_description_v1"
    assert len(episode.metadata["prompt_hash"]) == 16
    assert episode.metadata["episode_started_at"]
    assert episode.metadata["episode_completed_at"]
    assert episode.metadata["buyer_max_completion_tokens"] == 1024
    assert episode.metadata["buyer_runtime"]["backend"] == "scripted"
    assert episode.metadata["seller_runtime"]["backend"] == "scripted"
    assert episode.metadata["model_panel_manifest"]["path"] == str(manifest_path)
    assert len(episode.metadata["model_panel_manifest"]["hash"]) == 16


@pytest.mark.asyncio
async def test_run_episode_persists_turn_protocol_metadata(monkeypatch, sample_scenario):
    def fake_resolve_agent(
        model: str,
        stance: str = "default",
        timeout: float | None = None,
        system_prompt: str = "",
        max_completion_tokens: int = 1024,
    ):
        if model == "protocol-buyer":
            return _ProtocolAwareMessageAgent()
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    episode = await run_episode(
        sample_scenario,
        buyer_model="protocol-buyer",
        seller_model="honest",
    )

    buyer_turn = episode.turns[0]
    assert buyer_turn.metadata["protocol"]["content_empty"] is True
    assert buyer_turn.metadata["protocol"]["retries_used"] == 2
    assert buyer_turn.metadata["protocol"]["max_tokens_reached"] is True


@pytest.mark.asyncio
async def test_run_episode_passes_buyer_prompt_and_token_overrides(monkeypatch, sample_scenario):
    captured: list[dict] = []

    def fake_resolve_agent(
        model: str,
        stance: str = "default",
        timeout: float | None = None,
        system_prompt: str = "",
        max_completion_tokens: int = 1024,
    ):
        captured.append(
            {
                "model": model,
                "stance": stance,
                "timeout": timeout,
                "system_prompt": system_prompt,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        return _WalkAwayAgent()

    monkeypatch.setattr("concord.runners.run_episode._resolve_agent", fake_resolve_agent)

    episode = await run_episode(
        sample_scenario,
        buyer_model="gpt-5.4-nano",
        seller_model="honest",
        buyer_system_prompt="custom buyer prompt",
        buyer_max_completion_tokens=4096,
    )

    assert captured[0]["model"] == "gpt-5.4-nano"
    assert captured[0]["system_prompt"] == "custom buyer prompt"
    assert captured[0]["max_completion_tokens"] == 4096
    assert captured[1]["model"] == "honest"
    assert captured[1]["system_prompt"] == ""
    assert captured[1]["max_completion_tokens"] == 1024
    assert episode.metadata["buyer_max_completion_tokens"] == 4096
