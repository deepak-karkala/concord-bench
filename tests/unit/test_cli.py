from click.testing import CliRunner

from concord.cli import main


def test_run_uses_honest_seller_by_default(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    async def fake_run_episode(*args, **kwargs):
        recorded.update(kwargs)

        class DummyEpisode:
            scenario_id = "seed-test"
            turns = []
            deal = None
            metadata = {}

            def model_dump(self):
                return {}

        return DummyEpisode()

    monkeypatch.setattr("concord.cli._find_scenario", lambda scenario: object())
    monkeypatch.setattr("concord.runners.run_episode.run_episode", fake_run_episode)

    result = CliRunner().invoke(
        main,
        ["run", "--model", "greedy", "--scenario", "seed-test", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert recorded["seller_model"] == "honest"


def test_run_batch_uses_honest_seller_by_default(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    async def fake_run_batch(*args, **kwargs):
        recorded.update(kwargs)
        return []

    monkeypatch.setattr("concord.cli.load_seeds", lambda **kwargs: [object()])
    monkeypatch.setattr("concord.runners.run_batch._resolve_concurrency_limit", lambda **kwargs: 2)
    monkeypatch.setattr("concord.runners.run_batch.run_batch", fake_run_batch)

    result = CliRunner().invoke(
        main,
        ["run-batch", "--models", "greedy", "--scenarios", "all", "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert recorded["seller_model"] == "honest"


def test_run_allows_explicit_seller_override(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    async def fake_run_episode(*args, **kwargs):
        recorded.update(kwargs)

        class DummyEpisode:
            scenario_id = "seed-test"
            turns = []
            deal = None
            metadata = {}

            def model_dump(self):
                return {}

        return DummyEpisode()

    monkeypatch.setattr("concord.cli._find_scenario", lambda scenario: object())
    monkeypatch.setattr("concord.runners.run_episode.run_episode", fake_run_episode)

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--model",
            "greedy",
            "--seller",
            "galaxy_brain",
            "--scenario",
            "seed-test",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert recorded["seller_model"] == "galaxy_brain"
