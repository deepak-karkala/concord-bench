import pytest

from concord.analysis.aggregate import aggregate_by_model
from concord.analysis.bootstrap_ci import bootstrap_ci, build_dimension_score
from concord.analysis.model_card import generate_model_card, model_card_to_markdown
from concord.schemas.episode import EpisodeLog, GradeReport


@pytest.fixture
def sample_episodes():
    return [
        EpisodeLog(
            scenario_id="test-1",
            grades=GradeReport(principal_utility=0.7, joint_welfare=0.6, coercion_score=0.1, cultural_sensitivity_score=0.9, privacy_leak=False, batna_leaked=False),
            metadata={"buyer_model": "model-a"},
        ),
        EpisodeLog(
            scenario_id="test-2",
            grades=GradeReport(principal_utility=0.8, joint_welfare=0.7, coercion_score=0.2, cultural_sensitivity_score=0.85, privacy_leak=False, batna_leaked=False),
            metadata={"buyer_model": "model-a"},
        ),
        EpisodeLog(
            scenario_id="test-3",
            grades=GradeReport(principal_utility=0.6, joint_welfare=0.5, coercion_score=0.3, cultural_sensitivity_score=0.8, privacy_leak=True, batna_leaked=True),
            metadata={"buyer_model": "model-b"},
        ),
    ]


class TestAggregate:
    def test_aggregate_by_model(self, sample_episodes):
        agg = aggregate_by_model(sample_episodes)
        assert "model-a" in agg
        assert "model-b" in agg
        assert agg["model-a"]["principal_utility"] == pytest.approx(0.75)

    def test_empty_episodes(self):
        agg = aggregate_by_model([])
        assert agg == {}


class TestBootstrapCI:
    def test_ci_range(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        ci = bootstrap_ci(values, n_iterations=100)
        assert ci.lower < ci.upper
        assert ci.confidence == 0.95

    def test_empty_values(self):
        ci = bootstrap_ci([], n_iterations=100)
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_build_dimension_score(self):
        ds = build_dimension_score([0.5, 0.6, 0.7, 0.8, 0.9], "test")
        assert ds.mean == pytest.approx(0.7)
        assert ds.n_episodes == 5
        assert ds.ci95 is not None


class TestModelCard:
    def test_generate_card(self, sample_episodes):
        model_a_eps = [e for e in sample_episodes if e.metadata["buyer_model"] == "model-a"]
        card = generate_model_card("model-a", "0.1.0", model_a_eps)
        assert card.model_id == "model-a"
        assert card.total_episodes == 2
        assert "principal_utility" in card.outcome

    def test_markdown_output(self, sample_episodes):
        model_a_eps = [e for e in sample_episodes if e.metadata["buyer_model"] == "model-a"]
        card = generate_model_card("model-a", "0.1.0", model_a_eps)
        md = model_card_to_markdown(card)
        assert "# Model Card: model-a" in md
        assert "principal_utility" in md
