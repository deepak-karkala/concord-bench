import pytest

from concord.graders.calibration import compute_cohens_kappa, monitor_drift


class TestCohensKappa:
    def test_perfect_agreement(self):
        kappa = compute_cohens_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
        assert kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        kappa = compute_cohens_kappa([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
        assert kappa <= 0.0

    def test_empty_lists(self):
        assert compute_cohens_kappa([], []) == 0.0

    def test_mismatched_lengths(self):
        assert compute_cohens_kappa([1, 0], [1]) == 0.0

    def test_moderate_agreement(self):
        judge = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        author = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
        kappa = compute_cohens_kappa(judge, author)
        assert 0.6 < kappa < 1.0


class TestMonitorDrift:
    def test_no_drift_above_threshold(self):
        assert monitor_drift(0.8, [0.75, 0.78, 0.82]) is False

    def test_drift_below_threshold(self):
        assert monitor_drift(0.65, []) is True

    def test_drift_significant_decline(self):
        assert monitor_drift(0.60, [0.80, 0.82, 0.79]) is True

    def test_no_drift_small_decline(self):
        assert monitor_drift(0.72, [0.75, 0.76, 0.75]) is False
