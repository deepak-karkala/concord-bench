from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from concord.schemas.episode import ConfidenceInterval, DimensionScore


def bootstrap_ci(values: list[float], n_iterations: int = 1000, confidence: float = 0.95) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(lower=0.0, upper=0.0, confidence=confidence)

    import random

    means: list[float] = []
    n = len(values)
    for _ in range(n_iterations):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    tail = (1 - confidence) / 2
    lower_idx = int(tail * n_iterations)
    upper_idx = int((1 - tail) * n_iterations) - 1

    return ConfidenceInterval(
        lower=means[max(0, lower_idx)],
        upper=means[min(len(means) - 1, upper_idx)],
        confidence=confidence,
    )


def build_dimension_score(values: list[float], dim_name: str) -> DimensionScore:
    mean_val = sum(values) / len(values) if values else 0.0
    ci = bootstrap_ci(values) if len(values) >= 5 else None
    return DimensionScore(mean=mean_val, ci95=ci, n_episodes=len(values))
