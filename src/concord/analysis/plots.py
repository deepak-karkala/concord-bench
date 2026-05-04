from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def radar_chart(scores: dict[str, dict[str, float]], title: str = "Model Comparison") -> "Figure":
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib and numpy. Install with: pip install matplotlib numpy")

    dimensions = list(next(iter(scores.values())).keys())
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, dims in scores.items():
        values = [dims.get(d, 0) for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=8)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_ylim(0, 1)

    return fig


def distribution_plot(values_list: list[list[float]], labels: list[str], title: str = "Score Distribution") -> "Figure":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values_list, labels=labels)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    return fig


def repeated_game_trajectory(
    round_scores: dict[str, dict[int, float]],
    title: str = "Repeated Game Trajectory",
) -> "Figure":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, scores_by_round in round_scores.items():
        rounds = sorted(scores_by_round.keys())
        vals = [scores_by_round[r] for r in rounds]
        ax.plot(rounds, vals, "o-", linewidth=2, label=model)

    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    return fig
