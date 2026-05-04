from concord.schemas.episode import EpisodeLog, ModelCard
from concord.analysis.bootstrap_ci import build_dimension_score


def generate_model_card(
    model_id: str,
    concord_version: str,
    episodes: list[EpisodeLog],
) -> ModelCard:
    utility_vals: list[float] = []
    welfare_vals: list[float] = []
    coercion_vals: list[float] = []
    cultural_vals: list[float] = []
    constraint_vals: list[float] = []
    privacy_vals: list[float] = []
    batna_vals: list[float] = []

    for ep in episodes:
        g = ep.grades
        utility_vals.append(g.principal_utility or 0)
        welfare_vals.append(g.joint_welfare or 0)
        coercion_vals.append(g.coercion_score or 0)
        cultural_vals.append(g.cultural_sensitivity_score or 0)
        constraint_vals.append(1.0 if g.hard_constraint_violations else 0.0)
        privacy_vals.append(1.0 if g.privacy_leak else 0.0)
        batna_vals.append(1.0 if g.batna_leaked else 0.0)

    card = ModelCard(
        model_id=model_id,
        concord_version=concord_version,
        outcome={
            "principal_utility": build_dimension_score(utility_vals, "principal_utility"),
            "joint_welfare": build_dimension_score(welfare_vals, "joint_welfare"),
        },
        constraints={
            "hard_constraint_violations": build_dimension_score(constraint_vals, "hard_constraint_violations"),
        },
        social={
            "coercion": build_dimension_score(coercion_vals, "coercion"),
            "cultural_sensitivity": build_dimension_score(cultural_vals, "cultural_sensitivity"),
        },
        robustness={
            "privacy_leak": build_dimension_score(privacy_vals, "privacy_leak"),
            "batna_leaked": build_dimension_score(batna_vals, "batna_leaked"),
        },
        total_episodes=len(episodes),
    )
    return card


def model_card_to_markdown(card: ModelCard) -> str:
    lines = [
        f"# Model Card: {card.model_id}",
        f"Concord version: {card.concord_version}",
        f"Total episodes: {card.total_episodes}",
        "",
        "## Outcome Metrics",
    ]
    for name, score in card.outcome.items():
        ci_str = f" ({score.ci95.lower:.3f}–{score.ci95.upper:.3f})" if score.ci95 else ""
        lines.append(f"- **{name}**: {score.mean:.3f}{ci_str} (n={score.n_episodes})")

    lines.extend(["", "## Constraint Metrics"])
    for name, score in card.constraints.items():
        ci_str = f" ({score.ci95.lower:.3f}–{score.ci95.upper:.3f})" if score.ci95 else ""
        lines.append(f"- **{name}**: {score.mean:.3f}{ci_str} (n={score.n_episodes})")

    lines.extend(["", "## Social Metrics"])
    for name, score in card.social.items():
        ci_str = f" ({score.ci95.lower:.3f}–{score.ci95.upper:.3f})" if score.ci95 else ""
        lines.append(f"- **{name}**: {score.mean:.3f}{ci_str} (n={score.n_episodes})")

    lines.extend(["", "## Robustness Metrics"])
    for name, score in card.robustness.items():
        ci_str = f" ({score.ci95.lower:.3f}–{score.ci95.upper:.3f})" if score.ci95 else ""
        lines.append(f"- **{name}**: {score.mean:.3f}{ci_str} (n={score.n_episodes})")

    return "\n".join(lines)
