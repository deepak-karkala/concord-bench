from collections import defaultdict

from concord.schemas.episode import EpisodeLog


def aggregate_by_model(episodes: list[EpisodeLog]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ep in episodes:
        model = ep.metadata.get("buyer_model", "unknown")
        grades = ep.grades
        results[model]["principal_utility"].append(grades.principal_utility or 0)
        results[model]["joint_welfare"].append(grades.joint_welfare or 0)
        results[model]["coercion_score"].append(grades.coercion_score or 0)
        results[model]["cultural_sensitivity"].append(grades.cultural_sensitivity_score or 0)
        results[model]["constraint_violations"].append(float(len(grades.hard_constraint_violations)))
        results[model]["privacy_leak"].append(1.0 if grades.privacy_leak else 0.0)
        results[model]["batna_leaked"].append(1.0 if grades.batna_leaked else 0.0)

    aggregated: dict[str, dict[str, float]] = {}
    for model, dims in results.items():
        aggregated[model] = {
            dim: sum(vals) / len(vals) if vals else 0.0
            for dim, vals in dims.items()
        }
    return aggregated


def aggregate_by_domain(episodes: list[EpisodeLog]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ep in episodes:
        scenario_id = ep.scenario_id
        domain = _infer_domain(scenario_id)

        results[domain]["principal_utility"].append(ep.grades.principal_utility or 0)
        results[domain]["joint_welfare"].append(ep.grades.joint_welfare or 0)
        results[domain]["coercion_score"].append(ep.grades.coercion_score or 0)

    aggregated: dict[str, dict[str, float]] = {}
    for domain, dims in results.items():
        aggregated[domain] = {
            dim: sum(vals) / len(vals) if vals else 0.0
            for dim, vals in dims.items()
        }
    return aggregated


def _infer_domain(scenario_id: str) -> str:
    parts = scenario_id.split("-")
    if len(parts) >= 2 and parts[1] in ("ecommerce", "saas", "settlement", "ethical"):
        return parts[1]
    if "ecom" in scenario_id:
        return "ecommerce"
    if "saas" in scenario_id or "procure" in scenario_id:
        return "saas_procurement"
    if "settle" in scenario_id:
        return "settlement"
    if "ethic" in scenario_id:
        return "ethical_business"
    return "ecommerce"
