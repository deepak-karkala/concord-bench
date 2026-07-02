"""Generate repeated-run reliability reports for powered rerun outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from concord.analysis.bootstrap_ci import bootstrap_ci
from concord.analysis.preregistration import load_preregistration


def load_episodes_by_scenario(model_dir: Path) -> dict[str, list[dict]]:
    by_scenario: dict[str, list[dict]] = {}
    for ep_file in sorted(model_dir.glob("*.json")):
        if ep_file.name.endswith("_grades.json"):
            continue
        try:
            episode = json.loads(ep_file.read_text())
            if not isinstance(episode, dict):
                continue
            scenario_id = episode.get("scenario_id")
            if not scenario_id:
                continue
            grade_file = ep_file.with_suffix("").with_name(ep_file.stem + "_grades.json")
            if grade_file.exists():
                try:
                    episode["_grades"] = json.loads(grade_file.read_text())
                except Exception:
                    pass
            by_scenario.setdefault(str(scenario_id), []).append(episode)
        except Exception as exc:
            print(f"  Warning: failed to load {ep_file}: {exc}")
    return by_scenario


def _metric_values(episodes: list[dict], metric: str) -> list[float]:
    values: list[float] = []
    for episode in episodes:
        grades = episode.get("_grades") if isinstance(episode.get("_grades"), dict) else episode.get("grades", {})
        if not isinstance(grades, dict):
            grades = {}
        if metric == "deal_rate":
            values.append(1.0 if episode.get("deal") is not None else 0.0)
            continue
        value = grades.get(metric)
        if value is None:
            continue
        values.append(float(value))
    return values


def compute_within_scenario_variance(by_scenario: dict[str, list[dict]], metric: str) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for scenario_id, episodes in sorted(by_scenario.items()):
        values = _metric_values(episodes, metric)
        report[scenario_id] = {
            "n_runs": len(episodes),
            "n_with_metric": len(values),
            "mean": statistics.mean(values) if values else None,
            "variance": statistics.pvariance(values) if len(values) >= 2 else None,
            "std": statistics.stdev(values) if len(values) >= 2 else None,
        }
    return report


def _scenario_means(by_scenario: dict[str, list[dict]], metric: str) -> list[float]:
    values: list[float] = []
    for episodes in by_scenario.values():
        metric_values = _metric_values(episodes, metric)
        if metric_values:
            values.append(sum(metric_values) / len(metric_values))
    return values


def _bootstrap_summary(values: list[float]) -> dict | None:
    if not values:
        return None
    if len(values) == 1:
        return {
            "mean": values[0],
            "n": 1,
            "ci95": {"lower": values[0], "upper": values[0], "confidence": 0.95},
        }
    ci = bootstrap_ci(values)
    return {
        "mean": sum(values) / len(values),
        "n": len(values),
        "ci95": {
            "lower": ci.lower,
            "upper": ci.upper,
            "confidence": ci.confidence,
        },
    }


def generate_reliability_report(
    results_dir: Path,
    output_dir: Path,
    *,
    preregistration_path: Path | None = None,
) -> dict:
    preregistration = load_preregistration(preregistration_path) if preregistration_path else None
    required_repeated_runs = (
        preregistration.required_repeated_runs if preregistration is not None else 3
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "__meta__": {
            "results_dir": str(results_dir),
            "output_dir": str(output_dir),
            "preregistration_path": str(preregistration_path) if preregistration_path else None,
            "required_repeated_runs": required_repeated_runs,
        }
    }

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        by_scenario = load_episodes_by_scenario(model_dir)
        if not by_scenario:
            continue

        within_scenario = {
            "principal_utility": compute_within_scenario_variance(by_scenario, "principal_utility"),
            "deal_rate": compute_within_scenario_variance(by_scenario, "deal_rate"),
        }
        scenario_bootstrap = {
            "principal_utility": _bootstrap_summary(_scenario_means(by_scenario, "principal_utility")),
            "deal_rate": _bootstrap_summary(_scenario_means(by_scenario, "deal_rate")),
        }
        scenario_run_counts = Counter(len(episodes) for episodes in by_scenario.values())
        minimum_runs_satisfied = all(
            len(episodes) >= required_repeated_runs for episodes in by_scenario.values()
        )
        report[model_dir.name] = {
            "scenario_count": len(by_scenario),
            "minimum_runs_satisfied": minimum_runs_satisfied,
            "required_repeated_runs": required_repeated_runs,
            "run_count_histogram": dict(sorted(scenario_run_counts.items())),
            "within_scenario_variance": within_scenario,
            "scenario_bootstrap_uncertainty": scenario_bootstrap,
        }

    (output_dir / "reliability_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated-run reliability report")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args()

    generate_reliability_report(
        args.results_dir,
        args.output,
        preregistration_path=args.preregistration,
    )


if __name__ == "__main__":
    main()
