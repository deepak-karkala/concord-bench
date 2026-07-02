from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PHASE4_SELLERS = (
    "honest_cooperative",
    "honest_hardball",
    "deceptive_or_pressure",
)


def load_confirmed_model_ids(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text())
    slots = payload.get("slots", [])
    models: list[str] = []
    for slot in slots:
        if slot.get("status") != "confirmed":
            continue
        model_id = slot.get("model_id")
        if isinstance(model_id, str) and model_id:
            models.append(model_id)
    if not models:
        raise ValueError(f"No confirmed model IDs found in manifest: {manifest_path}")
    return models


def build_phase4_seller_policy_plan(
    *,
    manifest_path: Path,
    scenarios_dir: Path,
    output_root: Path,
    outputs_manifest_path: Path,
    preregistration_path: Path,
    seeds: list[int],
    sellers: list[str] | tuple[str, ...] = DEFAULT_PHASE4_SELLERS,
    concurrency: int | None = None,
    budget_cap: float | None = None,
) -> dict:
    buyer_models = load_confirmed_model_ids(manifest_path)
    seller_runs = []
    for seller in sellers:
        results_dir = output_root / seller
        report_dir = results_dir / "report"
        seller_runs.append(
            {
                "seller": seller,
                "buyer_models": buyer_models,
                "results_dir": str(results_dir),
                "report_dir": str(report_dir),
                "run_batch_command": build_run_batch_command(
                    buyer_models=buyer_models,
                    seller=seller,
                    scenarios_dir=scenarios_dir,
                    output_dir=results_dir,
                    seeds=seeds,
                    manifest_path=manifest_path,
                    concurrency=concurrency,
                    budget_cap=budget_cap,
                ),
                "report_command": build_report_command(
                    results_dir=results_dir,
                    scenarios_dir=scenarios_dir,
                    report_dir=report_dir,
                    outputs_manifest_path=outputs_manifest_path,
                    preregistration_path=preregistration_path,
                ),
            }
        )

    return {
        "phase": "phase4_seller_policy_rerun",
        "status": "planned",
        "manifest_path": str(manifest_path),
        "buyer_models": buyer_models,
        "scenario_source": str(scenarios_dir),
        "output_root": str(output_root),
        "outputs_manifest_path": str(outputs_manifest_path),
        "preregistration_path": str(preregistration_path),
        "repeated_runs_per_scenario": len(seeds),
        "seeds": seeds,
        "budget_cap": budget_cap,
        "seller_runs": seller_runs,
    }


def build_run_batch_command(
    *,
    buyer_models: list[str],
    seller: str,
    scenarios_dir: Path,
    output_dir: Path,
    seeds: list[int],
    manifest_path: Path,
    concurrency: int | None = None,
    budget_cap: float | None = None,
) -> list[str]:
    command = [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "-m",
        "concord.cli",
        "-q",
        "run-batch",
        "--models",
        ",".join(buyer_models),
        "--seller",
        seller,
        "--scenarios",
        str(scenarios_dir),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--model-panel-manifest",
        str(manifest_path),
        "--output",
        str(output_dir),
    ]
    if concurrency is not None:
        command.extend(["--concurrency", str(concurrency)])
    if budget_cap is not None:
        command.extend(["--budget-cap", str(budget_cap)])
    return command


def build_report_command(
    *,
    results_dir: Path,
    scenarios_dir: Path,
    report_dir: Path,
    outputs_manifest_path: Path,
    preregistration_path: Path,
) -> list[str]:
    return [
        "PYTHONPATH=src",
        ".venv/bin/python",
        "scripts/smoke_test_report.py",
        "--results-dir",
        str(results_dir),
        "--scenarios-dir",
        str(scenarios_dir),
        "--output",
        str(report_dir),
        "--outputs-manifest",
        str(outputs_manifest_path),
        "--preregistration",
        str(preregistration_path),
    ]


def write_phase4_plan(plan: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "phase4_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path
