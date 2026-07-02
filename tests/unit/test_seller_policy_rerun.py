import json
from pathlib import Path

from concord.analysis.seller_policy_rerun import (
    build_phase4_seller_policy_plan,
    load_confirmed_model_ids,
)


def test_load_confirmed_model_ids_reads_confirmed_slots(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "slots": [
                    {"status": "confirmed", "model_id": "openrouter/anthropic/model-a"},
                    {"status": "fallback_required", "model_id": None},
                    {"status": "confirmed", "model_id": "openrouter/openai/model-b"},
                ]
            }
        )
        + "\n"
    )

    assert load_confirmed_model_ids(manifest_path) == [
        "openrouter/anthropic/model-a",
        "openrouter/openai/model-b",
    ]


def test_build_phase4_seller_policy_plan_builds_three_seller_runs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "slots": [
                    {"status": "confirmed", "model_id": "openrouter/anthropic/model-a"},
                    {"status": "confirmed", "model_id": "openrouter/openai/model-b"},
                ]
            }
        )
        + "\n"
    )
    scenarios_dir = tmp_path / "scenarios"
    output_root = tmp_path / "phase4"
    outputs_manifest_path = tmp_path / "outputs_manifest.json"
    preregistration_path = tmp_path / "preregistration.json"

    plan = build_phase4_seller_policy_plan(
        manifest_path=manifest_path,
        scenarios_dir=scenarios_dir,
        output_root=output_root,
        outputs_manifest_path=outputs_manifest_path,
        preregistration_path=preregistration_path,
        seeds=[42, 43, 44],
        budget_cap=12.5,
    )

    assert plan["buyer_models"] == [
        "openrouter/anthropic/model-a",
        "openrouter/openai/model-b",
    ]
    assert plan["repeated_runs_per_scenario"] == 3
    assert [run["seller"] for run in plan["seller_runs"]] == [
        "honest_cooperative",
        "honest_hardball",
        "deceptive_or_pressure",
    ]
    assert plan["budget_cap"] == 12.5
    first = plan["seller_runs"][0]
    assert "--models" in first["run_batch_command"]
    assert "--seller" in first["run_batch_command"]
    assert "--budget-cap" in first["run_batch_command"]
    assert "--outputs-manifest" in first["report_command"]
    assert "--preregistration" in first["report_command"]
