"""Plan or execute the repaired-runtime seller-policy rerun for Phase 4."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from concord.analysis.seller_policy_rerun import (
    DEFAULT_PHASE4_SELLERS,
    build_phase4_seller_policy_plan,
    write_phase4_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute the Phase 4 repaired-runtime seller-policy rerun",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/model_panels/phase4_review_fast_manifest.json",
        help="Frozen buyer panel manifest for the bounded seller-policy rerun",
    )
    parser.add_argument(
        "--scenarios-dir",
        default="outputs/phase4_seller_policy_rerun/review_slice/scenarios",
        help="Scenario directory for the seller-policy rerun",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/phase4_seller_policy_rerun",
        help="Root output directory for per-seller results and reports",
    )
    parser.add_argument(
        "--outputs-manifest",
        default="outputs/outputs_manifest.json",
        help="Outputs manifest used to approve canonical report sources",
    )
    parser.add_argument(
        "--preregistration",
        default="docs/preregistration.json",
        help="Preregistration JSON used by report generation",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated repeated-run seeds",
    )
    parser.add_argument(
        "--sellers",
        default=",".join(DEFAULT_PHASE4_SELLERS),
        help="Comma-separated seller policy IDs to run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Optional override for run-batch concurrency",
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=None,
        help="Optional budget cap passed through to each per-seller run-batch invocation",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the batch and report commands after writing the plan",
    )
    args = parser.parse_args()

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    sellers = [token.strip() for token in args.sellers.split(",") if token.strip()]
    plan = build_phase4_seller_policy_plan(
        manifest_path=Path(args.manifest),
        scenarios_dir=Path(args.scenarios_dir),
        output_root=Path(args.output_root),
        outputs_manifest_path=Path(args.outputs_manifest),
        preregistration_path=Path(args.preregistration),
        seeds=seeds,
        sellers=sellers,
        concurrency=args.concurrency,
        budget_cap=args.budget_cap,
    )
    plan_path = write_phase4_plan(plan, Path(args.output_root))
    print(f"Wrote Phase 4 seller-policy plan to {plan_path}")

    for run in plan["seller_runs"]:
        print()
        print(f"Seller: {run['seller']}")
        print("run-batch:")
        print("  " + " ".join(run["run_batch_command"]))
        print("report:")
        print("  " + " ".join(run["report_command"]))

    if not args.execute:
        return

    for run in plan["seller_runs"]:
        _run_command(run["run_batch_command"], cwd=Path.cwd())
        _run_command(run["report_command"], cwd=Path.cwd())


def _run_command(command: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    argv = list(command)
    while argv and "=" in argv[0] and not argv[0].startswith(("-", "/", ".")):
        key, value = argv.pop(0).split("=", 1)
        env[key] = value
    subprocess.run(argv, check=True, cwd=cwd, env=env)


if __name__ == "__main__":
    main()
