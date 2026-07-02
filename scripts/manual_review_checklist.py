#!/usr/bin/env python3
"""Scaffold and check manual review notes for generated scenarios.

Usage:
    uv run python scripts/manual_review_checklist.py scaffold \
        --scenarios-dir outputs/generated_10h/ \
        --output-dir outputs/generated_10h_manual_review/

    uv run python scripts/manual_review_checklist.py check \
        --review-dir outputs/generated_10h_manual_review/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQUIRED_FIELDS = {
    "reviewer",
    "review_date",
    "structurally_coherent",
    "publication_worthy",
    "domain_realistic",
}


def scaffold_review_files(scenarios_dir: Path, output_dir: Path) -> None:
    """Create one blank review YAML per scenario file in scenarios_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario_file in sorted(scenarios_dir.glob("*.yaml")):
        try:
            scenario = yaml.safe_load(scenario_file.read_text()) or {}
        except Exception:
            scenario = {}
        scenario_id = str(scenario.get("id", scenario_file.stem))
        review_path = output_dir / f"{scenario_id}.yaml"
        if review_path.exists():
            continue
        template = {
            "scenario_id": scenario_id,
            "reviewer": None,
            "review_date": None,
            "structurally_coherent": None,
            "publication_worthy": None,
            "domain_realistic": None,
            "notes": "",
        }
        review_path.write_text(yaml.safe_dump(template, sort_keys=False))


def check_review_completeness(review_dir: Path) -> list[str]:
    """Return scenario IDs whose review note is incomplete."""
    incomplete: list[str] = []
    for review_file in sorted(review_dir.glob("*.yaml")):
        review = yaml.safe_load(review_file.read_text()) or {}
        scenario_id = str(review.get("scenario_id", review_file.stem))
        missing_required = any(review.get(field) is None for field in REQUIRED_FIELDS)
        missing_notes = not str(review.get("notes", "")).strip()
        if missing_required or missing_notes:
            incomplete.append(scenario_id)
    return incomplete


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold and check manual review notes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold_parser = subparsers.add_parser("scaffold", help="Create blank review note files")
    scaffold_parser.add_argument("--scenarios-dir", type=Path, required=True)
    scaffold_parser.add_argument("--output-dir", type=Path, required=True)

    check_parser = subparsers.add_parser("check", help="Check review note completeness")
    check_parser.add_argument("--review-dir", type=Path, required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "scaffold":
        scaffold_review_files(args.scenarios_dir, args.output_dir)
        print(f"Scaffolded review files in {args.output_dir}")
        return
    if args.command == "check":
        missing = check_review_completeness(args.review_dir)
        if missing:
            print(f"Incomplete reviews ({len(missing)}):")
            for scenario_id in missing:
                print(f"  - {scenario_id}")
            raise SystemExit(1)
        print("All reviews complete.")


if __name__ == "__main__":
    main()
