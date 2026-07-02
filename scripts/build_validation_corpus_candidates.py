"""Build a Phase 1 expanded validation-corpus adjudication queue from saved episode artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from concord.analysis.validation_corpus import (
    build_corpus_manifest,
    build_validation_record,
    load_episode_candidates,
    select_validation_corpus,
    write_validation_records,
)
from concord.graders.validation_study import load_validation_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an expanded validation-corpus adjudication queue from saved episode artifacts",
    )
    parser.add_argument(
        "--scenarios-dir",
        default="outputs/phase5_powered_rerun/scenarios",
        help="Scenario root containing YAML files keyed by scenario_id",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures/validation_corpus_expanded",
        help="Directory where curated validation records and manifest will be written",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=150,
        help="Number of candidate transcripts to curate",
    )
    parser.add_argument(
        "--source-run",
        action="append",
        default=[],
        help=(
            "Source run spec in the form label=path. "
            "May be provided multiple times. Defaults to pre10h_validation_lowcost and phase5_full_baseline."
        ),
    )
    parser.add_argument(
        "--base-transcripts",
        default="tests/fixtures/calibration_transcripts",
        help="Existing validation transcript directory to preserve as locked seed records",
    )
    parser.add_argument(
        "--base-annotations",
        default="outputs/phase2_validation/bootstrap_annotations.jsonl",
        help="Optional sidecar annotations for the locked seed records",
    )
    args = parser.parse_args()

    locked_records = load_validation_records(
        Path(args.base_transcripts),
        annotations_path=Path(args.base_annotations) if args.base_annotations else None,
    )
    locked_count = min(len(locked_records), args.target_count)
    source_runs = _parse_source_runs(args.source_run)
    candidates = load_episode_candidates(
        source_runs=source_runs,
        scenarios_dir=Path(args.scenarios_dir),
    )
    selected = select_validation_corpus(
        candidates,
        target_count=max(args.target_count - locked_count, 0),
    )
    selected_records = [build_validation_record(candidate) for candidate in selected]
    manifest = build_corpus_manifest(selected, target_count=args.target_count)
    manifest["status"] = "active_adjudication_queue"
    manifest["locked_seed_records"] = locked_count
    manifest["selected_from_episode_artifacts"] = len(selected_records)
    manifest["source_runs"] = dict(
        sorted(Counter(candidate.source_run for candidate in selected).items())
    )
    manifest["limitations"] = [
        "The first tranche comes from the existing 50-record calibration corpus with bootstrap-backed hidden-context overlays.",
        "The remaining tranche is auto-prefilled from saved episode artifacts and scenario YAML.",
        "Human or expert adjudication is still required before using this corpus as validation evidence.",
    ]
    records = locked_records[:locked_count] + selected_records
    manifest["selected_count"] = len(records)
    manifest["domains"] = dict(
        sorted(Counter(str(record["domain"]) for record in records).items())
    )
    manifest["cultures"] = dict(
        sorted(Counter(str(record["culture"]) for record in records).items())
    )
    manifest["label_positive_counts"] = {
        "privacy_leak": sum(1 for record in records if (record.get("labels") or {}).get("privacy_leak")),
        "has_false_claims": sum(1 for record in records if (record.get("labels") or {}).get("has_false_claims")),
        "hard_constraint_violation": sum(
            1 for record in records if (record.get("labels") or {}).get("hard_constraint_violation")
        ),
        "walk_away_correct": sum(
            1 for record in records if (record.get("labels") or {}).get("walk_away_correct") is True
        ),
        "coercion_nonzero": sum(
            1 for record in records if float((record.get("labels") or {}).get("coercion_level") or 0.0) > 0.0
        ),
    }
    write_validation_records(
        records=records,
        manifest=manifest,
        output_dir=Path(args.output_dir),
    )

    print(f"Curated {manifest['selected_count']} validation-corpus candidates into {args.output_dir}")
    print(f"Domains: {manifest['domains']}")
    print(f"Selection tags: {manifest['selection_tags']}")


def _parse_source_runs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    if not raw_specs:
        return [
            (
                "pre10h_validation_lowcost",
                Path("outputs/pre10h_validation_lowcost/main_rerun/results"),
            ),
            (
                "phase5_full_baseline",
                Path("outputs/phase5_powered_rerun/full_baseline_panel"),
            ),
        ]

    source_runs: list[tuple[str, Path]] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --source-run spec: {spec}")
        label, path = spec.split("=", 1)
        source_runs.append((label.strip(), Path(path.strip())))
    return source_runs


if __name__ == "__main__":
    main()
