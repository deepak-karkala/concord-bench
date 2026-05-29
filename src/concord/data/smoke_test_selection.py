from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from concord.data.seed_audit import _is_multi_issue, _is_no_zopa

SMOKE_TEST_TARGETS: list[tuple[str, int, int]] = [
    ("ecommerce", 0, 3),
    ("ecommerce", 1, 3),
    ("ecommerce", 2, 3),
    ("ecommerce", 3, 5),
    ("saas_procurement", 1, 3),
    ("saas_procurement", 2, 3),
    ("saas_procurement", 3, 5),
    ("settlement", 1, 3),
    ("settlement", 2, 3),
    ("settlement", 3, 5),
    ("ethical_business", 1, 3),
    ("ethical_business", 2, 3),
    ("ethical_business", 3, 5),
]
SELECTION_SUMMARY_FILENAME = "selection_summary.json"
MIN_NO_ZOPA = 3
MIN_MULTI_ISSUE = 5


class SelectionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SeedRecord:
    path: Path
    data: dict[str, Any]


def _load_seed_records(src: Path) -> list[SeedRecord]:
    return [
        SeedRecord(path=path, data=yaml.safe_load(path.read_text()))
        for path in sorted(src.rglob("*.yaml"))
    ]


def _get_tier(data: dict[str, Any]) -> int:
    return int(data.get("metadata", {}).get("difficulty_tier", 1))


def _get_domain(data: dict[str, Any]) -> str:
    return str(data.get("domain", "unknown"))


def _is_galaxy_brain(data: dict[str, Any]) -> bool:
    return data.get("metadata", {}).get("pressure_type") == "galaxy_brain"


def _validate_target_availability(records: list[SeedRecord]) -> None:
    available: Counter[tuple[str, int]] = Counter(
        (_get_domain(record.data), _get_tier(record.data)) for record in records
    )
    missing: list[str] = []
    for domain, tier, count in SMOKE_TEST_TARGETS:
        actual = available[(domain, tier)]
        if actual < count:
            missing.append(
                f"{domain}/T{tier}: need {count}, found {actual}"
            )

    if missing:
        details = "; ".join(missing)
        raise SelectionValidationError(f"Missing required smoke-test coverage: {details}")


def _pick_required_targets(records: list[SeedRecord], seed: int) -> list[SeedRecord]:
    rng = random.Random(seed)
    selected: list[SeedRecord] = []
    selected_names: set[str] = set()

    for domain, tier, count in SMOKE_TEST_TARGETS:
        pool = [
            record
            for record in records
            if _get_domain(record.data) == domain
            and _get_tier(record.data) == tier
            and record.path.name not in selected_names
        ]
        chosen = rng.sample(pool, count)
        selected.extend(chosen)
        selected_names.update(record.path.name for record in chosen)

    return selected


def _ensure_minimum_examples(
    selected: list[SeedRecord],
    records: list[SeedRecord],
    predicate: Any,
    minimum: int,
) -> list[SeedRecord]:
    selected_names = {record.path.name for record in selected}
    current = sum(1 for record in selected if predicate(record.data))
    if current >= minimum:
        return selected

    extras = [
        record
        for record in records
        if predicate(record.data) and record.path.name not in selected_names
    ]
    needed = minimum - current
    if len(extras) < needed:
        raise SelectionValidationError(
            f"Unable to satisfy smoke-test minimum of {minimum} extra examples."
        )

    selected.extend(extras[:needed])
    return selected


def _build_summary(selected: list[SeedRecord], src: Path, out: Path, seed: int) -> dict[str, Any]:
    tier_counts = Counter(str(_get_tier(record.data)) for record in selected)
    domain_counts = Counter(_get_domain(record.data) for record in selected)
    no_zopa_count = sum(1 for record in selected if _is_no_zopa(record.data))
    multi_issue_count = sum(1 for record in selected if _is_multi_issue(record.data))
    galaxy_brain_count = sum(1 for record in selected if _is_galaxy_brain(record.data))

    return {
        "seed": seed,
        "source_dir": str(src),
        "output_dir": str(out),
        "selected_count": len(selected),
        "tier_counts": {key: tier_counts[key] for key in sorted(tier_counts)},
        "domain_counts": {key: domain_counts[key] for key in sorted(domain_counts)},
        "no_zopa_count": no_zopa_count,
        "multi_issue_count": multi_issue_count,
        "galaxy_brain_count": galaxy_brain_count,
        "targets": [
            {"domain": domain, "tier": tier, "count": count}
            for domain, tier, count in SMOKE_TEST_TARGETS
        ],
    }


def select_smoke_test_seeds(src: Path, out: Path, seed: int = 42) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    records = _load_seed_records(src)
    _validate_target_availability(records)

    selected = _pick_required_targets(records, seed)
    selected = _ensure_minimum_examples(selected, records, _is_no_zopa, MIN_NO_ZOPA)
    selected = _ensure_minimum_examples(selected, records, _is_multi_issue, MIN_MULTI_ISSUE)

    for record in selected:
        shutil.copy(record.path, out / record.path.name)

    summary = _build_summary(selected, src=src, out=out, seed=seed)
    summary_path = out / SELECTION_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Found {len(records)} seed files in {src}")
    print(f"\nSmoke test set: {summary['selected_count']} scenarios -> {out}")
    print(f"  Tiers:   {summary['tier_counts']}")
    print(f"  Domains: {summary['domain_counts']}")
    print(f"  No-ZOPA: {summary['no_zopa_count']}")
    print(f"  Multi-issue: {summary['multi_issue_count']}")
    print(f"  Galaxy-brain: {summary['galaxy_brain_count']}")
    print(f"  Summary: {summary_path}")

    return [record.path for record in selected]
