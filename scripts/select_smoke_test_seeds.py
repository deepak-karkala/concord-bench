"""Select a representative smoke test seed set for Phase 1 pipeline validation.

Usage:
    uv run python scripts/select_smoke_test_seeds.py
    uv run python scripts/select_smoke_test_seeds.py --output outputs/my_smoke_test/scenarios
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_metadata(f: Path) -> dict:
    with f.open() as fh:
        return yaml.safe_load(fh)


def is_no_zopa(d: dict) -> bool:
    bc = d.get("buyer_context", {})
    sc = d.get("seller_context", {})
    return bc.get("reserve_price", 0) < sc.get("reserve_price", 0)


def is_multi_issue(d: dict) -> bool:
    schema = d.get("deal_schema", {})
    multi_fields = {"delivery_days", "payment_terms_days", "contract_length_months",
                    "onboarding_support_hours", "payment_structure", "payment_timeline_days",
                    "audit_frequency_months", "transition_period_months"}
    return bool(multi_fields & set(schema.keys()))


def get_tier(d: dict) -> int:
    return d.get("metadata", {}).get("difficulty_tier", 1)


def get_domain(d: dict) -> str:
    return d.get("domain", "unknown")


def select_smoke_test_seeds(
    src: Path,
    out: Path,
    seed: int = 42,
) -> list[Path]:
    random.seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    all_seeds = list(src.rglob("*.yaml"))
    print(f"Found {len(all_seeds)} seed files in {src}")

    data_map: dict[Path, dict] = {f: load_metadata(f) for f in all_seeds}

    # Target composition: (domain, tier) -> count
    targets: list[tuple[str, int, int]] = [
        ("ecommerce",        0, 3),
        ("ecommerce",        1, 3),
        ("ecommerce",        2, 3),
        ("ecommerce",        3, 5),
        ("saas_procurement", 1, 3),
        ("saas_procurement", 2, 3),
        ("saas_procurement", 3, 5),
        ("settlement",       1, 3),
        ("settlement",       2, 3),
        ("settlement",       3, 5),
        ("ethical_business", 1, 3),
        ("ethical_business", 2, 3),
        ("ethical_business", 3, 5),
    ]

    selected: list[Path] = []
    selected_set: set[str] = set()

    for domain, tier, count in targets:
        pool = [f for f, d in data_map.items()
                if get_domain(d) == domain and get_tier(d) == tier
                and f.name not in selected_set]
        chosen = random.sample(pool, min(count, len(pool)))
        selected.extend(chosen)
        selected_set.update(f.name for f in chosen)

    # Add no-ZOPA seeds (at least 3)
    no_zopa = [f for f, d in data_map.items()
               if is_no_zopa(d) and f.name not in selected_set]
    for f in no_zopa[:5]:
        selected.append(f)
        selected_set.add(f.name)

    # Add multi-issue seeds (at least 5)
    multi = [f for f, d in data_map.items()
             if is_multi_issue(d) and f.name not in selected_set]
    for f in multi[:8]:
        selected.append(f)
        selected_set.add(f.name)

    # Copy selected seeds
    for f in selected:
        shutil.copy(f, out / f.name)

    # Print summary
    tiers_count: dict[int, int] = defaultdict(int)
    domains_count: dict[str, int] = defaultdict(int)
    no_zopa_count = 0
    multi_count = 0
    gb_count = 0

    for f in selected:
        d = data_map[f]
        tiers_count[get_tier(d)] += 1
        domains_count[get_domain(d)] += 1
        if is_no_zopa(d):
            no_zopa_count += 1
        if is_multi_issue(d):
            multi_count += 1
        if d.get("metadata", {}).get("pressure_type") == "galaxy_brain":
            gb_count += 1

    print(f"\nSmoke test set: {len(selected)} scenarios → {out}")
    print(f"  Tiers:   {dict(tiers_count)}")
    print(f"  Domains: {dict(domains_count)}")
    print(f"  No-ZOPA: {no_zopa_count}")
    print(f"  Multi-issue: {multi_count}")
    print(f"  Galaxy-brain: {gb_count}")

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select smoke test seed set")
    parser.add_argument("--src", default="src/concord/data/seed_yamls",
                        help="Source seed directory")
    parser.add_argument("--output", default="outputs/smoke_test/scenarios",
                        help="Output directory for selected seeds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Seed directory not found: {src}")

    select_smoke_test_seeds(src=src, out=Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
