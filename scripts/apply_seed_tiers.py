"""Backfill deterministic difficulty tiers into the hand-authored seed corpus.

Usage:
    uv run python scripts/apply_seed_tiers.py
    uv run python scripts/apply_seed_tiers.py --seed-dir src/concord/data/seed_yamls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from concord.data.seed_tiering import apply_tier_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic seed difficulty tiers")
    parser.add_argument(
        "--seed-dir",
        default="src/concord/data/seed_yamls",
        help="Seed YAML root to rewrite in place",
    )
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    for path in sorted(seed_dir.rglob("*.yaml")):
        with path.open() as f:
            document = yaml.safe_load(f)
        updated = apply_tier_metadata(document)
        with path.open("w") as f:
            yaml.safe_dump(updated, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
