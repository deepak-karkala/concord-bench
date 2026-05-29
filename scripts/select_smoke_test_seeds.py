"""Select a representative smoke test seed set for Phase 1 pipeline validation.

Usage:
    uv run python scripts/select_smoke_test_seeds.py
    uv run python scripts/select_smoke_test_seeds.py --output outputs/my_smoke_test/scenarios
"""

from __future__ import annotations

import argparse
from pathlib import Path

from concord.data.smoke_test_selection import select_smoke_test_seeds


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
