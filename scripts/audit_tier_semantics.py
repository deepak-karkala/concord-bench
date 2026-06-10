"""Audit Concord tier semantics as orthogonal factors rather than a validated scale.

Usage:
    uv run python scripts/audit_tier_semantics.py
    uv run python scripts/audit_tier_semantics.py --seed-dir src/concord/data/seed_yamls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concord.data.seed_audit import audit_tier_semantics

DEFAULT_SEED_DIR = Path(__file__).resolve().parents[1] / "src/concord/data/seed_yamls"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Concord tier semantics")
    parser.add_argument(
        "--seed-dir",
        default=str(DEFAULT_SEED_DIR),
        help="Seed YAML root to audit",
    )
    args = parser.parse_args()

    report = audit_tier_semantics(Path(args.seed_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
