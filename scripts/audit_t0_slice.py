"""Audit the Task 20 T0 sanity slice.

Usage:
    uv run python scripts/audit_t0_slice.py
    uv run python scripts/audit_t0_slice.py --seed-dir src/concord/data/seed_yamls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concord.data.seed_audit import audit_t0_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Concord T0 seed coverage")
    parser.add_argument(
        "--seed-dir",
        default="src/concord/data/seed_yamls",
        help="Seed YAML root to audit",
    )
    args = parser.parse_args()

    report = audit_t0_slice(Path(args.seed_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
