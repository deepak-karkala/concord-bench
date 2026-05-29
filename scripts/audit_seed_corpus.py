"""Audit the hand-authored seed corpus used for Task 20 scenario generation.

Usage:
    uv run python scripts/audit_seed_corpus.py
    uv run python scripts/audit_seed_corpus.py --seed-dir src/concord/data/seed_yamls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concord.data.seed_audit import audit_seed_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Concord seed-tier coverage")
    parser.add_argument(
        "--seed-dir",
        default="src/concord/data/seed_yamls",
        help="Seed YAML root to audit",
    )
    args = parser.parse_args()

    report = audit_seed_corpus(Path(args.seed_dir))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
