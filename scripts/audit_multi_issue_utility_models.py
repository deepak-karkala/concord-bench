"""Audit explicit multi-issue utility-model coverage in the seed corpus.

Usage:
    concord/.venv/bin/python concord/scripts/audit_multi_issue_utility_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

from concord.data.seed_audit import DEFAULT_SEED_DIR, audit_multi_issue_utility_models


def main() -> None:
    report = audit_multi_issue_utility_models(Path(DEFAULT_SEED_DIR))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
