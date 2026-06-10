"""Backfill explicit multi-issue utility models into Concord seed YAMLs.

Usage:
    concord/.venv/bin/python concord/scripts/apply_multi_issue_utility_models.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from concord.data.multi_issue_utility_models import build_issue_utility_models
from concord.data.seed_audit import DEFAULT_SEED_DIR, _is_multi_issue


def main() -> None:
    for path in sorted(DEFAULT_SEED_DIR.rglob("*.yaml")):
        with path.open() as f:
            document = yaml.safe_load(f)

        if not _is_multi_issue(document):
            continue

        buyer_model, seller_model = build_issue_utility_models(document)
        document.setdefault("buyer_context", {})["price_weight"] = buyer_model["price_weight"]
        document.setdefault("buyer_context", {})["issue_utilities"] = buyer_model["issue_utilities"]
        document.setdefault("seller_context", {})["price_weight"] = seller_model["price_weight"]
        document.setdefault("seller_context", {})["issue_utilities"] = seller_model["issue_utilities"]

        with path.open("w") as f:
            yaml.safe_dump(document, f, sort_keys=False)


if __name__ == "__main__":
    main()
