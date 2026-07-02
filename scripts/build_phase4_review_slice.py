"""Build a bounded repaired-runtime seller-policy review slice for Phase 4."""

from __future__ import annotations

import argparse
from pathlib import Path

from concord.analysis.seller_policy_review_slice import (
    load_seller_policy_slice_inventory,
    select_phase4_review_slice,
    write_phase4_review_slice,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bounded Phase 4 seller-policy review slice",
    )
    parser.add_argument(
        "--scenarios-dir",
        default="outputs/phase5_powered_rerun/scenarios",
        help="Source scenario directory",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/phase4_seller_policy_rerun/review_slice",
        help="Output root for the review slice scenarios and manifest",
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=6,
        help="Number of scenarios to select per domain",
    )
    args = parser.parse_args()

    inventory = load_seller_policy_slice_inventory(Path(args.scenarios_dir))
    selected = select_phase4_review_slice(inventory, per_domain=args.per_domain)
    manifest_path = write_phase4_review_slice(selected, output_root=Path(args.output_root))
    print(f"Wrote Phase 4 review slice manifest to {manifest_path}")
    print(f"Selected {len(selected)} scenarios across {len({item.domain for item in selected})} domains")


if __name__ == "__main__":
    main()
