from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from concord.schemas.scenario import Domain, Scenario


@dataclass(frozen=True)
class SellerPolicySliceScenario:
    scenario_id: str
    domain: str
    culture: str
    source_path: str
    zopa_exists: bool
    multi_issue: bool
    difficulty_tier: int | None
    tiering_reason: str | None


def load_seller_policy_slice_inventory(scenarios_dir: Path) -> list[SellerPolicySliceScenario]:
    inventory: list[SellerPolicySliceScenario] = []
    for scenario_path in sorted(scenarios_dir.rglob("*.yaml")):
        payload = yaml.safe_load(scenario_path.read_text())
        scenario = Scenario.model_validate(payload)
        buyer_reserve = scenario.buyer_context.reserve_price
        seller_reserve = scenario.seller_context.reserve_price
        zopa_exists = (
            buyer_reserve is not None
            and seller_reserve is not None
            and float(buyer_reserve) >= float(seller_reserve)
        )
        multi_issue = bool(
            scenario.buyer_context.issue_utilities or scenario.seller_context.issue_utilities
        )
        metadata = scenario.metadata or {}
        difficulty_tier = metadata.get("difficulty_tier")
        inventory.append(
            SellerPolicySliceScenario(
                scenario_id=scenario.id,
                domain=str(scenario.domain),
                culture=scenario.culture,
                source_path=str(scenario_path),
                zopa_exists=zopa_exists,
                multi_issue=multi_issue,
                difficulty_tier=int(difficulty_tier) if isinstance(difficulty_tier, int | float) else None,
                tiering_reason=str(metadata.get("tiering_reason"))
                if metadata.get("tiering_reason") is not None
                else None,
            )
        )
    if not inventory:
        raise ValueError(f"No scenario YAMLs found under {scenarios_dir}")
    return inventory


def select_phase4_review_slice(
    inventory: list[SellerPolicySliceScenario],
    *,
    per_domain: int = 6,
) -> list[SellerPolicySliceScenario]:
    by_domain: dict[str, list[SellerPolicySliceScenario]] = defaultdict(list)
    for item in inventory:
        by_domain[item.domain].append(item)

    selected: list[SellerPolicySliceScenario] = []
    seen_ids: set[str] = set()

    for domain in sorted(str(value) for value in Domain):
        domain_items = sorted(
            by_domain[domain],
            key=lambda item: (
                item.difficulty_tier if item.difficulty_tier is not None else 999,
                item.scenario_id,
            ),
        )
        if not domain_items:
            raise ValueError(f"No scenarios found for domain {domain}")

        domain_selected: list[SellerPolicySliceScenario] = []
        domain_seen: set[str] = set()

        def add_matching(
            predicate,
            *,
            limit: int = 1,
        ) -> None:
            for item in domain_items:
                if len(domain_selected) >= per_domain:
                    return
                if len([selected_item for selected_item in domain_selected if predicate(selected_item)]) >= limit:
                    return
                if item.scenario_id in seen_ids or item.scenario_id in domain_seen:
                    continue
                if not predicate(item):
                    continue
                domain_selected.append(item)
                seen_ids.add(item.scenario_id)
                domain_seen.add(item.scenario_id)

        # Establish the key reviewer-facing coverage first.
        add_matching(lambda item: not item.zopa_exists, limit=2)
        add_matching(lambda item: item.tiering_reason == "multi_issue_tradeoff")
        add_matching(lambda item: item.tiering_reason == "wide_zopa_simple")
        add_matching(lambda item: item.tiering_reason == "walkaway_or_constraint_pressure")

        # Ensure each domain contributes enough multi-issue coverage when available.
        add_matching(lambda item: item.multi_issue, limit=3)

        for item in domain_items:
            if len(domain_selected) >= per_domain:
                break
            if item.scenario_id in seen_ids or item.scenario_id in domain_seen:
                continue
            domain_selected.append(item)
            seen_ids.add(item.scenario_id)
            domain_seen.add(item.scenario_id)

        if len(domain_selected) < per_domain:
            raise ValueError(
                f"Unable to satisfy per-domain selection target for {domain}: "
                f"wanted {per_domain}, got {len(domain_selected)}"
            )
        selected.extend(domain_selected)

    return sorted(selected, key=lambda item: (item.domain, item.scenario_id))


def write_phase4_review_slice(
    selected: list[SellerPolicySliceScenario],
    *,
    output_root: Path,
) -> Path:
    scenarios_out = output_root / "scenarios"
    scenarios_out.mkdir(parents=True, exist_ok=True)

    for item in selected:
        source = Path(item.source_path)
        target = scenarios_out / item.domain / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest = {
        "phase": "phase4_seller_policy_review_slice",
        "selected_count": len(selected),
        "per_domain_target": len(selected) // max(len({item.domain for item in selected}), 1),
        "domains": dict(sorted(Counter(item.domain for item in selected).items())),
        "zopa_counts": {
            "zopa_exists": sum(1 for item in selected if item.zopa_exists),
            "no_zopa": sum(1 for item in selected if not item.zopa_exists),
        },
        "multi_issue_count": sum(1 for item in selected if item.multi_issue),
        "tiering_reasons": dict(
            sorted(
                Counter(item.tiering_reason or "unknown" for item in selected).items()
            )
        ),
        "selected_scenarios": [asdict(item) for item in selected],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
