from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EpisodeCandidate:
    transcript_id: str
    scenario_id: str
    domain: str
    culture: str
    buyer_model: str
    source_run: str
    source_episode_path: Path
    source_scenario_path: Path
    transcript: list[dict[str, Any]]
    deal: dict[str, Any] | None
    grades: dict[str, Any]
    buyer_private_context: dict[str, Any]
    seller_private_context: dict[str, Any]
    forbidden_claims: list[str]
    scenario_metadata: dict[str, Any]
    selection_tags: tuple[str, ...]


def load_episode_candidates(
    *,
    source_runs: list[tuple[str, Path]],
    scenarios_dir: Path,
) -> list[EpisodeCandidate]:
    scenario_index = _load_scenarios(scenarios_dir)
    candidates: list[EpisodeCandidate] = []
    for source_run, results_dir in source_runs:
        for model_dir in sorted(results_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            buyer_model = model_dir.name
            for episode_path in sorted(model_dir.glob("*.json")):
                payload = json.loads(episode_path.read_text())
                if "scenario_id" not in payload or "turns" not in payload:
                    continue
                scenario_id = str(payload["scenario_id"])
                scenario = scenario_index[scenario_id]
                grades = payload.get("grades") or {}
                transcript = payload.get("turns") or []
                deal = payload.get("deal")
                selection_tags = tuple(
                    sorted(
                        _derive_selection_tags(
                            scenario=scenario,
                            grades=grades,
                            transcript=transcript,
                            buyer_model=buyer_model,
                        )
                    )
                )
                candidates.append(
                    EpisodeCandidate(
                        transcript_id=f"{source_run}::{buyer_model}::{episode_path.stem}",
                        scenario_id=scenario_id,
                        domain=str(scenario["domain"]),
                        culture=str(scenario.get("culture", "US")),
                        buyer_model=buyer_model,
                        source_run=source_run,
                        source_episode_path=episode_path,
                        source_scenario_path=scenario["__path__"],
                        transcript=transcript,
                        deal=deal,
                        grades=grades,
                        buyer_private_context=scenario["buyer_context"],
                        seller_private_context=scenario["seller_context"],
                        forbidden_claims=list(scenario.get("forbidden_claims") or []),
                        scenario_metadata=dict(scenario.get("metadata") or {}),
                        selection_tags=selection_tags,
                    )
                )
    return candidates


def select_validation_corpus(
    candidates: list[EpisodeCandidate],
    *,
    target_count: int = 150,
) -> list[EpisodeCandidate]:
    if target_count <= 0:
        return []

    domain_targets = _domain_targets(candidates, target_count)
    selected: list[EpisodeCandidate] = []
    selected_ids: set[str] = set()
    selected_scenarios: set[str] = set()
    tag_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    model_counts: Counter[tuple[str, str]] = Counter()

    grouped_by_domain: dict[str, list[EpisodeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped_by_domain[candidate.domain].append(candidate)

    for domain, target in domain_targets.items():
        while domain_counts[domain] < target:
            candidate = _pick_best_candidate(
                grouped_by_domain[domain],
                selected_ids=selected_ids,
                selected_scenarios=selected_scenarios,
                tag_counts=tag_counts,
                domain=domain,
                model_counts=model_counts,
            )
            if candidate is None:
                break
            _record_selection(
                candidate,
                selected,
                selected_ids,
                selected_scenarios,
                tag_counts,
                domain_counts,
                model_counts,
            )

    while len(selected) < min(target_count, len(candidates)):
        candidate = _pick_best_candidate(
            candidates,
            selected_ids=selected_ids,
            selected_scenarios=selected_scenarios,
            tag_counts=tag_counts,
            domain=None,
            model_counts=model_counts,
        )
        if candidate is None:
            break
        _record_selection(
            candidate,
            selected,
            selected_ids,
            selected_scenarios,
            tag_counts,
            domain_counts,
            model_counts,
        )

    return selected


def build_validation_record(candidate: EpisodeCandidate) -> dict[str, Any]:
    grades = candidate.grades
    false_claims = grades.get("false_claims") or []
    forbidden_claim_violations = grades.get("forbidden_claim_violations") or []
    hard_constraint_violations = grades.get("hard_constraint_violations") or []
    return {
        "transcript_id": candidate.transcript_id,
        "domain": candidate.domain,
        "culture": candidate.culture,
        "transcript": candidate.transcript,
        "labels": {
            "coercion_level": float(grades.get("coercion_score") or 0.0),
            "privacy_leak": bool(grades.get("privacy_leak")),
            "walk_away_correct": grades.get("walk_away_correct"),
            "has_false_claims": bool(false_claims or forbidden_claim_violations),
            "hard_constraint_violation": bool(hard_constraint_violations),
        },
        "validation_context": {
            "buyer_private_context": candidate.buyer_private_context,
            "seller_private_context": candidate.seller_private_context,
            "deal": candidate.deal,
            "buyer_engaged": _buyer_engaged(candidate.transcript, grades),
            "zopa_exists": _zopa_exists(candidate),
            "forbidden_claims": candidate.forbidden_claims,
            "annotation_source": "curated_episode_artifact_prefill",
            "annotation_notes": [
                f"source_run={candidate.source_run}",
                f"source_episode_path={candidate.source_episode_path}",
                f"source_scenario_path={candidate.source_scenario_path}",
                "Auto-prefilled from episode grades and scenario YAML.",
                "Human or expert adjudication is still required before this corpus is used as validation evidence.",
            ],
        },
    }


def build_corpus_manifest(selected: list[EpisodeCandidate], *, target_count: int) -> dict[str, Any]:
    domain_counts = Counter(candidate.domain for candidate in selected)
    culture_counts = Counter(candidate.culture for candidate in selected)
    source_run_counts = Counter(candidate.source_run for candidate in selected)
    buyer_model_counts = Counter(candidate.buyer_model for candidate in selected)
    tag_counts = Counter(tag for candidate in selected for tag in candidate.selection_tags)
    return {
        "status": "active_adjudication_queue",
        "target_count": target_count,
        "selected_count": len(selected),
        "domains": dict(sorted(domain_counts.items())),
        "cultures": dict(sorted(culture_counts.items())),
        "source_runs": dict(sorted(source_run_counts.items())),
        "buyer_models": dict(sorted(buyer_model_counts.items())),
        "selection_tags": dict(sorted(tag_counts.items())),
        "records": [
            {
                "transcript_id": candidate.transcript_id,
                "scenario_id": candidate.scenario_id,
                "domain": candidate.domain,
                "culture": candidate.culture,
                "buyer_model": candidate.buyer_model,
                "source_run": candidate.source_run,
                "source_episode_path": str(candidate.source_episode_path),
                "source_scenario_path": str(candidate.source_scenario_path),
                "selection_tags": list(candidate.selection_tags),
            }
            for candidate in selected
        ],
        "limitations": [
            "Labels are auto-prefilled from episode artifacts and scenario YAML.",
            "This corpus is an adjudication queue, not a completed human-labeled validation study.",
            "Do not use this corpus alone to promote semantic metrics.",
        ],
    }


def write_validation_corpus(
    *,
    selected: list[EpisodeCandidate],
    output_dir: Path,
    target_count: int,
) -> dict[str, Any]:
    records = [build_validation_record(candidate) for candidate in selected]
    manifest = build_corpus_manifest(selected, target_count=target_count)
    write_validation_records(records=records, manifest=manifest, output_dir=output_dir)
    return manifest


def write_validation_records(
    *,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.jsonl"):
        existing.unlink()
    for index, record in enumerate(records, start=1):
        file_path = output_dir / f"candidate-{index:03d}.jsonl"
        file_path.write_text(json.dumps(record) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _pick_best_candidate(
    candidates: list[EpisodeCandidate],
    *,
    selected_ids: set[str],
    selected_scenarios: set[str],
    tag_counts: Counter[str],
    domain: str | None,
    model_counts: Counter[tuple[str, str]],
) -> EpisodeCandidate | None:
    desired_tag_mins = {
        "protocol_failure": 12,
        "passive_baseline": 12,
        "privacy_positive": 12,
        "false_claim_positive": 12,
        "hard_constraint_positive": 12,
        "walk_away_episode": 20,
        "no_zopa": 20,
        "multi_issue": 60,
        "bright_line": 40,
    }
    best: EpisodeCandidate | None = None
    best_score: tuple[float, ...] | None = None
    for candidate in candidates:
        if candidate.transcript_id in selected_ids:
            continue
        if candidate.scenario_id in selected_scenarios:
            continue
        score = 0.0
        score += 3.0
        for tag in candidate.selection_tags:
            desired = desired_tag_mins.get(tag)
            if desired is not None and tag_counts[tag] < desired:
                score += 5.0
            score += max(0.0, 2.0 - 0.1 * tag_counts[tag])
        domain_model_key = (candidate.domain, candidate.buyer_model)
        score += max(0.0, 3.0 - 0.5 * model_counts[domain_model_key])
        if domain is not None and candidate.domain != domain:
            score -= 100.0
        tie_break = (
            score,
            1.0 if candidate.source_run == "pre10h_validation_lowcost" else 0.0,
            1.0 if "protocol_failure" in candidate.selection_tags else 0.0,
            1.0 if "bright_line" in candidate.selection_tags else 0.0,
            -float(candidate.source_episode_path.stat().st_size),
        )
        if best_score is None or tie_break > best_score:
            best_score = tie_break
            best = candidate
    return best


def _record_selection(
    candidate: EpisodeCandidate,
    selected: list[EpisodeCandidate],
    selected_ids: set[str],
    selected_scenarios: set[str],
    tag_counts: Counter[str],
    domain_counts: Counter[str],
    model_counts: Counter[tuple[str, str]],
) -> None:
    selected.append(candidate)
    selected_ids.add(candidate.transcript_id)
    selected_scenarios.add(candidate.scenario_id)
    domain_counts[candidate.domain] += 1
    model_counts[(candidate.domain, candidate.buyer_model)] += 1
    for tag in candidate.selection_tags:
        tag_counts[tag] += 1


def _domain_targets(candidates: list[EpisodeCandidate], target_count: int) -> dict[str, int]:
    domains = sorted({candidate.domain for candidate in candidates})
    base = target_count // len(domains)
    remainder = target_count % len(domains)
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(domains)
    }


def _load_scenarios(scenarios_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for scenario_path in sorted(scenarios_dir.rglob("*.yaml")):
        payload = yaml.safe_load(scenario_path.read_text())
        payload["__path__"] = scenario_path
        index[str(payload["id"])] = payload
    return index


def _derive_selection_tags(
    *,
    scenario: dict[str, Any],
    grades: dict[str, Any],
    transcript: list[dict[str, Any]],
    buyer_model: str,
) -> set[str]:
    tags = {str(scenario["domain"]), str(scenario.get("culture", "US")).lower()}
    metadata = scenario.get("metadata") or {}
    if (metadata.get("difficulty_tier") or 0) >= 3 or metadata.get("bright_line"):
        tags.add("bright_line")
    if _zopa_exists_from_scenario(scenario) is False:
        tags.add("no_zopa")
    if _is_multi_issue(scenario):
        tags.add("multi_issue")
    if grades.get("privacy_leak"):
        tags.add("privacy_positive")
    if (grades.get("false_claims") or []) or (grades.get("forbidden_claim_violations") or []):
        tags.add("false_claim_positive")
    if grades.get("hard_constraint_violations"):
        tags.add("hard_constraint_positive")
    if grades.get("impasse_outcome") == "protocol_failure" or _protocol_failure(transcript):
        tags.add("protocol_failure")
    if buyer_model in {"always_walk_away", "accept_first_valid", "constraint_first_cautious", "price_only_rational"}:
        tags.add("scripted_baseline")
    if buyer_model == "always_walk_away":
        tags.add("passive_baseline")
    if any(turn.get("agent") == "buyer" and turn.get("action_type") == "walk_away" for turn in transcript):
        tags.add("walk_away_episode")
    return tags


def _zopa_exists(candidate: EpisodeCandidate) -> bool | None:
    scenario = {
        "buyer_context": candidate.buyer_private_context,
        "seller_context": candidate.seller_private_context,
    }
    return _zopa_exists_from_scenario(scenario)


def _zopa_exists_from_scenario(scenario: dict[str, Any]) -> bool | None:
    buyer_reserve = (scenario.get("buyer_context") or {}).get("reserve_price")
    seller_reserve = (scenario.get("seller_context") or {}).get("reserve_price")
    if buyer_reserve is None or seller_reserve is None:
        return None
    return float(buyer_reserve) >= float(seller_reserve)


def _is_multi_issue(scenario: dict[str, Any]) -> bool:
    schema = scenario.get("deal_schema") or {}
    return len(schema.keys()) >= 4


def _protocol_failure(transcript: list[dict[str, Any]]) -> bool:
    buyer_turns = [turn for turn in transcript if turn.get("agent") == "buyer"]
    if not buyer_turns:
        return True
    return all(
        (
            not str(turn.get("content", "")).strip()
            and turn.get("action_type") == "message"
        )
        for turn in buyer_turns
    )


def _buyer_engaged(transcript: list[dict[str, Any]], grades: dict[str, Any]) -> bool:
    engagement = grades.get("engagement_metrics")
    if isinstance(engagement, dict) and "buyer_engaged" in engagement:
        return bool(engagement["buyer_engaged"])

    buyer_turns = [turn for turn in transcript if turn.get("agent") == "buyer"]
    for turn in buyer_turns:
        action_type = turn.get("action_type")
        content = str(turn.get("content", "")).strip()
        if action_type in {"offer", "accept", "walk_away"}:
            return True
        if content:
            return True
    return False
