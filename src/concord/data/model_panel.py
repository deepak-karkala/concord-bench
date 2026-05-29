import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class PanelSlotSpec:
    slot_name: str
    family: str
    intended_role: str
    backend: str
    candidate_slugs: tuple[str, ...] = ()
    fallback_model_id: str | None = None


PANEL_SPECS: dict[str, tuple[PanelSlotSpec, ...]] = {
    "phase1": (
        PanelSlotSpec(
            slot_name="baseline_greedy",
            family="scripted",
            intended_role="baseline",
            backend="scripted",
        ),
        PanelSlotSpec(
            slot_name="anthropic_fast",
            family="anthropic",
            intended_role="fast",
            backend="openrouter",
            candidate_slugs=("anthropic/claude-haiku-4.5", "anthropic/claude-haiku"),
            fallback_model_id="claude-haiku-4-5-20251001",
        ),
        PanelSlotSpec(
            slot_name="anthropic_strong",
            family="anthropic",
            intended_role="strong",
            backend="openrouter",
            candidate_slugs=("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet"),
            fallback_model_id="claude-sonnet-4-6",
        ),
        PanelSlotSpec(
            slot_name="openai_frontier",
            family="openai",
            intended_role="frontier",
            backend="openrouter",
            candidate_slugs=("openai/gpt-5.5", "openai/gpt-5"),
        ),
    ),
    "phase15": (
        PanelSlotSpec(
            slot_name="baseline_greedy",
            family="scripted",
            intended_role="baseline",
            backend="scripted",
        ),
        PanelSlotSpec(
            slot_name="anthropic_fast",
            family="anthropic",
            intended_role="fast",
            backend="openrouter",
            candidate_slugs=("anthropic/claude-haiku-4.5", "anthropic/claude-haiku"),
            fallback_model_id="claude-haiku-4-5-20251001",
        ),
        PanelSlotSpec(
            slot_name="anthropic_strong",
            family="anthropic",
            intended_role="strong",
            backend="openrouter",
            candidate_slugs=("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet"),
            fallback_model_id="claude-sonnet-4-6",
        ),
        PanelSlotSpec(
            slot_name="anthropic_frontier",
            family="anthropic",
            intended_role="frontier",
            backend="openrouter",
            candidate_slugs=("anthropic/claude-opus-4.7", "anthropic/claude-opus"),
            fallback_model_id="claude-opus-4-7",
        ),
        PanelSlotSpec(
            slot_name="openai_frontier",
            family="openai",
            intended_role="frontier",
            backend="openrouter",
            candidate_slugs=("openai/gpt-5.5", "openai/gpt-5"),
        ),
        PanelSlotSpec(
            slot_name="openai_fast",
            family="openai",
            intended_role="fast",
            backend="openrouter",
            candidate_slugs=("openai/gpt-mini", "openai/gpt-5-mini"),
        ),
        PanelSlotSpec(
            slot_name="openai_nano",
            family="openai",
            intended_role="nano",
            backend="openrouter",
            candidate_slugs=("openai/gpt-nano", "openai/gpt-5-nano"),
        ),
        PanelSlotSpec(
            slot_name="gemini_frontier",
            family="google",
            intended_role="frontier",
            backend="openrouter",
            candidate_slugs=("google/gemini-3.1-pro", "google/gemini-3-pro", "google/gemini-pro"),
        ),
        PanelSlotSpec(
            slot_name="gemini_fast",
            family="google",
            intended_role="fast",
            backend="openrouter",
            candidate_slugs=("google/gemini-3.2-flash", "google/gemini-3-flash", "google/gemini-flash"),
        ),
    ),
}


def fetch_openrouter_models(api_key: str) -> list[dict]:
    request = Request(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter Models API failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach OpenRouter Models API: {exc.reason}") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter Models API returned an invalid response shape")
    return data


def load_catalog(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported catalog format in {path}")


def freeze_model_panel(panel: str, catalog: list[dict]) -> dict:
    if panel not in PANEL_SPECS:
        raise ValueError(f"Unknown model panel '{panel}'")

    slots = [_resolve_slot(slot, catalog) for slot in PANEL_SPECS[panel]]
    confirmed = sum(slot["status"] == "confirmed" for slot in slots)
    fallback_required = sum(slot["status"] == "fallback_required" for slot in slots)
    deferred = sum(slot["status"] == "deferred" for slot in slots)

    return {
        "panel": panel,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "catalog_model_count": len(catalog),
            "catalog_endpoint": OPENROUTER_MODELS_URL,
        },
        "summary": {
            "total_slots": len(slots),
            "confirmed_slots": confirmed,
            "fallback_required_slots": fallback_required,
            "deferred_slots": deferred,
        },
        "slots": slots,
    }


def manifest_has_unresolved_slots(manifest: dict) -> bool:
    summary = manifest.get("summary", {})
    return bool(summary.get("fallback_required_slots") or summary.get("deferred_slots"))


def write_manifest(manifest: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def get_openrouter_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    return api_key


def _resolve_slot(slot: PanelSlotSpec, catalog: list[dict]) -> dict:
    if slot.backend == "scripted":
        return {
            "slot_name": slot.slot_name,
            "family": slot.family,
            "intended_role": slot.intended_role,
            "backend": "scripted",
            "model_id": "greedy",
            "fallback_model_id": None,
            "status": "confirmed",
            "resolution": "static",
            "candidate_slugs": [],
            "catalog_match": None,
        }

    model = _find_catalog_match(slot.candidate_slugs, catalog)
    if model is None:
        status = "fallback_required" if slot.fallback_model_id else "deferred"
        return {
            "slot_name": slot.slot_name,
            "family": slot.family,
            "intended_role": slot.intended_role,
            "backend": "openrouter",
            "model_id": None,
            "fallback_model_id": slot.fallback_model_id,
            "status": status,
            "resolution": "unresolved",
            "candidate_slugs": list(slot.candidate_slugs),
            "catalog_match": None,
        }

    canonical_slug = model.get("canonical_slug") or model.get("id")
    return {
        "slot_name": slot.slot_name,
        "family": slot.family,
        "intended_role": slot.intended_role,
        "backend": "openrouter",
        "model_id": f"openrouter/{canonical_slug}",
        "fallback_model_id": slot.fallback_model_id,
        "status": "confirmed",
        "resolution": "catalog_match",
        "candidate_slugs": list(slot.candidate_slugs),
        "catalog_match": {
            "id": model.get("id"),
            "canonical_slug": model.get("canonical_slug"),
            "name": model.get("name"),
            "context_length": model.get("context_length"),
            "pricing": model.get("pricing"),
        },
    }


def _find_catalog_match(candidate_slugs: tuple[str, ...], catalog: list[dict]) -> dict | None:
    by_slug = {}
    for model in catalog:
        canonical_slug = model.get("canonical_slug")
        model_id = model.get("id")
        if canonical_slug:
            by_slug[canonical_slug] = model
        if model_id:
            by_slug[model_id] = model

    for slug in candidate_slugs:
        if slug in by_slug:
            return by_slug[slug]
    return None
