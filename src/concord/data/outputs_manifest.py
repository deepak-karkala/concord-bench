from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ALLOWED_CLASSIFICATIONS = {"canonical", "superseded", "debug", "scratch"}
DEFAULT_MANIFEST_FILENAME = "outputs_manifest.json"


class OutputsManifestError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def outputs_root() -> Path:
    return repo_root() / "outputs"


def load_outputs_manifest(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise OutputsManifestError(f"Invalid outputs manifest in {manifest_path}: expected object")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise OutputsManifestError(f"Invalid outputs manifest in {manifest_path}: missing entries list")

    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise OutputsManifestError(f"Invalid outputs manifest entry in {manifest_path}: expected object")
        path_value = entry.get("path")
        classification = entry.get("classification")
        if not isinstance(path_value, str) or not path_value.strip():
            raise OutputsManifestError(f"Invalid outputs manifest entry in {manifest_path}: missing path")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise OutputsManifestError(
                f"Invalid outputs manifest entry in {manifest_path}: "
                f"unsupported classification '{classification}'"
            )
        normalized = dict(entry)
        normalized["path"] = path_value.strip().rstrip("/")
        normalized_entries.append(normalized)

    payload["entries"] = normalized_entries
    return payload


def find_outputs_manifest(start_path: Path) -> Path | None:
    for candidate in [start_path, *start_path.parents]:
        manifest_path = candidate / DEFAULT_MANIFEST_FILENAME
        if manifest_path.exists():
            return manifest_path
    return None


def resolve_manifest_entry(
    target_path: Path,
    manifest: dict,
    manifest_path: Path,
) -> dict | None:
    target_resolved = target_path.resolve()
    best_entry: dict | None = None
    best_path: Path | None = None

    for entry in manifest.get("entries", []):
        entry_path = _resolve_entry_path(entry.get("path"), manifest_path)
        if entry_path is None:
            continue
        try:
            if not target_resolved.is_relative_to(entry_path):
                continue
        except AttributeError:
            if entry_path not in target_resolved.parents and target_resolved != entry_path:
                continue

        if best_path is None or len(entry_path.parts) > len(best_path.parts):
            best_entry = entry
            best_path = entry_path

    return best_entry


def ensure_path_approved(
    target_path: Path,
    manifest: dict,
    manifest_path: Path,
    *,
    allowed_classifications: Iterable[str] = ("canonical", "debug"),
) -> dict:
    entry = resolve_manifest_entry(target_path, manifest, manifest_path)
    if entry is None:
        raise OutputsManifestError(
            f"{target_path} is not approved by outputs manifest {manifest_path}"
        )
    if entry.get("classification") not in set(allowed_classifications):
        raise OutputsManifestError(
            f"{target_path} is classified as {entry.get('classification')} "
            f"in outputs manifest {manifest_path}, which is not allowed for this report"
        )
    return entry


def _resolve_entry_path(path_value: object, manifest_path: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    path = Path(path_value.strip())
    if path.is_absolute():
        return path.resolve()

    outputs_dir = manifest_path.parent.resolve()
    return (outputs_dir / path).resolve()
