import json
from datetime import datetime, timezone
from pathlib import Path


def append_audit_log(
    log_path: str | Path,
    scenario_id: str,
    original_culture: str,
    target_culture: str,
    adapted_fields: list[str],
    auditor_comments: str = "",
) -> None:
    entry = {
        "scenario_id": scenario_id,
        "original_culture": original_culture,
        "target_culture": target_culture,
        "adapted_fields": adapted_fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auditor_comments": auditor_comments,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
