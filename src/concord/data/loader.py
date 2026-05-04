from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from concord.schemas.scenario import Scenario

DATASET_REPO = "deepak-karkala/concord-bench"

if TYPE_CHECKING:
    from huggingface_hub import snapshot_download  # noqa: F811


def _get_hf_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F811
        return snapshot_download
    except ImportError:
        raise ImportError(
            "Concord scenario downloading requires huggingface-hub. "
            "Install with: pip install concord-bench"
        )


def load_seeds(
    domain: str | None = None,
    seed_dir: Path | None = None,
) -> list[Scenario]:
    if seed_dir is None:
        seed_dir = Path(__file__).parent / "seed_yamls"
    if not seed_dir.exists():
        return []
    yaml_paths = list(seed_dir.rglob("*.yaml"))
    if domain:
        yaml_paths = [p for p in yaml_paths if p.parent.name == domain]
    scenarios: list[Scenario] = []
    for path in sorted(yaml_paths):
        with path.open() as f:
            data = yaml.safe_load(f)
        scenarios.append(Scenario.model_validate(data))
    return scenarios


def load_scenarios(
    version: str = "v0.1.0",
    domain: str | None = None,
    culture: str | None = None,
    cache_dir: str | None = None,
) -> list[Scenario]:
    try:
        snapshot_download = _get_hf_hub()
    except ImportError as e:
        raise RuntimeError(
            "Cannot download scenarios: huggingface-hub is not available. "
            "Use load_seeds() for offline access, or install concord-bench "
            "with: pip install concord-bench"
        ) from e

    try:
        local_path = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=version,
            cache_dir=cache_dir,
            allow_patterns=[f"data/scenarios/{version}/**/*.yaml"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download scenarios from Hugging Face Hub ({DATASET_REPO}). "
            f"Check your network connection and that the dataset exists at "
            f"https://huggingface.co/datasets/{DATASET_REPO}. "
            f"Error: {e}"
        ) from e

    scenario_dir = Path(local_path) / "data" / "scenarios" / version
    yaml_paths = list(scenario_dir.rglob("*.yaml"))
    if domain:
        yaml_paths = [p for p in yaml_paths if p.parent.parent.name == domain]
    if culture:
        yaml_paths = [p for p in yaml_paths if p.parent.name == culture]

    scenarios: list[Scenario] = []
    for path in sorted(yaml_paths):
        with path.open() as f:
            data = yaml.safe_load(f)
        scenarios.append(Scenario.model_validate(data))
    return scenarios
