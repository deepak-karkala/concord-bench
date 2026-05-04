import shutil
from pathlib import Path

import pytest
import yaml

from concord.data.loader import DATASET_REPO, load_scenarios, load_seeds
from concord.schemas.scenario import Domain


class TestLoadSeeds:
    def test_empty_seed_dir(self):
        result = load_seeds()
        assert isinstance(result, list)

    def test_domain_filter(self, temp_dir: Path):
        ecom_dir = temp_dir / "ecommerce"
        ecom_dir.mkdir(parents=True)
        scenario_data = {
            "id": "test-filter-001",
            "domain": "ecommerce",
            "culture": "US",
            "buyer_context": {"batna": 100.0},
            "seller_context": {"batna": 200.0},
            "deal_schema": {"price": "float"},
        }
        with open(ecom_dir / "test.yaml", "w") as f:
            yaml.safe_dump(scenario_data, f)

        result = load_seeds(domain="ecommerce", seed_dir=temp_dir)
        assert len(result) == 1
        assert result[0].id == "test-filter-001"

    def test_domain_filter_no_match(self, temp_dir: Path):
        ecom_dir = temp_dir / "ecommerce"
        ecom_dir.mkdir(parents=True)
        scenario_data = {
            "id": "test-nomatch",
            "domain": "ecommerce",
            "culture": "US",
            "buyer_context": {"batna": 100.0},
            "seller_context": {"batna": 200.0},
            "deal_schema": {"price": "float"},
        }
        with open(ecom_dir / "test.yaml", "w") as f:
            yaml.safe_dump(scenario_data, f)

        result = load_seeds(domain="saas_procurement", seed_dir=temp_dir)
        assert len(result) == 0

    def test_invalid_yaml_raises(self, temp_dir: Path):
        ecom_dir = temp_dir / "ecommerce"
        ecom_dir.mkdir(parents=True)
        bad_data = {
            "id": "bad",
            "domain": "not_a_domain",
            "buyer_context": {},
            "seller_context": {},
            "deal_schema": {},
        }
        with open(ecom_dir / "bad.yaml", "w") as f:
            yaml.safe_dump(bad_data, f)

        with pytest.raises(Exception):
            load_seeds(domain="ecommerce", seed_dir=temp_dir)

    def test_load_seeds_with_reference_fixture(self, temp_dir: Path):
        ecom_dir = temp_dir / "ecommerce"
        ecom_dir.mkdir(parents=True)
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "reference_scenarios"
        src = fixtures_dir / "ecommerce.yaml"
        shutil.copy(src, ecom_dir / "ecommerce.yaml")

        result = load_seeds(domain="ecommerce", seed_dir=temp_dir)
        assert len(result) == 1
        s = result[0]
        assert s.id == "ref-ecommerce-001"
        assert s.domain == Domain.ECOMMERCE
        assert s.buyer_context.batna == 3000.0
        assert s.seller_context.batna == 5000.0


class TestLoadScenarios:
    def test_dataset_repo_constant(self):
        assert "concord-bench" in DATASET_REPO
        assert "/" in DATASET_REPO

    def test_network_error_graceful(self):
        with pytest.raises(RuntimeError, match="Failed to download"):
            load_scenarios(version="v0.1.0", domain="ecommerce")

    def test_network_error_includes_repo_name(self):
        with pytest.raises(RuntimeError, match=DATASET_REPO):
            load_scenarios(version="v0.1.0")

    @pytest.mark.slow
    @pytest.mark.requires_api
    def test_load_from_hf_hub(self):
        scenarios = load_scenarios(version="v0.1.0", domain="ecommerce")
        assert len(scenarios) > 0
        for s in scenarios:
            assert s.domain == Domain.ECOMMERCE
