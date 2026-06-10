from click.testing import CliRunner

from concord.cli import main
from concord.schemas.scenario import Domain, PrivateContext, Scenario


def test_generate_fails_when_semantic_audit_finds_blocking_issues(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "concord.cli.load_seeds",
        lambda **kwargs: [
            Scenario(
                id="bad-seed-1",
                domain=Domain.ECOMMERCE,
                culture="US",
                buyer_context=PrivateContext(
                    batna=70.0,
                    reserve_price=60.0,
                    hard_constraints=["quality_meets_standards"],
                    private_info=["TBD"],
                ),
                seller_context=PrivateContext(
                    batna=40.0,
                    reserve_price=50.0,
                    hard_constraints=["minimum_order_100_units"],
                    private_info=["placeholder"],
                ),
                deal_schema={"price": "float"},
                scenario_description="TODO generic scenario",
                metadata={"difficulty_tier": 1},
            )
        ],
    )

    result = CliRunner().invoke(
        main,
        [
            "generate",
            "--domain",
            "ecommerce",
            "--count",
            "1",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Semantic audit failed" in result.output
    assert (tmp_path / "semantic_audit_report.json").exists()
