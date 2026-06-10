from concord.data.seed_audit import audit_tier_semantics


def test_seed_corpus_exposes_provisional_tier_factorization() -> None:
    report = audit_tier_semantics()

    assert report["findings"]["tiers_are_provisional_operational_labels"] is True
    assert report["findings"]["empirical_calibration_required_after_scale"] is True
    assert (
        report["findings"]["difficulty_should_not_be_inferred_from_tier_label_alone"]
        is True
    )
    assert set(report["factor_dimensions"]["difficulty_label_counts"]) >= {"0", "1", "2", "3"}
    assert set(report["factor_dimensions"]["zopa_status_counts"]) >= {
        "none",
        "narrow_or_moderate",
        "wide",
    }
    assert set(report["factor_dimensions"]["issue_structure_counts"]) == {
        "multi_issue",
        "single_issue",
    }
    assert "bright_line" in report["factor_dimensions"]["pressure_type_counts"]
    assert "hard_constraints" in report["factor_dimensions"]["constraint_type_counts"]
    assert "privacy" in report["factor_dimensions"]["constraint_type_counts"]
    assert report["factor_dimensions"]["seller_policy_dimension"] == {
        "status": "runtime_controlled",
        "available_policies": [
            "honest",
            "honest_cooperative",
            "honest_hardball",
            "deceptive_or_pressure",
        ],
    }
