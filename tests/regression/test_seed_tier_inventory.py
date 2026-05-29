from concord.data.seed_audit import audit_seed_corpus


def test_seed_corpus_has_explicit_tier_inventory() -> None:
    report = audit_seed_corpus()

    assert report["missing_difficulty_tier_count"] == 0
    assert {"0", "1", "2", "3"} <= set(report["explicit_tier_counts"])
    assert report["by_domain_and_effective_tier"]["ecommerce"]["0"] >= 3
    assert report["by_domain_and_effective_tier"]["ecommerce"]["1"] >= 3
    assert report["by_domain_and_effective_tier"]["ecommerce"]["2"] >= 3
    assert report["by_domain_and_effective_tier"]["saas_procurement"]["2"] >= 3
    assert report["by_domain_and_effective_tier"]["settlement"]["2"] >= 3
    assert report["by_domain_and_effective_tier"]["ethical_business"]["2"] >= 3
