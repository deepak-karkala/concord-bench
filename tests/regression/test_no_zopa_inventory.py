from concord.data.seed_audit import audit_no_zopa_slice


def test_seed_corpus_has_audited_no_zopa_slice() -> None:
    report = audit_no_zopa_slice()

    assert 40 <= report["no_zopa_count"] <= 60
    assert report["structurally_valid_no_zopa_count"] == report["no_zopa_count"]
    assert report["positive_control_count"] >= 10
    assert report["findings"]["target_range_met"] is True
    assert report["findings"]["minimum_domain_coverage_met"] is True
    assert report["findings"]["all_no_zopa_structurally_valid"] is True
    assert report["no_zopa_by_domain"]["ecommerce"] >= 5
    assert report["no_zopa_by_domain"]["ethical_business"] >= 5
    assert report["no_zopa_by_domain"]["saas_procurement"] >= 5
    assert report["no_zopa_by_domain"]["settlement"] >= 5
