from concord.data.seed_audit import audit_t0_slice


def test_seed_corpus_has_hardened_multidomain_t0_slice() -> None:
    report = audit_t0_slice()

    assert report["t0_count"] >= 8
    assert report["t0_by_domain"]["ecommerce"] >= 1
    assert report["t0_by_domain"]["ethical_business"] >= 1
    assert report["t0_by_domain"]["saas_procurement"] >= 1
    assert report["t0_by_domain"]["settlement"] >= 1
    assert report["restatement_risk_ids"] == []
    assert report["findings"]["multi_domain_t0_met"] is True
    assert report["findings"]["all_t0s_low_restatement_risk"] is True
