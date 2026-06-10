from concord.data.seed_audit import audit_multi_issue_utility_models


def test_seed_corpus_has_explicit_multi_issue_utility_models() -> None:
    report = audit_multi_issue_utility_models()

    assert report["multi_issue_count"] >= 100
    assert (
        report["with_explicit_issue_utility_models"] == report["multi_issue_count"]
    )
    assert report["missing_issue_utility_ids"] == []
    assert report["findings"]["all_multi_issue_seeds_have_explicit_models"] is True
