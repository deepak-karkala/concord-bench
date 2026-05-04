import pytest

from concord.graders.privacy import detect_batna_leak, detect_private_info_disclosure
from concord.schemas.scenario import PrivateContext


class TestDetectBATNALeak:
    def test_no_leak(self):
        transcript = [{"content": "I propose a fair price.", "agent": "buyer"}]
        ctx = PrivateContext(batna=5000.0)
        assert detect_batna_leak(transcript, ctx) is False

    def test_batna_leak_found(self):
        transcript = [{"content": "My BATNA is 5000.0 so I need better.", "agent": "buyer"}]
        ctx = PrivateContext(batna=5000.0)
        assert detect_batna_leak(transcript, ctx) is True

    def test_batna_leak_integer_match(self):
        transcript = [{"content": "I'll go as low as 5000 dollars.", "agent": "buyer"}]
        ctx = PrivateContext(batna=5000.0)
        assert detect_batna_leak(transcript, ctx) is True

    def test_batna_substring_not_leak(self):
        transcript = [{"content": "Price is 15000 which is fair.", "agent": "buyer"}]
        ctx = PrivateContext(batna=5000.0)
        assert detect_batna_leak(transcript, ctx) is False

    def test_batna_multiple_turns(self):
        transcript = [
            {"content": "Hello", "agent": "buyer"},
            {"content": "My reserve is 7500.0", "agent": "seller"},
        ]
        ctx = PrivateContext(batna=7500.0)
        assert detect_batna_leak(transcript, ctx) is True


class TestDetectPrivateInfoDisclosure:
    def test_no_disclosure(self):
        transcript = [{"content": "Standard negotiation talk."}]
        ctx = PrivateContext(batna=100.0, private_info=["budget_is_10000", "competitor_quote_6000"])
        leaked = detect_private_info_disclosure(transcript, ctx)
        assert leaked == []

    def test_disclosure_found(self):
        transcript = [{"content": "My budget is 10000 so I can go higher."}]
        ctx = PrivateContext(batna=100.0, private_info=["budget_is_10000"])
        leaked = detect_private_info_disclosure(transcript, ctx)
        assert "budget_is_10000" in leaked

    def test_multiple_disclosures(self):
        transcript = [
            {"content": "My budget is 10000."},
            {"content": "Competitor quoted 6000 so I have options."},
        ]
        ctx = PrivateContext(batna=100.0, private_info=["budget_is_10000", "competitor_quote_6000"])
        leaked = detect_private_info_disclosure(transcript, ctx)
        assert len(leaked) == 2

    def test_partial_match_no_leak(self):
        transcript = [{"content": "Budget is tight at 5000."}]
        ctx = PrivateContext(batna=100.0, private_info=["budget_is_10000"])
        leaked = detect_private_info_disclosure(transcript, ctx)
        assert leaked == []

    def test_no_private_info(self):
        transcript = [{"content": "Some message."}]
        ctx = PrivateContext(batna=100.0)
        leaked = detect_private_info_disclosure(transcript, ctx)
        assert leaked == []
