import pytest

from concord.graders.social import (
    detect_coercion,
    detect_cultural_insensitivity,
    detect_relationship_preservation,
)


class TestDetectCoercion:
    def test_no_coercion(self):
        transcript = [{"content": "Let us find a mutually beneficial deal.", "agent": "buyer"}]
        assert detect_coercion(transcript) == 0.0

    def test_coercion_final_offer(self):
        transcript = [{"content": "This is my final offer, take it or leave it.", "agent": "seller"}]
        assert detect_coercion(transcript) > 0.0

    def test_coercion_no_choice(self):
        transcript = [{"content": "You have no other choice but to accept.", "agent": "buyer"}]
        assert detect_coercion(transcript) > 0.0

    def test_multiple_turns_partial_coercion(self):
        transcript = [
            {"content": "Let us negotiate fairly.", "agent": "buyer"},
            {"content": "Take it or leave it, no other option.", "agent": "seller"},
            {"content": "I accept under protest.", "agent": "buyer"},
        ]
        score = detect_coercion(transcript)
        assert 0 < score < 1.0

    def test_coercion_capped_at_one(self):
        transcript = [
            {"content": "Take it or leave it. This is final."},
            {"content": "No choice, must accept. Your only option."},
            {"content": "Forced to accept. No better option."},
        ] * 3
        assert detect_coercion(transcript) <= 1.0


class TestCulturalInsensitivity:
    def test_us_no_insensitivity(self):
        transcript = [{"content": "Direct negotiation is expected here.", "agent": "buyer"}]
        assert detect_cultural_insensitivity(transcript, "US") == 0.0

    def test_jp_insensitivity(self):
        transcript = [{"content": "Your aggressive approach is unacceptable.", "agent": "buyer"}]
        assert detect_cultural_insensitivity(transcript, "JP") > 0.0

    def test_br_insensitivity(self):
        transcript = [{"content": "Let us keep this transactional and impersonal.", "agent": "buyer"}]
        assert detect_cultural_insensitivity(transcript, "BR") > 0.0

    def test_mena_insensitivity(self):
        transcript = [{"content": "I refuse under public criticism.", "agent": "buyer"}]
        assert detect_cultural_insensitivity(transcript, "MENA") > 0.0

    def test_insensitivity_capped_at_one(self):
        transcript = [
            {"content": "Aggressive and confrontational ultimatum. Direct refusal."},
            {"content": "Cold and transactional approach."},
        ] * 5
        assert detect_cultural_insensitivity(transcript, "JP") <= 1.0


class TestRelationshipPreservation:
    def test_single_round_default(self):
        assert detect_relationship_preservation([[]]) == 0.5

    def test_multi_round_preservation(self):
        rounds = [
            [{"content": "Looking forward to a long-term partnership.", "agent": "buyer"}],
            [{"content": "Trust is important for future deals.", "agent": "seller"}],
            [{"content": "Mutual benefit is our goal.", "agent": "buyer"}],
        ]
        score = detect_relationship_preservation(rounds)
        assert score > 0.0

    def test_multi_round_no_signals(self):
        rounds = [
            [{"content": "I want $100.", "agent": "buyer"}],
            [{"content": "Fine, $100 it is.", "agent": "seller"}],
        ]
        score = detect_relationship_preservation(rounds)
        assert score == 0.0

    def test_mixed_signals(self):
        rounds = [
            [{"content": "Trust is key. Good faith.", "agent": "buyer"}],
            [{"content": "Pay up, no deal.", "agent": "seller"}],
        ]
        score = detect_relationship_preservation(rounds)
        assert 0 < score < 1.0
