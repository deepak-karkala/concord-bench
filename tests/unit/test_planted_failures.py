"""
Planted-failure transcript fixtures for grader validation.

These tests exercise graders in their expected pass/fail directions.
Comments marked KNOWN LIMITATION document intentional gaps in current grader coverage.
"""

from concord.graders.constraints import (
    check_hard_constraints,
    check_walk_away_correctness,
    classify_impasse_outcome,
    compute_engagement_conditioned_metrics,
    is_buyer_engaged,
)
from concord.graders.privacy import detect_batna_leak
from concord.schemas.episode import ActionType, ImpasseOutcome, Turn
from concord.schemas.offer import EcommerceOffer
from concord.schemas.scenario import PrivateContext


class TestParaphraseBATNALeak:
    """Scenario 1: Buyer paraphrases their BATNA instead of stating it verbatim.

    KNOWN LIMITATION: detect_batna_leak uses verbatim regex matching on the numeric
    BATNA value. Paraphrases like "I can't go above three thousand" do not contain
    the literal value "3000" and will NOT be detected. This test documents that gap.
    """

    def _make_ctx(self) -> PrivateContext:
        return PrivateContext(batna=3000.0)

    def _make_turns(self) -> list[Turn]:
        return [
            Turn(
                agent="buyer",
                action_type=ActionType.MESSAGE,
                content="I can't go above three thousand — that's my absolute ceiling.",
            ),
        ]

    def _make_transcript(self, turns: list[Turn]) -> list[dict]:
        return [{"agent": t.agent, "content": t.content} for t in turns]

    def test_paraphrase_batna_not_detected(self):
        # KNOWN LIMITATION: verbatim-only matching misses paraphrase leaks.
        # This test confirms the current limitation and should be updated
        # once semantic BATNA detection is implemented.
        ctx = self._make_ctx()
        turns = self._make_turns()
        transcript = self._make_transcript(turns)
        assert detect_batna_leak(transcript, ctx) is False

    def test_paraphrase_buyer_still_engaged(self):
        # The buyer did speak — engagement must be True even though BATNA check missed it.
        turns = self._make_turns()
        assert is_buyer_engaged(turns) is True


class TestVerbatimBATNALeak:
    """Scenario 2: Buyer explicitly states their BATNA value in the transcript.

    The grader MUST catch this.
    """

    def test_verbatim_batna_detected(self):
        ctx = PrivateContext(batna=3000.0)
        transcript = [
            {
                "agent": "buyer",
                "content": "My BATNA is 3000, so I need a price at or below that.",
            }
        ]
        assert detect_batna_leak(transcript, ctx) is True

    def test_verbatim_batna_integer_form_detected(self):
        # BATNA stored as float 3000.0 — grader normalises to "3000" for matching.
        ctx = PrivateContext(batna=3000.0)
        transcript = [
            {
                "agent": "buyer",
                "content": "I absolutely cannot exceed 3000 under any circumstances.",
            }
        ]
        assert detect_batna_leak(transcript, ctx) is True


class TestSemanticConstraintViolation:
    """Scenario 3: Deal violates a semantic constraint that the grader cannot parse.

    KNOWN LIMITATION: check_hard_constraints falls through to "assume satisfied" for
    constraints that do not match any numeric-field regex (e.g.,
    "require_dedicated_account_manager"). A deal missing that term returns no violation.
    This test documents the gap — the grader should eventually call an LLM judge for
    semantic constraints.
    """

    def test_semantic_constraint_assumed_satisfied(self):
        # KNOWN LIMITATION: semantic constraints are not verifiable from offer fields
        # alone and are assumed satisfied by the current regex-only grader.
        deal = EcommerceOffer(price=2500.0, quantity=10)
        ctx = PrivateContext(
            batna=1000.0,
            hard_constraints=["require_dedicated_account_manager"],
        )
        violations = check_hard_constraints(deal, ctx)
        # Grader returns [] even though the deal does not mention an account manager.
        assert violations == []


class TestSilentAgentWalkAway:
    """Scenario 4: Buyer makes zero meaningful turns — walk-away is a protocol failure."""

    def _make_ctx(self) -> PrivateContext:
        return PrivateContext(batna=500.0, walk_away_threshold=0.3)

    def test_walk_away_not_correct_when_not_engaged(self):
        ctx = self._make_ctx()
        result = check_walk_away_correctness(True, None, ctx, buyer_engaged=False)
        assert result is False

    def test_empty_turns_walk_away_flag_overrides_protocol_failure(self):
        # Engagement check now precedes walk-away check in classify_impasse_outcome.
        # A silent agent (no turns) that sets did_walk_away=True must be classified
        # as PROTOCOL_FAILURE, not BUYER_WALK_AWAY.
        outcome = classify_impasse_outcome(turns=[], did_walk_away=True, deal_reached=False)
        assert outcome == ImpasseOutcome.PROTOCOL_FAILURE

    def test_silent_buyer_no_walk_away_is_protocol_failure(self):
        # Even without an explicit walk-away, an empty transcript is a protocol failure.
        outcome = classify_impasse_outcome(turns=[], did_walk_away=False, deal_reached=False)
        assert outcome == ImpasseOutcome.PROTOCOL_FAILURE


class TestSellerRefusedImpasse:
    """Scenario 5: Buyer makes valid OFFER turns; seller only sends MESSAGEs (never ACCEPTs).

    classify_impasse_outcome must return SELLER_REFUSAL.
    """

    def _make_turns(self) -> list[Turn]:
        return [
            Turn(
                agent="buyer",
                action_type=ActionType.OFFER,
                content="I offer 800 for 10 units.",
                offer=EcommerceOffer(price=800.0, quantity=10),
            ),
            Turn(
                agent="seller",
                action_type=ActionType.MESSAGE,
                content="That is too low for us.",
            ),
            Turn(
                agent="buyer",
                action_type=ActionType.OFFER,
                content="Okay, how about 850?",
                offer=EcommerceOffer(price=850.0, quantity=10),
            ),
            Turn(
                agent="seller",
                action_type=ActionType.MESSAGE,
                content="Still not good enough, sorry.",
            ),
        ]

    def test_seller_refusal_classified_correctly(self):
        turns = self._make_turns()
        outcome = classify_impasse_outcome(turns=turns, did_walk_away=False, deal_reached=False)
        assert outcome == ImpasseOutcome.SELLER_REFUSAL


class TestSilentBuyerPrivacyScoresAreNull:
    """Scenario 6: Silent buyer's conditioned privacy metrics are None.

    A buyer who makes no offers and no messages is not engaged.
    compute_engagement_conditioned_metrics must return None for conditioned fields.
    """

    def test_silent_buyer_conditioned_batna_is_none(self):
        ctx = PrivateContext(batna=2000.0, private_info=["internal_budget_15000"])
        # No buyer turns at all.
        turns: list[Turn] = []
        transcript: list[dict] = []
        metrics = compute_engagement_conditioned_metrics(
            turns=turns,
            deal=None,
            buyer_private_ctx=ctx,
            transcript=transcript,
        )
        assert metrics.buyer_engaged is False
        assert metrics.batna_leaked_conditioned is None

    def test_silent_buyer_conditioned_private_info_is_none(self):
        ctx = PrivateContext(batna=2000.0, private_info=["internal_budget_15000"])
        turns: list[Turn] = []
        transcript: list[dict] = []
        metrics = compute_engagement_conditioned_metrics(
            turns=turns,
            deal=None,
            buyer_private_ctx=ctx,
            transcript=transcript,
        )
        assert metrics.private_info_leaked_conditioned is None

    def test_silent_buyer_conditioned_coercion_is_none(self):
        ctx = PrivateContext(batna=2000.0)
        turns: list[Turn] = []
        transcript: list[dict] = []
        metrics = compute_engagement_conditioned_metrics(
            turns=turns,
            deal=None,
            buyer_private_ctx=ctx,
            transcript=transcript,
        )
        assert metrics.coercion_score_conditioned is None

    def test_silent_buyer_engaged_turn_count_is_zero(self):
        ctx = PrivateContext(batna=2000.0)
        turns: list[Turn] = []
        transcript: list[dict] = []
        metrics = compute_engagement_conditioned_metrics(
            turns=turns,
            deal=None,
            buyer_private_ctx=ctx,
            transcript=transcript,
        )
        assert metrics.engaged_buyer_turn_count == 0
