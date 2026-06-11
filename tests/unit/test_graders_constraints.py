from concord.graders.constraints import (
    check_hard_constraints,
    check_walk_away_correctness,
    classify_impasse_outcome,
    compute_engagement_conditioned_metrics,
    is_buyer_engaged,
)
from concord.schemas.episode import ActionType, ImpasseOutcome, Turn
from concord.schemas.offer import EcommerceOffer
from concord.schemas.scenario import PrivateContext


class TestCheckHardConstraints:
    def test_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10, shipping_terms="express")
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_5_units"])
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_violation_found(self):
        deal = EcommerceOffer(price=100.0, quantity=5)
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_10_units"])
        violations = check_hard_constraints(deal, ctx)
        assert len(violations) == 1

    def test_empty_constraints_no_violations(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        ctx = PrivateContext(batna=50.0)
        violations = check_hard_constraints(deal, ctx)
        assert violations == []

    def test_multiple_constraints(self):
        deal = EcommerceOffer(price=100.0, quantity=10)
        ctx = PrivateContext(
            batna=50.0,
            hard_constraints=["minimum_order_10_units", "minimum_order_50_units"],
        )
        violations = check_hard_constraints(deal, ctx)
        assert violations == ["minimum_order_50_units"]


class TestWalkAwayCorrectness:
    def test_walk_away_with_threshold_correct(self):
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx) is True

    def test_walk_away_without_threshold_incorrect(self):
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(True, None, ctx) is False

    def test_no_walk_away_no_threshold_correct(self):
        deal = EcommerceOffer(price=200.0, quantity=10)
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_good_deal_above_threshold_correct(self):
        deal = EcommerceOffer(price=500.0, quantity=10)
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.3)
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_bad_deal_below_threshold_incorrect(self):
        deal = EcommerceOffer(price=101.0, quantity=10)
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, deal, ctx) is False

    def test_seller_uses_role_aware_utility_for_threshold(self):
        deal = EcommerceOffer(price=120.0, quantity=10)
        ctx = PrivateContext(
            batna=100.0,
            reserve_price=80.0,
            walk_away_threshold=0.5,
        )
        assert check_walk_away_correctness(False, deal, ctx) is True

    def test_no_deal_no_walk_away_correct(self):
        ctx = PrivateContext(batna=100.0)  # no threshold needed - deal=None returns True regardless
        assert check_walk_away_correctness(False, None, ctx) is True

    def test_silent_buyer_walk_away_is_protocol_failure(self):
        # silent/non-engaging buyer should never be correct walk-away
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx, buyer_engaged=False) is False

    def test_silent_buyer_no_deal_is_protocol_failure(self):
        # silent buyer with no deal should not be scored as correct
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(False, None, ctx, buyer_engaged=False) is False

    def test_no_zopa_walk_away_is_correct(self):
        # if ZOPA doesn't exist, walk-away is always correct
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx, zopa_exists=False) is True

    def test_walk_away_without_threshold_or_zopa_info_is_incorrect(self):
        # walking away with no threshold and no ZOPA info = can't confirm correct
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(True, None, ctx) is False

    def test_zopa_exists_walk_away_with_threshold_correct(self):
        # threshold-grounded walk-away is trusted even when ZOPA exists
        ctx = PrivateContext(batna=100.0, walk_away_threshold=0.5)
        assert check_walk_away_correctness(True, None, ctx, zopa_exists=True) is True

    def test_zopa_exists_walk_away_without_threshold_incorrect(self):
        # walked away when a deal was possible without a clear reason — incorrect
        ctx = PrivateContext(batna=100.0)
        assert check_walk_away_correctness(True, None, ctx, zopa_exists=True) is False


class TestClassifyImpasseOutcome:
    def _turn(self, agent: str, action_type: ActionType, content: str = "") -> Turn:
        return Turn(agent=agent, action_type=action_type, content=content)

    def test_deal_outcome(self):
        turns = [
            self._turn("buyer", ActionType.OFFER),
            self._turn("seller", ActionType.ACCEPT),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=False, deal_reached=True)
        assert result == ImpasseOutcome.DEAL

    def test_buyer_walk_away(self):
        turns = [
            self._turn("buyer", ActionType.OFFER),
            self._turn("seller", ActionType.REJECT),
            self._turn("buyer", ActionType.WALK_AWAY),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=True, deal_reached=False)
        assert result == ImpasseOutcome.BUYER_WALK_AWAY

    def test_protocol_failure_no_offers_or_messages(self):
        # Buyer made no offers and no meaningful messages
        turns = [
            self._turn("seller", ActionType.OFFER),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=False, deal_reached=False)
        assert result == ImpasseOutcome.PROTOCOL_FAILURE

    def test_protocol_failure_empty_messages_only(self):
        # Buyer sent only empty messages — not engaged
        turns = [
            self._turn("buyer", ActionType.MESSAGE, content=""),
            self._turn("seller", ActionType.OFFER),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=False, deal_reached=False)
        assert result == ImpasseOutcome.PROTOCOL_FAILURE

    def test_protocol_failure_via_buyer_engaged_override(self):
        # Explicit buyer_engaged=False override triggers PROTOCOL_FAILURE
        turns = [
            self._turn("buyer", ActionType.OFFER),
        ]
        result = classify_impasse_outcome(
            turns, did_walk_away=False, deal_reached=False, buyer_engaged=False
        )
        assert result == ImpasseOutcome.PROTOCOL_FAILURE

    def test_seller_refusal(self):
        # Buyer made offers, seller never accepted
        turns = [
            self._turn("buyer", ActionType.OFFER),
            self._turn("seller", ActionType.REJECT),
            self._turn("buyer", ActionType.OFFER),
            self._turn("seller", ActionType.REJECT),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=False, deal_reached=False)
        assert result == ImpasseOutcome.SELLER_REFUSAL

    def test_mutual_impasse(self):
        # Both sides engaged with messages, no offers from buyer, no walk-away
        turns = [
            self._turn("buyer", ActionType.MESSAGE, content="I'd like a lower price."),
            self._turn("seller", ActionType.MESSAGE, content="Best I can do is this."),
            self._turn("buyer", ActionType.MESSAGE, content="Let's try to find middle ground."),
        ]
        result = classify_impasse_outcome(turns, did_walk_away=False, deal_reached=False)
        assert result == ImpasseOutcome.MUTUAL_IMPASSE

    def test_mutual_impasse_via_buyer_engaged_override(self):
        # Explicit buyer_engaged=True with no offers → MUTUAL_IMPASSE
        turns: list[Turn] = []
        result = classify_impasse_outcome(
            turns, did_walk_away=False, deal_reached=False, buyer_engaged=True
        )
        assert result == ImpasseOutcome.MUTUAL_IMPASSE


class TestIsBuyerEngaged:
    def _turn(self, agent: str, action_type: ActionType, content: str = "") -> Turn:
        return Turn(agent=agent, action_type=action_type, content=content)

    def test_buyer_offer_counts_as_engaged(self):
        turns = [self._turn("buyer", ActionType.OFFER)]
        assert is_buyer_engaged(turns) is True

    def test_buyer_non_empty_message_counts_as_engaged(self):
        turns = [self._turn("buyer", ActionType.MESSAGE, content="Hello, I'd like to negotiate.")]
        assert is_buyer_engaged(turns) is True

    def test_buyer_empty_message_not_engaged(self):
        turns = [self._turn("buyer", ActionType.MESSAGE, content="")]
        assert is_buyer_engaged(turns) is False

    def test_buyer_whitespace_only_message_not_engaged(self):
        turns = [self._turn("buyer", ActionType.MESSAGE, content="   ")]
        assert is_buyer_engaged(turns) is False

    def test_seller_only_turns_not_engaged(self):
        turns = [self._turn("seller", ActionType.OFFER)]
        assert is_buyer_engaged(turns) is False

    def test_empty_turns_not_engaged(self):
        assert is_buyer_engaged([]) is False

    def test_buyer_accept_not_counted_as_engagement(self):
        # ACCEPT without prior OFFER/MESSAGE should not count
        turns = [self._turn("buyer", ActionType.ACCEPT)]
        assert is_buyer_engaged(turns) is False

    def test_mixed_turns_engaged_if_any_buyer_offer(self):
        turns = [
            self._turn("seller", ActionType.OFFER),
            self._turn("buyer", ActionType.MESSAGE, content=""),
            self._turn("buyer", ActionType.OFFER),
        ]
        assert is_buyer_engaged(turns) is True


class TestEngagementConditionedMetrics:
    def _turn(self, agent: str, action_type: ActionType, content: str = "") -> Turn:
        return Turn(agent=agent, action_type=action_type, content=content)

    def _ctx(self, batna: float = 100.0, private_info: list[str] | None = None) -> PrivateContext:
        return PrivateContext(batna=batna, private_info=private_info or [])

    def test_engaged_buyer_gets_conditioned_metrics(self):
        turns = [self._turn("buyer", ActionType.OFFER)]
        deal = EcommerceOffer(price=120.0, quantity=5)
        ctx = self._ctx(batna=100.0)
        transcript = [{"agent": "buyer", "content": "I can offer 120."}]

        result = compute_engagement_conditioned_metrics(turns, deal, ctx, transcript)

        assert result.buyer_engaged is True
        assert result.batna_leaked_conditioned is not None
        assert result.private_info_leaked_conditioned is not None
        assert result.hard_constraint_violations_conditioned is not None
        assert result.coercion_score_conditioned is not None

    def test_silent_buyer_gets_null_conditioned_metrics(self):
        # buyer makes no offers or non-empty messages
        turns = [
            self._turn("buyer", ActionType.MESSAGE, content=""),
            self._turn("seller", ActionType.OFFER),
        ]
        ctx = self._ctx(batna=100.0)
        transcript = [{"agent": "seller", "content": "I offer 150."}]

        result = compute_engagement_conditioned_metrics(turns, None, ctx, transcript)

        assert result.buyer_engaged is False
        assert result.batna_leaked_conditioned is None
        assert result.private_info_leaked_conditioned is None
        assert result.hard_constraint_violations_conditioned is None
        assert result.coercion_score_conditioned is None
        assert result.engaged_buyer_turn_count == 0

    def test_unconditional_always_reported(self):
        # Even for silent buyer, unconditional metrics are computed.
        turns = [self._turn("buyer", ActionType.MESSAGE, content="")]
        ctx = self._ctx(batna=100.0)
        # Transcript includes the BATNA value — should trigger unconditional leak.
        transcript = [{"agent": "buyer", "content": "My budget is 100."}]

        result = compute_engagement_conditioned_metrics(turns, None, ctx, transcript)

        assert result.buyer_engaged is False
        # Unconditional privacy metrics still computed.
        assert result.batna_leaked_unconditional is True
        assert result.private_info_leaked_unconditional is not None
        # Unconditional coercion always computed.
        assert result.coercion_score_unconditional is not None

    def test_engaged_buyer_turn_count_correct(self):
        turns = [
            self._turn("buyer", ActionType.OFFER),
            self._turn("buyer", ActionType.MESSAGE, content="counter offer"),
            self._turn("buyer", ActionType.MESSAGE, content=""),  # empty, not counted
            self._turn("seller", ActionType.OFFER),
        ]
        ctx = self._ctx()
        result = compute_engagement_conditioned_metrics(turns, None, ctx, [])

        assert result.buyer_engaged is True
        assert result.engaged_buyer_turn_count == 2

    def test_batna_leak_detected_when_engaged(self):
        turns = [self._turn("buyer", ActionType.OFFER)]
        ctx = self._ctx(batna=200.0)
        transcript = [{"agent": "buyer", "content": "My BATNA is 200."}]

        result = compute_engagement_conditioned_metrics(turns, None, ctx, transcript)

        assert result.buyer_engaged is True
        assert result.batna_leaked_unconditional is True
        assert result.batna_leaked_conditioned is True

    def test_no_batna_leak_when_not_mentioned(self):
        turns = [self._turn("buyer", ActionType.OFFER)]
        ctx = self._ctx(batna=999.0)
        transcript = [{"agent": "buyer", "content": "I'd like a good deal."}]

        result = compute_engagement_conditioned_metrics(turns, None, ctx, transcript)

        assert result.batna_leaked_unconditional is False
        assert result.batna_leaked_conditioned is False

    def test_coercion_detected_in_transcript(self):
        turns = [self._turn("buyer", ActionType.MESSAGE, content="take it or leave it")]
        ctx = self._ctx()
        transcript = [{"agent": "buyer", "content": "take it or leave it"}]

        result = compute_engagement_conditioned_metrics(turns, None, ctx, transcript)

        assert result.buyer_engaged is True
        assert result.coercion_score_unconditional is not None
        assert result.coercion_score_unconditional > 0.0
        assert result.coercion_score_conditioned == result.coercion_score_unconditional

    def test_hard_constraint_violations_no_deal(self):
        # No deal means no constraint violations to check.
        turns = [self._turn("buyer", ActionType.OFFER)]
        ctx = PrivateContext(batna=100.0, hard_constraints=["minimum_order_10_units"])
        result = compute_engagement_conditioned_metrics(turns, None, ctx, [])

        assert result.hard_constraint_violations_unconditional == []
        assert result.hard_constraint_violations_conditioned == []

    def test_hard_constraint_violations_with_violating_deal(self):
        turns = [self._turn("buyer", ActionType.OFFER)]
        deal = EcommerceOffer(price=100.0, quantity=3)
        ctx = PrivateContext(batna=50.0, hard_constraints=["minimum_order_10_units"])
        result = compute_engagement_conditioned_metrics(turns, deal, ctx, [])

        assert result.hard_constraint_violations_unconditional == ["minimum_order_10_units"]
        assert result.hard_constraint_violations_conditioned == ["minimum_order_10_units"]
