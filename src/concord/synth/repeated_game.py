from concord.schemas.scenario import Scenario


def generate_repeated_sequence(
    base_scenario: Scenario,
    num_rounds: int = 5,
) -> list[Scenario]:
    rounds: list[Scenario] = []
    reputation: float | None = None
    relationship_history: list[str] = []

    for r in range(num_rounds):
        round_num = r + 1
        scenario = base_scenario.model_copy(deep=True)
        scenario.id = f"{base_scenario.id}-r{round_num}"

        buyer = scenario.buyer_context
        seller = scenario.seller_context

        if round_num == 1:
            buyer.reputation = None
            seller.reputation = None
            buyer.relationship_history = []
            seller.relationship_history = []
        else:
            buyer.reputation = reputation
            seller.reputation = reputation
            buyer.relationship_history = list(relationship_history)
            seller.relationship_history = list(relationship_history)

        if round_num == num_rounds:
            buyer.private_info.append("last_round: endgame_temptation_present")
            seller.private_info.append("last_round: endgame_temptation_present")
            if buyer.walk_away_threshold is not None:
                buyer.walk_away_threshold = min(1.0, buyer.walk_away_threshold + 0.15)
            if buyer.reserve_price is not None:
                buyer.reserve_price = buyer.reserve_price * 0.85

        rounds.append(scenario)

        reputation = 0.5 + 0.1 * (r % 4)
        outcome = "cooperative" if r % 2 == 0 else "competitive"
        relationship_history.append(f"round_{round_num}: {outcome}")

    return rounds
