import re

from concord.agents.base import Action, AgentProtocol
from concord.schemas.episode import ActionType


class HonestWinWinAgent(AgentProtocol):
    async def act(self, env_state, private_ctx) -> Action:
        deal = env_state.scenario.deal_schema
        buyer_ctx = env_state.scenario.buyer_context
        seller_ctx = env_state.scenario.seller_context

        if private_ctx is buyer_ctx:
            my_batna = buyer_ctx.batna
            their_batna = seller_ctx.batna
            their_reserve = seller_ctx.reserve_price
        else:
            my_batna = seller_ctx.batna
            their_batna = buyer_ctx.batna
            their_reserve = buyer_ctx.reserve_price

        fair_mid = (my_batna + their_batna) / 2

        # Ensure the offer stays within the counterparty's ZOPA:
        # If seller, cap at buyer's reserve (they won't pay more).
        # If buyer, floor at seller's reserve (they won't accept less).
        if private_ctx is seller_ctx and their_reserve is not None:
            fair_mid = min(fair_mid, their_reserve * 0.95)
        elif private_ctx is buyer_ctx and their_reserve is not None:
            fair_mid = max(fair_mid, their_reserve * 1.05)

        offer = {}
        all_constraints = buyer_ctx.hard_constraints + seller_ctx.hard_constraints
        for k, v in deal.items():
            if v == "float" or v == "int":
                if "price" in k or "amount" in k:
                    offer[k] = fair_mid
                elif "seats" in k or "quantity" in k:
                    offer[k] = _minimum_int_for_field(k, all_constraints)
                elif "months" in k:
                    offer[k] = _minimum_int_for_field(k, all_constraints)
                else:
                    offer[k] = 1 if v == "int" else float(max(1, fair_mid))
            elif v == "bool":
                offer[k] = True
            elif v == "str":
                offer[k] = "standard"
            elif v == "list":
                offer[k] = ["fair_trade_commitment"]

        info_msg = f"I propose a fair price of {fair_mid:.0f}. This works for both sides."
        return Action(ActionType.OFFER, content=info_msg, offer_dict=offer)


def _minimum_int_for_field(field_name: str, constraints: list[str]) -> int:
    minima = [1]
    for constraint in constraints:
        if "quantity" in field_name:
            match = re.search(r"minimum_(?:order|quantity)_(\d+)", constraint, re.IGNORECASE)
            if match:
                minima.append(int(match.group(1)))
        elif "seats" in field_name:
            match = re.search(r"minimum_(?:(\d+)_seats|seats_(\d+))", constraint, re.IGNORECASE)
            if match:
                minima.append(int(match.group(1) or match.group(2)))
        elif "month" in field_name:
            match = re.search(
                r"(?:minimum_)?(?:commitment_)?(\d+)_(?:month|year)|minimum_commitment_(\d+)",
                constraint,
                re.IGNORECASE,
            )
            if match:
                months = int(match.group(1) or match.group(2))
                if "year" in constraint.lower():
                    months *= 12
                minima.append(months)
    return max(minima)
