AWM_SYSTEM_PROMPT_ADDENDUM = """

6. MULTI-ISSUE TRADE-OFFS: At least 30% of generated scenarios must have explicit
   private valuations on at least 2 non-price terms. Each party must have a different
   preferred trade-off. The Pareto-optimal deal should involve a non-obvious exchange
   across dimensions — not just splitting the price.

   For ecommerce: contested non-price terms are delivery_days and payment_terms_days.
   For SaaS: contested terms are contract_length_months and onboarding_support_hours.
   For settlement: contested terms are payment_structure and payment_timeline_days.
   For ethical: contested terms are audit_frequency_months and transition_period_months.

   Private valuations must be explicit in private_info using this format:
   - "[term]_worth_[amount]_premium_to_principal"
   - "willing_to_accept_[term]_if_[condition]"
"""
