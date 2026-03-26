"""Builds compact knowledge context (~300 tokens) for LLM system prompt injection.

Assembles visa, budget, exchange, and seasonality data based on user context.
"""
from typing import Optional

from knowledge.visa_tiers import VISA_TIERS
from knowledge.budget_tiers import BUDGET_TIERS
from knowledge.exchange_rates import EXCHANGE_FAVORABILITY
from knowledge.seasonality import SEASONALITY


def build_context(
    passport: str,
    budget: str,
    month: Optional[int] = None,
) -> str:
    visa = VISA_TIERS.get(passport, {})
    budget_tier = BUDGET_TIERS.get(passport, {}).get(budget, {})
    exchange = EXCHANGE_FAVORABILITY.get(passport, {})

    parts = [f"TRAVEL INTELLIGENCE (for {passport} passport, {budget} budget):"]

    if visa.get("visa_free"):
        parts.append(f"- Visa-free: {', '.join(visa['visa_free'][:8])}")
    if visa.get("e_visa_easy"):
        parts.append(f"- E-visa easy: {', '.join(visa['e_visa_easy'][:6])}")
    if visa.get("hard_visa"):
        parts.append(f"- Hard visa (avoid unless asked): {', '.join(visa['hard_visa'][:5])}")
    if budget_tier.get("daily_range"):
        parts.append(f"- Budget range: {budget_tier['daily_range']}")
    if exchange.get("great_value"):
        parts.append(f"- Great value currencies: {', '.join(exchange['great_value'][:5])}")
    if exchange.get("poor_value"):
        parts.append(f"- Poor value currencies: {', '.join(exchange['poor_value'][:5])}")
    if exchange.get("note"):
        parts.append(f"- {exchange['note']}")

    if month:
        good_now = [d for d, s in SEASONALITY.items() if month in s["best"]]
        if good_now:
            parts.append(f"- Good in month {month}: {', '.join(good_now[:8])}")

    return "\n".join(parts)
