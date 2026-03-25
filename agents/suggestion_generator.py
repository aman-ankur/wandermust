"""Suggestion generator agent — reasons about destinations based on profile + intent.

Takes user_profile + trip_intent and uses LLM to suggest 3-5 destinations,
reasoning about visa requirements, budget, flights, seasonality, and preferences.
"""
import json
import logging
from models import DiscoveryState

logger = logging.getLogger("wandermust.suggestions")
from config import settings
from agents.llm_helper import get_llm, parse_json_response

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(settings.discovery_model)
    return _llm

SUGGESTION_PROMPT = """You are an expert travel advisor. Based on the user's profile and trip intent, suggest 3-5 destinations.

User Profile:
{profile}

Trip Intent:
{intent}

Consider these dimensions when reasoning:
1. **Visa**: Based on passport country ({passport}), prefer visa-free or e-visa destinations
2. **Budget**: Match the user's budget level ({budget}) to destination cost tiers
3. **Seasonality**: The user wants to travel in {month} — pick destinations that are in their best season
4. **Interests**: Focus on destinations matching: {interests}
5. **History**: They've visited {history} — suggest similar-but-new destinations, avoid repeats
6. **Constraints**: {constraints}

Return ONLY valid JSON — an array of 3-5 objects with this schema:
[
    {{
        "destination": "City, Country",
        "country": "Country",
        "reason": "2-3 sentence explanation of why this is a great match",
        "estimated_budget_per_day": <number in INR>,
        "best_months": [<month numbers 1-12>],
        "match_score": <float 0-1>,
        "tags": ["relevant", "tags"]
    }}
]

Sort by match_score descending. Be specific and practical.
"""


def generate_suggestions(profile: dict, trip_intent: dict) -> list[dict]:
    """Use LLM to generate destination suggestions."""
    passport = profile.get("passport_country", "IN")
    budget = profile.get("budget_level", "moderate")
    month = trip_intent.get("travel_month", "flexible")
    interests = ", ".join(trip_intent.get("interests", ["general travel"]))
    history = ", ".join(profile.get("travel_history", [])) or "none mentioned"
    constraints = ", ".join(trip_intent.get("constraints", [])) or "none"

    prompt = SUGGESTION_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        intent=json.dumps(trip_intent, indent=2),
        passport=passport,
        budget=budget,
        month=month,
        interests=interests,
        history=history,
        constraints=constraints,
    )

    try:
        llm = _get_llm()
        logger.info(f"Suggestions: calling LLM (passport={passport}, budget={budget}, month={month})")
        response = llm.invoke(prompt)
        logger.info(f"Suggestions: LLM response received ({len(response.content)} chars)")
        suggestions = parse_json_response(response.content)
        if isinstance(suggestions, list):
            logger.info(f"Suggestions: generated {len(suggestions)} destinations")
            return suggestions
        logger.warning("Suggestions: LLM returned non-list")
        return []
    except Exception as e:
        logger.error(f"Suggestions: LLM failed — {e}")
        return []


def suggestion_generator_node(state: DiscoveryState) -> dict:
    """LangGraph node: generate destination suggestions."""
    errors = list(state.get("errors", []))
    profile = state.get("user_profile", {})
    trip_intent = state.get("trip_intent", {})

    suggestions = generate_suggestions(profile, trip_intent)

    if not suggestions:
        errors.append("Suggestion generator: no suggestions generated")

    return {
        "suggestions": suggestions,
        "errors": errors,
    }
