"""LLM-based intent extraction from free text answers.

Extracts multiple travel preference facts from a single free-text response.
Uses the fast model (gpt-4o-mini) for low latency. Falls back to empty dict on failure.
"""
import logging
import time
from typing import Any, Dict

from agents.llm_helper import get_llm, parse_json_response
from config import settings

logger = logging.getLogger("wandermust.intent_extractor")

VALID_TOPIC_KEYS = {
    "passport", "budget_level", "travel_style", "trip_duration",
    "timing", "companions", "interests", "deal_breakers",
}

EXTRACTION_PROMPT = """Extract travel preferences from this text. Only extract facts you're confident about.
Map to these topics (skip any not clearly mentioned):

- passport: one of ["Indian", "US", "UK", "EU/Schengen", "Other"]
- budget_level: one of ["Budget-friendly", "Mid-range", "Comfortable", "Luxury"]
- travel_style: one of ["Adventure & outdoors", "Culture & history", "Relaxation & beaches", "Food & culinary", "Mix of everything"]
- trip_duration: one of ["Long weekend (3-4 days)", "About a week", "Two weeks", "Extended trip (3+ weeks)", "Depends on the destination"]
- timing: one of ["Next 1-2 months", "3-6 months out", "6+ months out", "Flexible"]
- companions: one of ["Solo", "With partner", "With friends", "Family with kids"]
- interests: list from ["Street food & local cuisine", "Nature & hiking", "History & architecture", "Nightlife & bars", "Beaches & water sports", "Markets & shopping"]
- deal_breakers: list from ["Long flights (10+ hours)", "Extreme heat", "Overcrowded tourist spots", "Complex visa process", "No deal-breakers"]

Text: "{free_text}"
Already known (do NOT re-extract these): {known_keys}

Return ONLY valid JSON with topic keys and values. Return empty {{}} if nothing is clearly extractable."""

_extractor_llm = None


def _get_extractor_llm():
    global _extractor_llm
    if _extractor_llm is None:
        _extractor_llm = get_llm(settings.discovery_v2_personality_model).bind(
            response_format={"type": "json_object"},
            max_tokens=256,
        )
    return _extractor_llm


def extract_facts_from_text(
    free_text: str,
    known_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract travel preference facts from free text using LLM.

    Returns dict of {topic_key: value} for newly extracted facts.
    Never overwrites already-known facts. Returns {} on failure.
    """
    if not free_text or len(free_text.strip()) < 5:
        return {}

    known_keys = ", ".join(known_facts.keys()) if known_facts else "none"
    prompt = EXTRACTION_PROMPT.format(free_text=free_text, known_keys=known_keys)

    try:
        llm = _get_extractor_llm()
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[PERF] intent extraction: {ms:.0f}ms")

        data = parse_json_response(response.content)

        # Filter: only valid topic keys, only new facts
        result = {}
        for key, value in data.items():
            if key in VALID_TOPIC_KEYS and key not in known_facts and value:
                result[key] = value

        if result:
            logger.info(f"Extracted {len(result)} facts from free text: {list(result.keys())}")
        return result

    except Exception as e:
        logger.error(f"Intent extraction failed: {e}")
        return {}
