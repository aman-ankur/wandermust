"""Deterministic conversation controller — pure Python, zero LLM calls.

Manages topic selection, answer parsing, phase transitions, and fact tracking.
The LLM personality layer only phrases questions; it cannot break the flow.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("wandermust.conversation_controller")

MAX_PROFILE_TURNS = 5
MAX_DISCOVERY_TURNS = 6


@dataclass
class TopicConfig:
    key: str
    phase: str
    question_hint: str
    option_templates: List[Dict[str, str]]
    multi_select: bool = False


TOPIC_REGISTRY: List[TopicConfig] = [
    # Profile phase
    TopicConfig(
        "passport", "profile", "what passport they hold",
        [
            {"id": "in", "label": "Indian"},
            {"id": "us", "label": "US"},
            {"id": "uk", "label": "UK"},
            {"id": "eu", "label": "EU/Schengen"},
            {"id": "other", "label": "Other"},
        ],
    ),
    TopicConfig(
        "budget_level", "profile", "their budget comfort level",
        [
            {"id": "budget", "label": "Budget-friendly"},
            {"id": "moderate", "label": "Mid-range"},
            {"id": "comfortable", "label": "Comfortable"},
            {"id": "luxury", "label": "Luxury"},
        ],
    ),
    TopicConfig(
        "travel_style", "profile", "their preferred travel style",
        [
            {"id": "adventure", "label": "Adventure & outdoors"},
            {"id": "culture", "label": "Culture & history"},
            {"id": "relaxation", "label": "Relaxation & beaches"},
            {"id": "foodie", "label": "Food & culinary"},
            {"id": "mix", "label": "Mix of everything"},
        ],
    ),
    # Discovery phase
    TopicConfig(
        "timing", "discovery", "when they plan to travel",
        [
            {"id": "soon", "label": "Next 1-2 months"},
            {"id": "quarter", "label": "3-6 months out"},
            {"id": "later", "label": "6+ months out"},
            {"id": "flexible", "label": "Flexible"},
        ],
    ),
    TopicConfig(
        "companions", "discovery", "who they're traveling with",
        [
            {"id": "solo", "label": "Solo"},
            {"id": "couple", "label": "With partner"},
            {"id": "friends", "label": "With friends"},
            {"id": "family", "label": "Family with kids"},
        ],
    ),
    TopicConfig(
        "interests", "discovery", "activities and experiences that excite them",
        [
            {"id": "food", "label": "Street food & local cuisine"},
            {"id": "nature", "label": "Nature & hiking"},
            {"id": "history", "label": "History & architecture"},
            {"id": "nightlife", "label": "Nightlife & bars"},
            {"id": "beaches", "label": "Beaches & water sports"},
            {"id": "shopping", "label": "Markets & shopping"},
        ],
        multi_select=True,
    ),
    TopicConfig(
        "deal_breakers", "discovery", "any deal-breakers to avoid",
        [
            {"id": "long_flight", "label": "Long flights (10+ hours)"},
            {"id": "extreme_heat", "label": "Extreme heat"},
            {"id": "crowds", "label": "Overcrowded tourist spots"},
            {"id": "visa_hassle", "label": "Complex visa process"},
            {"id": "none", "label": "No deal-breakers"},
        ],
        multi_select=True,
    ),
]

REQUIRED_PROFILE_FACTS = {"passport", "budget_level", "travel_style"}
REQUIRED_DISCOVERY_FACTS = {"timing", "companions", "interests"}

_TOPIC_BY_KEY: Dict[str, TopicConfig] = {t.key: t for t in TOPIC_REGISTRY}


def get_topic(key: str) -> Optional[TopicConfig]:
    return _TOPIC_BY_KEY.get(key)


def pick_next_topic(known_facts: Dict[str, Any], phase: str) -> Optional[TopicConfig]:
    """Return the first unfilled topic for the given phase, or None if all filled."""
    for topic in TOPIC_REGISTRY:
        if topic.phase == phase and topic.key not in known_facts:
            return topic
    return None


def parse_answer(
    topic: TopicConfig,
    option_ids: Optional[List[str]] = None,
    free_text: Optional[str] = None,
) -> Any:
    """Parse user answer into a structured fact value.

    Priority: option_ids > free_text keyword match > raw free_text.
    """
    label_by_id = {o["id"]: o["label"] for o in topic.option_templates}

    if option_ids:
        if topic.multi_select:
            return [label_by_id[oid] for oid in option_ids if oid in label_by_id]
        first_id = option_ids[0]
        return label_by_id.get(first_id, first_id)

    if free_text:
        text_lower = free_text.lower().strip()
        matched = []
        for opt in topic.option_templates:
            if opt["label"].lower() in text_lower or opt["id"].lower() in text_lower:
                matched.append(opt)

        if matched:
            if topic.multi_select:
                return [m["label"] for m in matched]
            return matched[0]["label"]

        if topic.multi_select:
            return [free_text]
        return free_text

    return None


def should_transition(
    phase: str,
    known_facts: Dict[str, Any],
    turn_count: int,
) -> Optional[str]:
    """Check if we should transition to the next phase.

    Returns the new phase name, or None to stay in current phase.
    """
    if phase == "profile":
        if REQUIRED_PROFILE_FACTS.issubset(known_facts.keys()):
            return "discovery"
        if turn_count >= MAX_PROFILE_TURNS:
            logger.warning(f"Profile safety net: forcing transition after {turn_count} turns")
            return "discovery"

    elif phase == "discovery":
        if REQUIRED_DISCOVERY_FACTS.issubset(known_facts.keys()):
            return "narrowing"
        if turn_count >= MAX_DISCOVERY_TURNS:
            logger.warning(f"Discovery safety net: forcing transition after {turn_count} turns")
            return "narrowing"

    return None


@dataclass
class ControllerResult:
    """Output of build_controller_turn — everything the personality layer needs."""
    topic: Optional[TopicConfig]
    phase: str
    known_facts: Dict[str, Any]
    phase_changed: bool
    multi_select: bool = False
    option_templates: List[Dict[str, str]] = field(default_factory=list)
    last_answer_text: Optional[str] = None


def build_controller_turn(
    session: Dict[str, Any],
    option_ids: Optional[List[str]] = None,
    free_text: Optional[str] = None,
) -> ControllerResult:
    """Main orchestrator: parse answer → check transition → pick next topic.

    Called on every /respond. Returns a ControllerResult for the personality layer.
    """
    known_facts: Dict[str, Any] = dict(session.get("known_facts", {}))
    phase = session["phase"]
    turn_count = session.get("turn_count", 0)
    last_topic_key = session.get("last_topic_key")

    last_answer_text = None
    if last_topic_key:
        topic = get_topic(last_topic_key)
        if topic:
            parsed = parse_answer(topic, option_ids, free_text)
            if parsed is not None:
                known_facts[topic.key] = parsed
                if isinstance(parsed, list):
                    last_answer_text = ", ".join(parsed)
                else:
                    last_answer_text = str(parsed)

    new_phase = should_transition(phase, known_facts, turn_count)
    phase_changed = new_phase is not None
    if phase_changed:
        phase = new_phase

    next_topic = pick_next_topic(known_facts, phase)

    return ControllerResult(
        topic=next_topic,
        phase=phase,
        known_facts=known_facts,
        phase_changed=phase_changed,
        multi_select=next_topic.multi_select if next_topic else False,
        option_templates=next_topic.option_templates if next_topic else [],
        last_answer_text=last_answer_text,
    )


def build_first_turn(session: Dict[str, Any]) -> ControllerResult:
    """Pick the first topic for a new session (no answer to parse)."""
    known_facts: Dict[str, Any] = dict(session.get("known_facts", {}))
    phase = session["phase"]

    next_topic = pick_next_topic(known_facts, phase)

    return ControllerResult(
        topic=next_topic,
        phase=phase,
        known_facts=known_facts,
        phase_changed=False,
        multi_select=next_topic.multi_select if next_topic else False,
        option_templates=next_topic.option_templates if next_topic else [],
    )


# --- Passport code mapping for profile extraction ---
_PASSPORT_CODE_MAP = {
    "Indian": "IN",
    "US": "US",
    "UK": "GB",
    "EU/Schengen": "EU",
    "Other": "XX",
}

_BUDGET_LEVEL_MAP = {
    "Budget-friendly": "budget",
    "Mid-range": "moderate",
    "Comfortable": "comfortable",
    "Luxury": "luxury",
}

_STYLE_MAP = {
    "Adventure & outdoors": "adventure",
    "Culture & history": "culture",
    "Relaxation & beaches": "relaxation",
    "Food & culinary": "foodie",
    "Mix of everything": "mix",
}

_COMPANIONS_MAP = {
    "Solo": "solo",
    "With partner": "couple",
    "With friends": "group",
    "Family with kids": "family",
}


def build_profile_from_facts(known_facts: Dict[str, Any], user_id: str = "default") -> Dict[str, Any]:
    """Build a structured profile dict from known_facts — zero LLM."""
    passport_label = known_facts.get("passport", "Other")
    passport_code = _PASSPORT_CODE_MAP.get(passport_label, "XX")

    budget_label = known_facts.get("budget_level", "Mid-range")
    budget_level = _BUDGET_LEVEL_MAP.get(budget_label, "moderate")

    style_label = known_facts.get("travel_style", "Mix of everything")
    style = _STYLE_MAP.get(style_label, "mix")

    return {
        "user_id": user_id,
        "travel_history": [],
        "preferences": {
            "climate": "moderate",
            "pace": "moderate",
            "style": style,
        },
        "budget_level": budget_level,
        "passport_country": passport_code,
    }


def build_trip_intent_from_facts(known_facts: Dict[str, Any]) -> Dict[str, Any]:
    """Build trip_intent dict from known_facts — zero LLM."""
    timing = known_facts.get("timing", "Flexible")
    timing_map = {
        "Next 1-2 months": "next month",
        "3-6 months out": "in 3-6 months",
        "6+ months out": "later this year",
        "Flexible": "flexible",
    }

    companions_label = known_facts.get("companions", "Solo")
    companions = _COMPANIONS_MAP.get(companions_label, "solo")

    interests = known_facts.get("interests", [])
    if isinstance(interests, str):
        interests = [interests]

    deal_breakers = known_facts.get("deal_breakers", [])
    if isinstance(deal_breakers, str):
        deal_breakers = [deal_breakers]
    constraints = [db for db in deal_breakers if db != "No deal-breakers"]

    return {
        "travel_month": timing_map.get(timing, timing),
        "duration_days": 7,
        "interests": interests,
        "constraints": constraints,
        "travel_companions": companions,
        "region_preference": "",
        "budget_total": 0,
    }
