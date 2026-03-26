"""Core conversation engine — generates ConversationTurn responses via LLM.

Replaces the separate onboarding + discovery_chat agents with a single
adaptive engine that uses pre-baked knowledge context.
"""
import json
import logging
from typing import Dict, List, Optional

from api.models import ConversationTurn, Option
from agents.llm_helper import get_llm, parse_json_response
from config import settings

logger = logging.getLogger("wandermust.conversation_engine")

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(settings.discovery_v2_model)
    return _llm


SYSTEM_PROMPT = """You are a well-traveled friend helping plan a trip. You've been to 60+ countries
and have strong opinions. You're not a search engine — you're someone who says
"oh you HAVE to go to Georgia, the food scene there blew my mind."

PERSONALITY:
- Opinionated: you have favorites and you share them with reasons
- Insightful: every option teaches something the user didn't know
- Reactive: you acknowledge what they said before moving forward
- Honest: if a destination is expensive or has visa issues, say so upfront
- Concise: reactions are 1-2 sentences, not paragraphs

RULES:
- Never suggest hard-visa destinations unless the user specifically asks
- Always frame budgets in the user's currency
- Flag exchange rate issues honestly ("Europe is incredible but INR to EUR hurts")
- Each option MUST include a brief insight, not just a label
- When reacting, be specific to what they said, not generic ("Great choice!")

CURRENT PHASE: {phase}
{phase_rules}

{knowledge_context}

Respond with a valid JSON object matching this schema:
{{
    "phase": "{phase}",
    "reaction": "brief insight reacting to previous answer (null if first turn)",
    "question": "your next question",
    "options": [{{"id": "string", "label": "string", "insight": "string", "emoji": "optional"}}],
    "multi_select": false,
    "can_free_text": true,
    "destination_hints": null,
    "thinking": null,
    "phase_complete": false
}}"""

PHASE_RULES = {
    "profile": (
        "Ask about the most important gaps. You need passport, budget comfort, "
        "and travel style to make good suggestions. Skip what you already know."
    ),
    "discovery": (
        "Understand this specific trip. Timing, companions, interests, constraints. "
        "React to each answer with a brief insight. When you have enough to start "
        "thinking about destinations, set phase_complete: true."
    ),
    "narrowing": (
        "Think out loud. Share 4-6 destination ideas in destination_hints. Include a "
        "'thinking' field explaining your reasoning. The user can react to each "
        "destination. Refine based on their reactions."
    ),
    "reveal": (
        "Present your final 3-5 curated suggestions in destination_hints. Each gets "
        "a rich hook (2-3 sentences), budget estimate, and one 'what you might not "
        "expect' insight. Set phase_complete: true."
    ),
}

FALLBACK_TURNS = {
    "profile": ConversationTurn(
        phase="profile",
        question="Let's start with the basics — what passport do you hold? This helps me suggest visa-friendly destinations.",
        options=[
            Option(id="in", label="Indian", insight="Many visa-free options in SE Asia and Caucasus"),
            Option(id="us", label="US", insight="Visa-free almost everywhere"),
            Option(id="other", label="Other", insight="Tell me and I'll check"),
        ],
    ),
    "discovery": ConversationTurn(
        phase="discovery",
        question="When are you thinking of traveling?",
        options=[
            Option(id="soon", label="Next 1-2 months", insight="Limited planning time, focus on easy destinations"),
            Option(id="quarter", label="3-6 months out", insight="Good lead time for most destinations"),
            Option(id="flexible", label="Flexible", insight="We can optimize for best season"),
        ],
    ),
    "narrowing": ConversationTurn(
        phase="narrowing",
        question="I'm having trouble generating ideas right now. Could you tell me more about what you're looking for?",
        options=[
            Option(id="retry", label="Try again", insight="I'll give it another shot"),
        ],
    ),
    "reveal": ConversationTurn(
        phase="reveal",
        question="I'm having trouble finalizing recommendations. Let me try again.",
        options=[
            Option(id="retry", label="Try again", insight="I'll give it another shot"),
        ],
    ),
}


PHASE_ORDER = ["profile", "discovery", "narrowing", "reveal"]

MIN_TURNS = {
    "profile": settings.discovery_v2_min_profile_turns,
    "discovery": settings.discovery_v2_min_discovery_turns,
    "narrowing": settings.discovery_v2_min_narrowing_turns,
    "reveal": 1,
}


def _build_prompt(
    phase: str,
    messages: List[Dict],
    profile: Dict,
    knowledge_context: str,
) -> List[Dict[str, str]]:
    phase_rules = PHASE_RULES.get(phase, "")
    system = SYSTEM_PROMPT.format(
        phase=phase,
        phase_rules=phase_rules,
        knowledge_context=knowledge_context,
    )

    prompt_messages = [{"role": "system", "content": system}]

    if profile:
        profile_summary = json.dumps(profile, indent=2)
        prompt_messages.append({
            "role": "system",
            "content": f"User profile:\n{profile_summary}",
        })

    for msg in messages:
        prompt_messages.append({"role": msg["role"], "content": msg["content"]})

    return prompt_messages


def generate_turn(
    phase: str,
    messages: List[Dict],
    profile: Dict,
    knowledge_context: str,
) -> ConversationTurn:
    """Generate the next ConversationTurn via LLM.

    Args:
        phase: Current conversation phase.
        messages: Conversation history (role/content dicts).
        profile: User profile dict (may be empty for first-time users).
        knowledge_context: Pre-built knowledge context string from context_builder.

    Returns:
        ConversationTurn with the next question, options, and optional insights.
    """
    prompt_messages = _build_prompt(phase, messages, profile, knowledge_context)

    try:
        llm = _get_llm()
        logger.info(f"ConversationEngine: generating turn (phase={phase}, msgs={len(messages)})")
        response = llm.invoke(prompt_messages)
        logger.info(f"ConversationEngine: got response ({len(response.content)} chars)")
        data = parse_json_response(response.content)
        turn = ConversationTurn(**data)
        return turn
    except Exception as e:
        logger.error(f"ConversationEngine: LLM failed — {e}")
        return FALLBACK_TURNS.get(phase, FALLBACK_TURNS["profile"])


def decide_next_phase(
    current_phase: str,
    turn_count: int,
    phase_complete: bool,
    profile: Dict,
) -> str:
    """Decide whether to transition to the next phase.

    Phase transitions are LLM-decided via phase_complete, with minimum
    turn counts as a safety net.
    """
    min_turns = MIN_TURNS.get(current_phase, 1)

    if turn_count < min_turns:
        return current_phase

    if not phase_complete:
        return current_phase

    try:
        idx = PHASE_ORDER.index(current_phase)
        if idx + 1 < len(PHASE_ORDER):
            return PHASE_ORDER[idx + 1]
    except ValueError:
        pass

    return current_phase


INTENT_EXTRACTION_PROMPT = """Based on this conversation, extract the trip intent.

Return ONLY valid JSON:
{{
    "travel_month": "month name or season",
    "duration_days": <integer>,
    "interests": ["list of interests"],
    "constraints": ["list of constraints"],
    "travel_companions": "solo|couple|family|group",
    "region_preference": "specific region or empty string",
    "budget_total": <estimated total budget number or 0>
}}

Conversation:
{conversation}"""


def extract_trip_intent_from_messages(messages: List[Dict]) -> Dict:
    """Extract structured trip intent from conversation history via LLM.

    Called when transitioning from discovery to narrowing phase.
    Populates session trip_intent for the optimizer bridge.
    """
    conversation = "\n".join(
        f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
        for m in messages
    )
    prompt = INTENT_EXTRACTION_PROMPT.format(conversation=conversation)

    try:
        llm = _get_llm()
        logger.info(f"ConversationEngine: extracting trip intent from {len(messages)} messages")
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        logger.error(f"ConversationEngine: trip intent extraction failed — {e}")
        return {
            "travel_month": "",
            "duration_days": 7,
            "interests": [],
            "constraints": [],
            "travel_companions": "solo",
            "region_preference": "",
            "budget_total": 0,
        }
