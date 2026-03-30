"""Conversation engine — LLM personality layer.

Two focused functions:
  - generate_personality(): Phrase questions with personality for profile/discovery
  - generate_destinations(): Generate destination suggestions for narrowing/reveal

The deterministic controller (conversation_controller.py) handles all logic.
This module only adds natural language flair.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from api.models import ConversationTurn, Option
from agents.llm_helper import get_llm, parse_json_response
from config import settings

logger = logging.getLogger("wandermust.conversation_engine")


def build_conversation_summary(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    """Build a compact conversation summary from recent messages."""
    if not messages:
        return ""
    recent = messages[-max_turns:]
    lines = []
    for msg in recent:
        role = "You" if msg["role"] == "assistant" else "User"
        lines.append(f"{role}: {msg['content']}")
    return "Recent conversation:\n" + "\n".join(lines)


_personality_llm = None
_destination_llm = None


def _get_personality_llm():
    global _personality_llm
    if _personality_llm is None:
        _personality_llm = get_llm(settings.discovery_v2_personality_model).bind(
            response_format={"type": "json_object"},
            max_tokens=512,
        )
    return _personality_llm


def _get_destination_llm():
    global _destination_llm
    if _destination_llm is None:
        _destination_llm = get_llm(settings.discovery_v2_destination_model).bind(
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
    return _destination_llm


PERSONALITY_FEW_SHOT = """
=== EXAMPLES ===
Example 1 — First question, no prior context:
Input: topic="passport", options=["Indian","US","UK","EU/Schengen","Other"], last_answer=null
Output: {{"reaction": null, "question": "First things first -- where's home base? Your passport is basically your travel superpower, and I need to know what I'm working with.", "option_insights": ["SE Asia is your playground -- visa-free gems like Thailand, Georgia, Bali", "You've got the golden ticket -- 185+ countries visa-free", "Strong across Europe and Commonwealth", "27 countries, zero borders, pure freedom", "No worries -- just tell me and I'll map out your options"]}}

Example 2 — Budget question after knowing passport:
Input: topic="budget_level", known_facts={{passport: "Indian"}}, last_answer="Indian"
Output: {{"reaction": "Indian passport -- solid! You've got visa-free access to some of my favorite corners of the world. Thailand, Georgia, Indonesia -- all waiting for you.", "question": "Now the real talk -- how do you like to travel? Are we stretching every rupee or treating ourselves?", "option_insights": ["2-3K/day gets you incredible experiences across SE Asia and the Caucasus", "4-6K/day opens up Turkey, Georgia, Eastern Europe comfortably", "8-12K/day for boutique stays and curated experiences", "15K+/day -- we're talking Maldives overwater villas and Swiss chalets"]}}
=== END EXAMPLES ==="""


PERSONALITY_PROMPT = """You are a well-traveled friend who's been to 60+ countries. Opinionated, insightful, honest.
You give advice that feels personal, not generic. Reference specific things the user said earlier.

User context: {known_facts_summary}
{conversation_history}
{knowledge_context}

Topic to ask about: {question_hint}
Options (keep these EXACT labels, do NOT change them): {option_labels}
User's last answer: {last_answer}

Personalize option_insights using what you know about this user. Reference their specific preferences, not generic descriptions.

{few_shot}

Return ONLY valid JSON:
{{"reaction": "1-2 specific sentences reacting to their last answer referencing what they actually said, or null if first turn", "question": "natural phrasing of the question that flows from the conversation", "option_insights": ["one personalized insight per option, same order as options"]}}"""


DESTINATION_FEW_SHOT = """
=== EXAMPLE ===
Input: Indian passport, mid-range budget, loves food + culture, traveling with partner, next 1-2 months
Output: {{"reaction": "A foodie couple trip on a mid-range budget with just a month to plan? I know exactly where to send you.", "question": "I've picked places where you two can eat your way through incredible cultures without breaking the bank. Here are my top picks:", "thinking": "Indian passport means visa-free SE Asia is ideal for short planning windows. Mid-range budget + food focus points to countries with incredible street food scenes. Couple trip means romantic but not resort-y.", "destination_hints": [{{"name": "Hanoi, Vietnam", "hook": "The best food city in SE Asia, full stop. You'll spend mornings in French-colonial cafes, afternoons lost in the Old Quarter, and evenings at street-side pho stalls where a life-changing bowl costs 80 rupees.", "match_reason": "Visa-free for Indians, legendary food scene, incredibly affordable for mid-range budgets", "budget_hint": "~3,500-5,000 INR/day for two including meals, stays, and experiences"}}], "options": [{{"id": "more_hanoi", "label": "Tell me more about Hanoi", "insight": "I've barely scratched the surface -- egg coffee alone is worth the trip"}}]}}
=== END EXAMPLE ==="""


DESTINATION_PROMPT = """You are a well-traveled friend who's been to 60+ countries. Opinionated, insightful, honest.
You know this user's preferences from your conversation and tailor every suggestion to them.

User profile:
{profile_summary}

Trip preferences:
{trip_summary}

{conversation_history}
{knowledge_context}

Phase: {phase}
{phase_instructions}

{few_shot}

Return ONLY valid JSON:
{{"reaction": "1-2 sentences connecting to what they told you", "question": "your question or presentation", "thinking": "your reasoning about why these destinations fit THIS specific user", "destination_hints": [{{"name": "City, Country", "hook": "2-3 sentence pitch referencing their interests", "match_reason": "why it fits their stated preferences", "budget_hint": "rough budget estimate in their currency"}}], "options": [{{"id": "id", "label": "Label", "insight": "1 sentence"}}]}}"""

DESTINATION_PHASE_INSTRUCTIONS = {
    "narrowing": (
        "Suggest 4-6 destinations that match their profile. Include a 'thinking' field. "
        "Options should be reactions: 'Tell me more about X', 'Love these!', 'Show me different options', 'Not quite right'."
    ),
    "reveal": (
        "The user has narrowed down their interests from the previous suggestions. "
        "Look at what they said in the conversation to understand which destinations they liked. "
        "Present final 3-5 curated picks focused on what excited them. Each gets a rich hook (2-3 sentences), "
        "a detailed budget breakdown in their currency, best time to visit, and one surprising insider tip. "
        "IMPORTANT: This is the FINAL reveal. Do NOT generate 'Tell me more' options. "
        "You MUST use exactly these options: "
        '[{"id": "sold", "label": "I\'m sold!", "insight": "Let\'s lock it in"}, '
        '{"id": "compare", "label": "Compare top 2", "insight": "Side by side breakdown"}, '
        '{"id": "start_over", "label": "Start over", "insight": "Back to square one"}]'
    ),
}


FALLBACK_TURNS = {
    "profile": ConversationTurn(
        phase="profile",
        question="Let's start with the basics -- what passport do you hold? This helps me suggest visa-friendly destinations.",
        options=[
            Option(id="in", label="Indian", insight="Many visa-free options in SE Asia and Caucasus"),
            Option(id="us", label="US", insight="Visa-free almost everywhere"),
            Option(id="uk", label="UK", insight="Great access across Europe and beyond"),
            Option(id="eu", label="EU/Schengen", insight="Free movement across 27 countries"),
            Option(id="other", label="Other", insight="Tell me and I'll check"),
        ],
        topic="passport",
    ),
    "discovery": ConversationTurn(
        phase="discovery",
        question="When are you thinking of traveling?",
        options=[
            Option(id="soon", label="Next 1-2 months", insight="Limited planning time, focus on easy destinations"),
            Option(id="quarter", label="3-6 months out", insight="Good lead time for most destinations"),
            Option(id="flexible", label="Flexible", insight="We can optimize for best season"),
        ],
        topic="timing",
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


def generate_personality(
    question_hint: str,
    option_labels: List[str],
    known_facts: Dict[str, Any],
    knowledge_context: str = "",
    last_answer: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Ask the LLM to phrase a question with personality.

    Returns dict with: reaction, question, option_insights
    Falls back to simple phrasing on LLM failure.
    """
    facts_summary = ", ".join(f"{k}: {v}" for k, v in known_facts.items()) if known_facts else "none yet"
    labels_str = ", ".join(option_labels)
    conversation_history = build_conversation_summary(messages or [])

    prompt = PERSONALITY_PROMPT.format(
        known_facts_summary=facts_summary,
        conversation_history=conversation_history,
        knowledge_context=knowledge_context or "",
        question_hint=question_hint,
        option_labels=labels_str,
        last_answer=last_answer or "(first question)",
        few_shot=PERSONALITY_FEW_SHOT,
    )

    try:
        llm = _get_personality_llm()
        logger.info(f"ConversationEngine: generating personality (hint={question_hint})")
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        llm_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[PERF] LLM generate_personality: {llm_ms:.0f}ms")

        data = parse_json_response(response.content)
        return {
            "reaction": data.get("reaction"),
            "question": data.get("question", f"Tell me about {question_hint}"),
            "option_insights": data.get("option_insights", []),
        }
    except Exception as e:
        logger.error(f"ConversationEngine: personality generation failed -- {e}")
        return {
            "reaction": None,
            "question": f"Tell me about {question_hint}",
            "option_insights": [],
        }


def generate_destinations(
    phase: str,
    known_facts: Dict[str, Any],
    profile: Dict[str, Any],
    knowledge_context: str = "",
    messages: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Generate destination suggestions for narrowing/reveal phases.

    Returns dict with: reaction, question, thinking, destination_hints, options
    """
    profile_summary = json.dumps(profile, indent=2) if profile else "{}"
    trip_summary = json.dumps(known_facts, indent=2)
    phase_instructions = DESTINATION_PHASE_INSTRUCTIONS.get(phase, "")
    conversation_history = build_conversation_summary(messages or [])

    prompt = DESTINATION_PROMPT.format(
        profile_summary=profile_summary,
        trip_summary=trip_summary,
        conversation_history=conversation_history,
        knowledge_context=knowledge_context or "",
        phase=phase,
        phase_instructions=phase_instructions,
        few_shot=DESTINATION_FEW_SHOT,
    )

    try:
        llm = _get_destination_llm()
        logger.info(f"ConversationEngine: generating destinations (phase={phase})")
        t0 = time.perf_counter()
        response = llm.invoke(prompt)
        llm_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[PERF] LLM generate_destinations: {llm_ms:.0f}ms")

        data = parse_json_response(response.content)
        return data
    except Exception as e:
        logger.error(f"ConversationEngine: destination generation failed -- {e}")
        fallback = FALLBACK_TURNS.get(phase, FALLBACK_TURNS["narrowing"])
        return {
            "reaction": fallback.reaction,
            "question": fallback.question,
            "thinking": None,
            "destination_hints": None,
            "options": [o.model_dump() for o in fallback.options],
        }
