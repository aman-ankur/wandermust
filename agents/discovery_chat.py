"""Discovery chat agent — asks adaptive questions about trip intent.

Uses LangGraph interrupt() for human-in-the-loop chat.
Extracts structured trip_intent from the conversation.
"""
import json
import logging
from langgraph.types import interrupt

logger = logging.getLogger("wandermust.discovery_chat")
from models import DiscoveryState
from config import settings
from agents.llm_helper import get_llm, parse_json_response

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(settings.discovery_model)
    return _llm

BASE_QUESTIONS = [
    "When are you thinking of traveling? (month or season)",
    "How many days do you have for this trip?",
    "What are you most excited about for this trip? (beaches, mountains, history, food, nightlife, nature...)",
    "Any must-have constraints? (visa-free only, direct flights, budget limit, etc.)",
    "Will you be traveling solo or with others?",
]

ADAPTIVE_QUESTION_PROMPT = """You are a travel discovery assistant. Based on the user's profile and conversation so far, generate the next question to ask about their trip intent.

User profile:
{profile}

Conversation so far:
{conversation}

Questions already asked: {asked_count}
Max questions: {max_questions}

Rules:
- If the user's profile already reveals their budget level, skip budget questions.
- If they mentioned a region or theme, ask a follow-up to narrow it down.
- Keep questions concise and friendly.
- If you have enough info (at least 3 exchanges), you may indicate completion.

Return ONLY a JSON object:
{{"question": "your next question here", "should_complete": false}}

If you have enough information, return:
{{"question": "", "should_complete": true}}
"""

INTENT_EXTRACTION_PROMPT = """You are a trip intent extractor. Based on the following conversation, extract a structured trip intent.

Return ONLY valid JSON with this exact schema:
{{
    "travel_month": "month name or season",
    "duration_days": <integer>,
    "interests": ["list of interests"],
    "constraints": ["list of constraints"],
    "travel_companions": "solo|couple|family|group",
    "region_preference": "any specific region mentioned or empty string",
    "budget_total": <estimated total budget number or 0 if not specified>
}}

Conversation:
{conversation}
"""


def get_next_question(profile: dict, messages: list[dict], asked_count: int) -> dict:
    """Use LLM to generate the next adaptive question or decide to complete."""
    conversation = "\n".join(
        f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
        for m in messages
    )
    prompt = ADAPTIVE_QUESTION_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        conversation=conversation,
        asked_count=asked_count,
        max_questions=settings.max_discovery_questions,
    )
    try:
        llm = _get_llm()
        logger.info(f"Discovery: calling LLM for next question (asked={asked_count})")
        response = llm.invoke(prompt)
        logger.info("Discovery: got next question from LLM")
        return parse_json_response(response.content)
    except Exception as e:
        logger.error(f"Discovery: LLM question generation failed — {e}")
        # Fallback to base questions
        if asked_count < len(BASE_QUESTIONS):
            return {"question": BASE_QUESTIONS[asked_count], "should_complete": False}
        return {"question": "", "should_complete": True}


def extract_trip_intent(messages: list[dict]) -> dict:
    """Use LLM to extract structured trip intent from discovery conversation."""
    conversation = "\n".join(
        f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
        for m in messages
    )
    prompt = INTENT_EXTRACTION_PROMPT.format(conversation=conversation)
    try:
        llm = _get_llm()
        logger.info(f"Discovery: extracting trip intent from {len(messages)} messages via LLM")
        response = llm.invoke(prompt)
        logger.info("Discovery: trip intent extraction complete")
        return parse_json_response(response.content)
    except Exception as e:
        logger.error(f"Discovery: trip intent extraction failed — {e}")
        return {
            "travel_month": "",
            "duration_days": 7,
            "interests": [],
            "constraints": [],
            "travel_companions": "solo",
            "region_preference": "",
            "budget_total": 0,
        }


def discovery_chat_node(state: DiscoveryState) -> dict:
    """LangGraph node: discovery conversation with interrupt/resume."""
    errors = list(state.get("errors", []))
    messages = list(state.get("discovery_messages", []))
    profile = state.get("user_profile", {})

    # Count how many Q&A exchanges we've had
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    max_questions = settings.max_discovery_questions

    # Determine next action
    if assistant_count < max_questions:
        # Use LLM to decide next question (or fallback to base questions)
        if assistant_count < len(BASE_QUESTIONS):
            # For first few questions, use base questions for reliability
            next_q = {"question": BASE_QUESTIONS[assistant_count], "should_complete": False}
        else:
            next_q = get_next_question(profile, messages, assistant_count)

        if next_q.get("should_complete") or not next_q.get("question"):
            # LLM says we have enough info
            pass
        else:
            question = next_q["question"]

            # Interrupt to get user response
            user_response = interrupt({"question": question, "phase": "discovery"})

            return {
                "discovery_messages": [
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": user_response},
                ],
                "discovery_complete": False,
                "errors": [],
            }

    # All questions done or LLM decided to complete — extract intent
    all_messages = list(state.get("discovery_messages", []))
    trip_intent = extract_trip_intent(all_messages)

    return {
        "discovery_messages": [],
        "discovery_complete": True,
        "trip_intent": trip_intent,
        "errors": errors,
    }
