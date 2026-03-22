"""Onboarding agent — collects user travel profile via conversation.

Uses LangGraph interrupt() for human-in-the-loop chat.
Only runs for first-time users (no profile in DB).
"""
import json
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from models import DiscoveryState
from config import settings
from db import HistoryDB

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.discovery_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _llm

ONBOARDING_QUESTIONS = [
    "Welcome to Wandermust! I'd love to learn about your travel style to give you better recommendations. What countries or cities have you visited before?",
    "What kind of climate do you prefer? (warm/cold/moderate/tropical)",
    "How would you describe your travel style? (adventure/relaxation/culture/foodie/mix)",
    "What's your typical budget comfort level? (budget/moderate/luxury)",
    "What passport do you hold? (This helps me suggest visa-friendly destinations)",
]

PROFILE_EXTRACTION_PROMPT = """You are a travel profile extractor. Based on the following onboarding conversation, extract a structured user profile.

Return ONLY valid JSON with this exact schema:
{{
    "travel_history": ["list of countries/cities mentioned"],
    "preferences": {{"climate": "warm|cold|moderate|tropical", "pace": "fast|relaxed|moderate", "style": "adventure|relaxation|culture|foodie|mix"}},
    "budget_level": "budget|moderate|luxury",
    "passport_country": "2-letter ISO country code"
}}

Conversation:
{conversation}
"""


def extract_profile_from_conversation(messages: list[dict]) -> dict:
    """Use LLM to extract structured profile from onboarding conversation."""
    conversation = "\n".join(
        f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
        for m in messages
    )
    prompt = PROFILE_EXTRACTION_PROMPT.format(conversation=conversation)
    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        return json.loads(response.content)
    except Exception:
        return {
            "travel_history": [],
            "preferences": {"climate": "moderate", "pace": "relaxed", "style": "mix"},
            "budget_level": "moderate",
            "passport_country": "IN",
        }


def onboarding_node(state: DiscoveryState) -> dict:
    """LangGraph node: onboarding conversation with interrupt/resume."""
    errors = list(state.get("errors", []))
    messages = list(state.get("onboarding_messages", []))

    # Check if profile already exists
    db = HistoryDB(settings.db_path)
    profile = db.get_profile("default")
    if profile:
        return {
            "user_profile": profile,
            "onboarding_complete": True,
            "onboarding_messages": [],
            "errors": [],
        }

    # Determine which question to ask next
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    user_count = sum(1 for m in messages if m["role"] == "user")

    if assistant_count <= user_count and assistant_count < len(ONBOARDING_QUESTIONS):
        # Ask next question
        question = ONBOARDING_QUESTIONS[assistant_count]
        messages.append({"role": "assistant", "content": question})

        # Interrupt to get user response
        user_response = interrupt({"question": question, "phase": "onboarding"})

        messages.append({"role": "user", "content": user_response})

        # Check if we have more questions
        next_q_idx = assistant_count + 1
        if next_q_idx < len(ONBOARDING_QUESTIONS):
            # More questions to ask — return and let graph re-enter
            return {
                "onboarding_messages": [
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": user_response},
                ],
                "onboarding_complete": False,
                "errors": [],
            }

    # All questions answered — extract profile
    all_messages = list(state.get("onboarding_messages", [])) + [
        m for m in messages if m not in state.get("onboarding_messages", [])
    ]
    extracted = extract_profile_from_conversation(all_messages)
    profile_data = {
        "user_id": "default",
        "travel_history": extracted.get("travel_history", []),
        "preferences": extracted.get("preferences", {}),
        "budget_level": extracted.get("budget_level", "moderate"),
        "passport_country": extracted.get("passport_country", "IN"),
    }

    # Save to DB
    try:
        db.save_profile(
            user_id="default",
            travel_history=profile_data["travel_history"],
            preferences=profile_data["preferences"],
            budget_level=profile_data["budget_level"],
            passport_country=profile_data["passport_country"],
        )
    except Exception as e:
        errors.append(f"Onboarding: DB save failed — {e}")

    return {
        "user_profile": profile_data,
        "onboarding_complete": True,
        "onboarding_messages": [
            {"role": "assistant", "content": ONBOARDING_QUESTIONS[-1]},
            {"role": "user", "content": messages[-1]["content"] if messages else ""},
        ],
        "errors": errors,
    }
