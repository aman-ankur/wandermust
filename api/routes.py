"""FastAPI routes for Discovery v2 conversation API.

Endpoints:
  POST /api/discovery/start   — start a new conversation, returns first turn
  POST /api/discovery/respond  — submit answer, returns next turn
  GET  /api/discovery/state    — get current session state
  POST /api/discovery/select   — pick a destination, bridge to optimizer
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.models import (
    ConversationTurn,
    DiscoveryRespondRequest,
    DiscoverySelectRequest,
    DiscoveryStartRequest,
)
from api.session import SessionStore
from api.conversation_engine import (
    generate_turn,
    decide_next_phase,
    extract_trip_intent_from_messages,
)
from knowledge.context_builder import build_context
from agents.llm_helper import get_llm, parse_json_response
from config import settings
from db import HistoryDB

logger = logging.getLogger("wandermust.api.routes")

router = APIRouter(prefix="/api/discovery")

_session_store = SessionStore()


def _get_session_store() -> SessionStore:
    return _session_store


def _get_db() -> HistoryDB:
    """Get a shared HistoryDB instance. Reuses connection."""
    if not hasattr(_get_db, "_instance"):
        _get_db._instance = HistoryDB(settings.db_path)
    return _get_db._instance


@router.post("/start")
def start(request: DiscoveryStartRequest) -> Dict[str, Any]:
    store = _get_session_store()

    profile = None
    try:
        db = _get_db()
        profile = db.get_profile(request.user_id)
    except Exception as e:
        logger.warning(f"Failed to load profile: {e}")

    session_id = store.create(user_id=request.user_id, profile=profile)
    session = store.get(session_id)

    knowledge_ctx = ""
    if profile:
        knowledge_ctx = build_context(
            passport=profile.get("passport_country", "IN"),
            budget=profile.get("budget_level", "moderate"),
        )

    turn = generate_turn(
        phase=session["phase"],
        messages=[],
        profile=profile or {},
        knowledge_context=knowledge_ctx,
    )

    store.add_message(session_id, role="assistant", content=turn.question)
    store.update(session_id, turn_count=1)

    return {"session_id": session_id, "turn": turn.model_dump()}


@router.post("/respond")
def respond(request: DiscoveryRespondRequest) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    store.add_message(request.session_id, role="user", content=request.answer)

    profile = session["profile"]
    knowledge_ctx = ""
    if profile:
        passport = profile.get("passport_country", "IN")
        budget = profile.get("budget_level", "moderate")
        knowledge_ctx = build_context(passport=passport, budget=budget)

    turn = generate_turn(
        phase=session["phase"],
        messages=session["messages"],
        profile=profile,
        knowledge_context=knowledge_ctx,
    )

    new_turn_count = session["turn_count"] + 1
    next_phase = decide_next_phase(
        current_phase=session["phase"],
        turn_count=new_turn_count,
        phase_complete=turn.phase_complete,
        profile=profile,
    )

    if next_phase != session["phase"]:
        new_turn_count = 0
        turn.phase = next_phase

    store.add_message(request.session_id, role="assistant", content=turn.question)
    store.update(
        request.session_id,
        phase=next_phase,
        turn_count=new_turn_count,
    )

    if session["phase"] == "profile" and next_phase == "discovery":
        _extract_and_save_profile(request.session_id, session)

    if session["phase"] == "discovery" and next_phase == "narrowing":
        trip_intent = extract_trip_intent_from_messages(session["messages"])
        store.update(request.session_id, trip_intent=trip_intent)

    return {"session_id": request.session_id, "turn": turn.model_dump()}


@router.get("/state")
def get_state(session_id: str) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "phase": session["phase"],
        "turn_count": session["turn_count"],
        "message_count": len(session["messages"]),
    }


@router.post("/select")
def select(request: DiscoverySelectRequest) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from agents.discovery_bridge import build_optimizer_state

    trip_intent = session.get("trip_intent", {})
    optimizer_state = build_optimizer_state(request.destination, trip_intent)

    try:
        db = _get_db()
        db.save_discovery_session(
            session["user_id"],
            trip_intent,
            session.get("suggestions", []),
            request.destination,
        )
    except Exception as e:
        logger.warning(f"Failed to save discovery session: {e}")

    store.delete(request.session_id)

    return {"destination": request.destination, "optimizer_state": optimizer_state}


_PROFILE_EXTRACTION_PROMPT = """Extract a user travel profile from this conversation.

Return ONLY valid JSON:
{{
    "travel_history": ["countries/cities"],
    "preferences": {{"climate": "warm|cold|moderate|tropical", "pace": "fast|relaxed|moderate", "style": "adventure|relaxation|culture|foodie|mix"}},
    "budget_level": "budget|moderate|luxury",
    "passport_country": "2-letter ISO code"
}}

Conversation:
{conversation}"""


def _extract_and_save_profile(session_id: str, session: Dict) -> None:
    """Extract profile from conversation via LLM and save to DB + session."""
    try:
        conversation = "\n".join(
            f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
            for m in session["messages"]
        )
        prompt = _PROFILE_EXTRACTION_PROMPT.format(conversation=conversation)
        llm = get_llm(settings.discovery_v2_model)
        response = llm.invoke(prompt)
        extracted = parse_json_response(response.content)

        profile = {
            "user_id": session["user_id"],
            "travel_history": extracted.get("travel_history", []),
            "preferences": extracted.get("preferences", {}),
            "budget_level": extracted.get("budget_level", "moderate"),
            "passport_country": extracted.get("passport_country", "IN"),
        }
        db = _get_db()
        db.save_profile(
            session["user_id"],
            profile["travel_history"],
            profile["preferences"],
            profile["budget_level"],
            profile["passport_country"],
        )
        store = _get_session_store()
        store.update(session_id, profile=profile)
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}")
