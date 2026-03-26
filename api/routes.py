"""FastAPI routes for Discovery v2 conversation API.

Endpoints:
  POST /api/discovery/start   -- start a new conversation, returns first turn
  POST /api/discovery/respond  -- submit answer, returns next turn
  GET  /api/discovery/state    -- get current session state
  POST /api/discovery/select   -- pick a destination, bridge to optimizer
"""
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.models import (
    ConversationTurn,
    DestinationHint,
    DiscoveryRespondRequest,
    DiscoverySelectRequest,
    DiscoveryStartRequest,
    Option,
)
from api.session import SessionStore
from api.conversation_controller import (
    ControllerResult,
    build_controller_turn,
    build_first_turn,
    build_profile_from_facts,
    build_trip_intent_from_facts,
)
from api.conversation_engine import (
    FALLBACK_TURNS,
    generate_destinations,
    generate_personality,
)
from knowledge.context_builder import build_context
from config import settings
from db import HistoryDB

logger = logging.getLogger("wandermust.api.routes")

router = APIRouter(prefix="/api/discovery")

_session_store = SessionStore()


def _get_session_store() -> SessionStore:
    return _session_store


def _get_db() -> HistoryDB:
    if not hasattr(_get_db, "_instance"):
        _get_db._instance = HistoryDB(settings.db_path)
    return _get_db._instance


def _build_knowledge_context(profile: Dict) -> str:
    if not profile:
        return ""
    return build_context(
        passport=profile.get("passport_country", "IN"),
        budget=profile.get("budget_level", "moderate"),
    )


def _assemble_turn(ctrl: ControllerResult, personality: Dict[str, Any]) -> ConversationTurn:
    """Combine controller options with LLM personality into a ConversationTurn."""
    insights = personality.get("option_insights", [])
    options = []
    for i, opt_tmpl in enumerate(ctrl.option_templates):
        insight = insights[i] if i < len(insights) else ""
        options.append(Option(
            id=opt_tmpl["id"],
            label=opt_tmpl["label"],
            insight=insight or opt_tmpl["label"],
        ))

    return ConversationTurn(
        phase=ctrl.phase,
        reaction=personality.get("reaction"),
        question=personality.get("question", "Tell me more..."),
        options=options,
        multi_select=ctrl.multi_select,
        can_free_text=True,
        topic=ctrl.topic.key if ctrl.topic else None,
    )


def _assemble_destination_turn(phase: str, dest_data: Dict[str, Any]) -> ConversationTurn:
    """Build a ConversationTurn from LLM destination data."""
    hints = None
    raw_hints = dest_data.get("destination_hints")
    if raw_hints:
        hints = []
        for h in raw_hints:
            hints.append(DestinationHint(
                name=h.get("name", "Unknown"),
                hook=h.get("hook", ""),
                match_reason=h.get("match_reason", h.get("budget_hint", "")),
            ))

    raw_options = dest_data.get("options", [])
    options = []
    for o in raw_options:
        if isinstance(o, dict):
            options.append(Option(
                id=o.get("id", o.get("label", "opt")),
                label=o.get("label", "Option"),
                insight=o.get("insight", ""),
            ))

    if not options:
        fallback = FALLBACK_TURNS.get(phase, FALLBACK_TURNS["narrowing"])
        options = fallback.options

    return ConversationTurn(
        phase=phase,
        reaction=dest_data.get("reaction"),
        question=dest_data.get("question", "Here are my suggestions..."),
        options=options,
        multi_select=False,
        can_free_text=True,
        destination_hints=hints,
        thinking=dest_data.get("thinking"),
    )


@router.post("/start")
def start(request: DiscoveryStartRequest) -> Dict[str, Any]:
    t_start = time.perf_counter()
    store = _get_session_store()

    profile = None
    try:
        db = _get_db()
        profile = db.get_profile(request.user_id)
    except Exception as e:
        logger.warning(f"Failed to load profile: {e}")

    session_id = store.create(user_id=request.user_id, profile=profile)
    session = store.get(session_id)

    if profile:
        known_facts = {}
        if profile.get("passport_country"):
            known_facts["passport"] = profile["passport_country"]
        if profile.get("budget_level"):
            known_facts["budget_level"] = profile["budget_level"]
        if profile.get("preferences", {}).get("style"):
            known_facts["travel_style"] = profile["preferences"]["style"]
        store.update(session_id, known_facts=known_facts)
        session = store.get(session_id)

    ctrl = build_first_turn(session)

    if ctrl.topic is None:
        fallback = FALLBACK_TURNS.get(ctrl.phase, FALLBACK_TURNS["profile"])
        store.add_message(session_id, role="assistant", content=fallback.question)
        store.update(session_id, turn_count=1)
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[PERF] /start total: {total_ms:.0f}ms (fallback)")
        return {"session_id": session_id, "turn": fallback.model_dump()}

    t_ctx = time.perf_counter()
    knowledge_ctx = _build_knowledge_context(profile or {})
    ctx_ms = (time.perf_counter() - t_ctx) * 1000
    logger.info(f"[PERF] build_context: {ctx_ms:.1f}ms")

    option_labels = [o["label"] for o in ctrl.option_templates]
    personality = generate_personality(
        question_hint=ctrl.topic.question_hint,
        option_labels=option_labels,
        known_facts=ctrl.known_facts,
        knowledge_context=knowledge_ctx,
    )

    turn = _assemble_turn(ctrl, personality)

    store.add_message(session_id, role="assistant", content=turn.question)
    store.update(
        session_id,
        turn_count=1,
        known_facts=ctrl.known_facts,
        last_topic_key=ctrl.topic.key,
    )

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"[PERF] /start total: {total_ms:.0f}ms")

    return {"session_id": session_id, "turn": turn.model_dump()}


@router.post("/respond")
def respond(request: DiscoveryRespondRequest) -> Dict[str, Any]:
    t_start = time.perf_counter()
    store = _get_session_store()
    session = store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    store.add_message(request.session_id, role="user", content=request.answer)
    session = store.get(request.session_id)

    ctrl = build_controller_turn(
        session,
        option_ids=request.option_ids,
        free_text=request.answer,
    )

    new_turn_count = session["turn_count"] + 1

    if ctrl.phase in ("narrowing", "reveal"):
        profile = session.get("profile", {})
        knowledge_ctx = _build_knowledge_context(profile)

        dest_data = generate_destinations(
            phase=ctrl.phase,
            known_facts=ctrl.known_facts,
            profile=profile,
            knowledge_context=knowledge_ctx,
            messages=session["messages"],
        )
        turn = _assemble_destination_turn(ctrl.phase, dest_data)

        if ctrl.phase_changed:
            new_turn_count = 1

        store.add_message(request.session_id, role="assistant", content=turn.question)
        store.update(
            request.session_id,
            phase=ctrl.phase,
            turn_count=new_turn_count,
            known_facts=ctrl.known_facts,
            last_topic_key=None,
        )

        if session["phase"] == "profile" and ctrl.phase == "discovery":
            _save_profile_from_facts(request.session_id, session, ctrl.known_facts)

        if ctrl.phase == "narrowing":
            trip_intent = build_trip_intent_from_facts(ctrl.known_facts)
            store.update(request.session_id, trip_intent=trip_intent)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[PERF] /respond total: {total_ms:.0f}ms (phase={ctrl.phase}, dest)")
        return {"session_id": request.session_id, "turn": turn.model_dump()}

    if ctrl.topic is None:
        fallback = FALLBACK_TURNS.get(ctrl.phase, FALLBACK_TURNS["profile"])
        store.add_message(request.session_id, role="assistant", content=fallback.question)
        store.update(
            request.session_id,
            phase=ctrl.phase,
            turn_count=new_turn_count,
            known_facts=ctrl.known_facts,
        )
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"[PERF] /respond total: {total_ms:.0f}ms (fallback)")
        return {"session_id": request.session_id, "turn": fallback.model_dump()}

    if ctrl.phase_changed:
        new_turn_count = 1

    if session["phase"] == "profile" and ctrl.phase == "discovery":
        _save_profile_from_facts(request.session_id, session, ctrl.known_facts)

    profile = session.get("profile", {})
    knowledge_ctx = _build_knowledge_context(profile)

    option_labels = [o["label"] for o in ctrl.option_templates]
    personality = generate_personality(
        question_hint=ctrl.topic.question_hint,
        option_labels=option_labels,
        known_facts=ctrl.known_facts,
        knowledge_context=knowledge_ctx,
        last_answer=ctrl.last_answer_text,
    )

    turn = _assemble_turn(ctrl, personality)

    store.add_message(request.session_id, role="assistant", content=turn.question)
    store.update(
        request.session_id,
        phase=ctrl.phase,
        turn_count=new_turn_count,
        known_facts=ctrl.known_facts,
        last_topic_key=ctrl.topic.key,
    )

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"[PERF] /respond total: {total_ms:.0f}ms (phase={ctrl.phase}, turn={new_turn_count})")

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
        "known_facts": session.get("known_facts", {}),
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


def _save_profile_from_facts(
    session_id: str,
    session: Dict[str, Any],
    known_facts: Dict[str, Any],
) -> None:
    """Build profile from known_facts and save to DB + session. Zero LLM."""
    try:
        profile = build_profile_from_facts(known_facts, user_id=session["user_id"])
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
        logger.info(f"Profile saved from facts: {known_facts}")
    except Exception as e:
        logger.error(f"Profile save from facts failed: {e}")
