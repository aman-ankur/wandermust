"""In-memory session store for discovery conversations.

Each session tracks conversation state across the 4-phase flow.
Sessions are ephemeral — persisted to DB only on completion.
"""
import uuid
from typing import Any, Dict, List, Optional


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        user_id: str = "default",
        profile: Optional[Dict] = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        initial_phase = "discovery" if profile else "profile"
        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "phase": initial_phase,
            "messages": [],
            "turn_count": 0,
            "profile": profile or {},
            "trip_intent": {},
            "destination_hints": [],
            "suggestions": [],
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def update(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.update(kwargs)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["messages"].append({"role": role, "content": content})

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
