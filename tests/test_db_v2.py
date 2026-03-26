import pytest
import json
from db import HistoryDB


def test_save_discovery_session_with_conversation_history():
    db = HistoryDB(":memory:")
    history = [
        {"role": "assistant", "content": "What passport?"},
        {"role": "user", "content": "Indian"},
    ]
    db.save_discovery_session(
        user_id="default",
        trip_intent={"travel_month": "July"},
        suggestions=[{"destination": "Georgia"}],
        chosen_destination="Tbilisi, Georgia",
        conversation_history=history,
    )
    sessions = db.get_discovery_sessions("default")
    assert len(sessions) == 1
    assert sessions[0]["conversation_history"] == history


def test_save_discovery_session_without_conversation_history():
    """Backward compatibility — conversation_history is optional."""
    db = HistoryDB(":memory:")
    db.save_discovery_session(
        user_id="default",
        trip_intent={},
        suggestions=[],
        chosen_destination="Tokyo",
    )
    sessions = db.get_discovery_sessions("default")
    assert len(sessions) == 1
    assert sessions[0]["conversation_history"] == []
