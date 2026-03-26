"""Tests for Discovery v2 API routes.

Tests the FastAPI endpoints with mocked LLM personality layer.
The controller is deterministic and doesn't need mocking.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app
from api.models import ConversationTurn, Option


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_db():
    """Mock DB so tests don't depend on local travel_history.db."""
    mock = MagicMock()
    mock.get_profile.return_value = None
    with patch("api.routes._get_db", return_value=mock):
        yield mock


MOCK_PERSONALITY = {
    "reaction": None,
    "question": "What passport do you hold? This helps me figure out visa-friendly spots.",
    "option_insights": [
        "Many visa-free options in SE Asia",
        "Visa-free almost everywhere",
        "Great access across Europe",
        "Free movement across 27 countries",
        "Tell me and I'll check",
    ],
}


def test_start_creates_session(client):
    with patch("api.routes.generate_personality", return_value=MOCK_PERSONALITY):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "turn" in data
    assert data["turn"]["phase"] == "profile"
    assert data["turn"]["topic"] == "passport"
    assert len(data["turn"]["options"]) == 5


def test_start_with_existing_profile_skips_profile_phase(client, mock_db):
    mock_profile = {
        "user_id": "default",
        "passport_country": "IN",
        "budget_level": "moderate",
        "travel_history": ["Japan"],
        "preferences": {"style": "culture"},
    }
    mock_db.get_profile.return_value = mock_profile

    discovery_personality = {
        "reaction": None,
        "question": "When are you thinking of traveling?",
        "option_insights": ["Soon!", "Good lead time", "Later", "Flexible"],
    }

    with patch("api.routes.generate_personality", return_value=discovery_personality):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    data = resp.json()
    assert data["turn"]["phase"] == "discovery"
    assert data["turn"]["topic"] == "timing"


def test_respond_returns_next_turn(client):
    with patch("api.routes.generate_personality", return_value=MOCK_PERSONALITY):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    next_personality = {
        "reaction": "Indian passport -- great options in SE Asia and the Caucasus!",
        "question": "What's your budget comfort level for this trip?",
        "option_insights": ["Stretch every rupee", "Good balance", "Treat yourself", "Go all out"],
    }
    with patch("api.routes.generate_personality", return_value=next_personality):
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id,
            "answer": "Indian passport",
            "option_ids": ["in"],
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"]["reaction"] is not None
    assert data["turn"]["topic"] == "budget_level"


def test_respond_invalid_session(client):
    resp = client.post("/api/discovery/respond", json={
        "session_id": "nonexistent",
        "answer": "hello",
    })
    assert resp.status_code == 404


def test_get_state(client):
    with patch("api.routes.generate_personality", return_value=MOCK_PERSONALITY):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    resp = client.get(f"/api/discovery/state?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] in ("profile", "discovery")
    assert "known_facts" in data


def test_get_state_invalid_session(client):
    resp = client.get("/api/discovery/state?session_id=nonexistent")
    assert resp.status_code == 404


def test_select_bridges_to_optimizer(client):
    with patch("api.routes.generate_personality", return_value=MOCK_PERSONALITY):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    from api.routes import _get_session_store
    store = _get_session_store()
    store.update(session_id, trip_intent={
        "travel_month": "July", "duration_days": 7,
        "interests": ["beaches"], "constraints": [],
        "travel_companions": "solo", "budget_total": 50000,
    })

    resp = client.post("/api/discovery/select", json={
        "session_id": session_id,
        "destination": "Tbilisi, Georgia",
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["destination"] == "Tbilisi, Georgia"
    assert "optimizer_state" in data
    assert data["optimizer_state"]["destination"] == "Tbilisi, Georgia"

    assert store.get(session_id) is None


def test_select_invalid_session(client):
    resp = client.post("/api/discovery/select", json={
        "session_id": "nonexistent",
        "destination": "Tokyo",
    })
    assert resp.status_code == 404


def test_respond_phase_transition_profile_to_discovery(client):
    """Walk through all 3 profile questions and verify transition to discovery."""
    personality_responses = [
        {
            "reaction": None,
            "question": "What passport do you hold?",
            "option_insights": ["SE Asia access", "Global access", "Europe access", "EU freedom", "Tell me"],
        },
        {
            "reaction": "Indian passport -- lots of visa-free options!",
            "question": "What's your budget comfort level?",
            "option_insights": ["Stretch every rupee", "Good balance", "Treat yourself", "Go all out"],
        },
        {
            "reaction": "Mid-range is smart -- great value destinations!",
            "question": "What's your travel style?",
            "option_insights": ["Thrills", "Deep culture", "Chill vibes", "Eat everything", "Best of all"],
        },
        {
            "reaction": "Adventure lover! I know just the places.",
            "question": "When are you thinking of traveling?",
            "option_insights": ["Soon!", "Good lead time", "Later", "Flexible"],
        },
    ]

    call_count = {"n": 0}
    def mock_personality(*args, **kwargs):
        resp = personality_responses[call_count["n"]]
        call_count["n"] += 1
        return resp

    with patch("api.routes.generate_personality", side_effect=mock_personality):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
        session_id = start_resp.json()["session_id"]
        assert start_resp.json()["turn"]["topic"] == "passport"

        resp1 = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Indian", "option_ids": ["in"],
        })
        assert resp1.json()["turn"]["topic"] == "budget_level"

        resp2 = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Mid-range", "option_ids": ["moderate"],
        })
        assert resp2.json()["turn"]["topic"] == "travel_style"

        resp3 = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Adventure & outdoors", "option_ids": ["adventure"],
        })
        assert resp3.json()["turn"]["phase"] == "discovery"
        assert resp3.json()["turn"]["topic"] == "timing"
