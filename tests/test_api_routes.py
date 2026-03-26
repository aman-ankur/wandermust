import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app
from api.models import ConversationTurn, Option


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


MOCK_TURN = ConversationTurn(
    phase="profile",
    question="What passport do you hold?",
    options=[Option(id="in", label="Indian", insight="Great SE Asia access")],
)

MOCK_TURN_JSON = MOCK_TURN.model_dump()


def test_start_creates_session(client):
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "turn" in data
    assert data["turn"]["phase"] == "profile"


def test_start_with_existing_profile_skips_profile_phase(client):
    mock_profile = {
        "user_id": "default",
        "passport_country": "IN",
        "budget_level": "moderate",
        "travel_history": ["Japan"],
        "preferences": {"style": "culture"},
    }
    discovery_turn = ConversationTurn(
        phase="discovery",
        question="When are you traveling?",
        options=[Option(id="summer", label="Summer", insight="Hot but cheap")],
    )

    with patch("api.routes.HistoryDB") as MockDB, \
         patch("api.routes.generate_turn", return_value=discovery_turn):
        MockDB.return_value.get_profile.return_value = mock_profile
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    data = resp.json()
    assert data["turn"]["phase"] == "discovery"


def test_respond_returns_next_turn(client):
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    next_turn = ConversationTurn(
        phase="profile",
        reaction="Indian passport — great options in SE Asia!",
        question="What's your budget comfort level?",
        options=[Option(id="mod", label="Moderate", insight="₹5-8k/day")],
    )
    with patch("api.routes.generate_turn", return_value=next_turn):
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id,
            "answer": "Indian passport",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"]["reaction"] is not None


def test_respond_invalid_session(client):
    resp = client.post("/api/discovery/respond", json={
        "session_id": "nonexistent",
        "answer": "hello",
    })
    assert resp.status_code == 404


def test_get_state(client):
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    resp = client.get(f"/api/discovery/state?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] in ("profile", "discovery")


def test_get_state_invalid_session(client):
    resp = client.get("/api/discovery/state?session_id=nonexistent")
    assert resp.status_code == 404


def test_select_bridges_to_optimizer(client):
    """POST /select should bridge to optimizer and clean up session."""
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    from api.routes import _get_session_store
    store = _get_session_store()
    store.update(session_id, trip_intent={
        "travel_month": "July", "duration_days": 7,
        "interests": ["beaches"], "constraints": [],
        "travel_companions": "solo", "budget_total": 50000,
    })

    with patch("api.routes.HistoryDB"):
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
