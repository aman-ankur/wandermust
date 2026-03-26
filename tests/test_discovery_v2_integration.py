"""End-to-end integration test for the Discovery v2 4-phase flow."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app
from api.models import ConversationTurn, Option, DestinationHint


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _mock_turn(phase, question, options, phase_complete=False, reaction=None,
               thinking=None, destination_hints=None):
    return ConversationTurn(
        phase=phase,
        question=question,
        options=[Option(id=o[0], label=o[1], insight=o[2]) for o in options],
        phase_complete=phase_complete,
        reaction=reaction,
        thinking=thinking,
        destination_hints=destination_hints,
    )


def test_full_4_phase_flow(client):
    """Walk through all 4 phases: profile -> discovery -> narrowing -> reveal."""
    turns = [
        _mock_turn("profile", "What passport?", [("in", "Indian", "SE Asia access")]),
        _mock_turn("profile", "Budget level?", [("mod", "Moderate", "₹5-8k/day")],
                   phase_complete=True, reaction="Indian passport — great options!"),
        _mock_turn("discovery", "When traveling?", [("jul", "July", "Monsoon in India")],
                   reaction="Moderate budget opens up most of Asia"),
        _mock_turn("discovery", "Interests?", [("food", "Food", "Street food tours")],
                   phase_complete=True, reaction="July is great for the Caucasus"),
        _mock_turn("narrowing", "What catches your eye?",
                   [("more", "Tell me more", "Dig deeper")],
                   phase_complete=True,
                   reaction="Here's what I'm thinking...",
                   thinking="I keep coming back to Georgia",
                   destination_hints=[
                       DestinationHint(name="Tbilisi", hook="Wine country", match_reason="Budget fit"),
                   ]),
        _mock_turn("reveal", "Pick your destination!",
                   [("pick", "Pick below", "Click to select")],
                   phase_complete=True,
                   reaction="My top picks for you!",
                   destination_hints=[
                       DestinationHint(name="Tbilisi, Georgia", hook="₹3k/day", match_reason="Perfect match"),
                   ]),
    ]
    turn_idx = [0]

    def mock_generate(*args, **kwargs):
        t = turns[turn_idx[0]]
        turn_idx[0] += 1
        return t

    with patch("api.routes.generate_turn", side_effect=mock_generate):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["session_id"]
        assert data["turn"]["phase"] == "profile"

        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Indian passport"})
        assert resp.status_code == 200

        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Moderate"})
        assert resp.status_code == 200

        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "July"})
        assert resp.status_code == 200

        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Food and culture"})
        assert resp.status_code == 200

        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Tell me more about Georgia"})
        assert resp.status_code == 200

        state_resp = client.get(f"/api/discovery/state?session_id={session_id}")
        assert state_resp.status_code == 200
        assert state_resp.json()["phase"] == "reveal"


def test_returning_user_skips_profile(client):
    """Returning user with profile in DB should start in discovery phase."""
    mock_profile = {
        "user_id": "default",
        "passport_country": "IN",
        "budget_level": "moderate",
        "travel_history": ["Japan"],
        "preferences": {},
    }
    discovery_turn = _mock_turn("discovery", "When traveling?",
                                [("jul", "July", "Good timing")])

    with patch("api.routes.HistoryDB") as MockDB, \
         patch("api.routes.generate_turn", return_value=discovery_turn):
        MockDB.return_value.get_profile.return_value = mock_profile
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.json()["turn"]["phase"] == "discovery"
