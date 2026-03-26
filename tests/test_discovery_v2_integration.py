"""End-to-end integration test for the Discovery v2 hybrid conversation flow.

Tests the full profile -> discovery -> narrowing flow with mocked LLM personality.
The controller is deterministic and runs unmocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app


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


def _make_personality(reaction, question, num_insights=5):
    return {
        "reaction": reaction,
        "question": question,
        "option_insights": [f"insight {i}" for i in range(num_insights)],
    }


def _make_destinations(reaction, question, destinations, options):
    return {
        "reaction": reaction,
        "question": question,
        "thinking": "Reasoning about destinations...",
        "destination_hints": destinations,
        "options": options,
    }


def test_full_profile_to_narrowing_flow(client):
    """Walk through profile (3 turns) -> discovery (4 turns) -> narrowing."""
    personality_queue = [
        _make_personality(None, "What passport do you hold?"),
        _make_personality("Indian passport -- great SE Asia access!", "What's your budget comfort level?", 4),
        _make_personality("Mid-range is smart!", "What's your travel style?"),
        _make_personality("Adventure lover!", "When are you thinking of traveling?", 4),
        _make_personality("Soon -- exciting!", "Who are you traveling with?", 4),
        _make_personality("Solo trip -- love it!", "What activities excite you?", 6),
        _make_personality("Great picks!", "Any deal-breakers to avoid?", 5),
    ]
    p_idx = {"n": 0}

    def mock_personality(*args, **kwargs):
        resp = personality_queue[p_idx["n"]]
        p_idx["n"] += 1
        return resp

    dest_response = _make_destinations(
        "Based on your profile, here are my top picks...",
        "What catches your eye?",
        [
            {"name": "Tbilisi, Georgia", "hook": "Wine country", "match_reason": "Budget fit", "budget_hint": "$50/day"},
            {"name": "Bali, Indonesia", "hook": "Beaches + culture", "match_reason": "Adventure fit", "budget_hint": "$40/day"},
        ],
        [
            {"id": "more", "label": "Tell me more", "insight": "Explore deeper"},
            {"id": "different", "label": "Show me different", "insight": "Try other regions"},
        ],
    )

    with patch("api.routes.generate_personality", side_effect=mock_personality), \
         patch("api.routes.generate_destinations", return_value=dest_response):

        # Start
        resp = client.post("/api/discovery/start", json={"user_id": "default"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        assert resp.json()["turn"]["phase"] == "profile"
        assert resp.json()["turn"]["topic"] == "passport"

        # Profile: passport
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Indian", "option_ids": ["in"],
        })
        assert resp.json()["turn"]["topic"] == "budget_level"

        # Profile: budget
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Mid-range", "option_ids": ["moderate"],
        })
        assert resp.json()["turn"]["topic"] == "travel_style"

        # Profile: style -> transitions to discovery
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Adventure & outdoors", "option_ids": ["adventure"],
        })
        assert resp.json()["turn"]["phase"] == "discovery"
        assert resp.json()["turn"]["topic"] == "timing"

        # Discovery: timing
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Next 1-2 months", "option_ids": ["soon"],
        })
        assert resp.json()["turn"]["topic"] == "companions"

        # Discovery: companions
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Solo", "option_ids": ["solo"],
        })
        assert resp.json()["turn"]["topic"] == "interests"
        assert resp.json()["turn"]["multi_select"] is True

        # Discovery: interests (multi-select)
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Street food, Nature",
            "option_ids": ["food", "nature"],
        })
        # interests filled -> all required discovery facts filled -> narrowing
        assert resp.json()["turn"]["phase"] == "narrowing"
        assert resp.json()["turn"]["destination_hints"] is not None
        assert len(resp.json()["turn"]["destination_hints"]) == 2

        # Verify state
        state = client.get(f"/api/discovery/state?session_id={session_id}")
        assert state.json()["phase"] == "narrowing"
        facts = state.json()["known_facts"]
        assert facts["passport"] == "Indian"
        assert facts["budget_level"] == "Mid-range"
        assert facts["timing"] == "Next 1-2 months"
        assert facts["companions"] == "Solo"
        assert "Street food & local cuisine" in facts["interests"]


def test_returning_user_skips_profile(client, mock_db):
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
        "option_insights": ["Soon", "Later", "Way later", "Flexible"],
    }

    with patch("api.routes.generate_personality", return_value=discovery_personality):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.json()["turn"]["phase"] == "discovery"
    assert resp.json()["turn"]["topic"] == "timing"


def test_options_always_match_topic(client):
    """Verify that options returned always correspond to the current topic."""
    personality_queue = [
        _make_personality(None, "What passport?"),
        _make_personality("Great!", "Budget?", 4),
    ]
    p_idx = {"n": 0}

    def mock_personality(*args, **kwargs):
        resp = personality_queue[p_idx["n"]]
        p_idx["n"] += 1
        return resp

    with patch("api.routes.generate_personality", side_effect=mock_personality):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})
        turn = resp.json()["turn"]
        assert turn["topic"] == "passport"
        option_ids = [o["id"] for o in turn["options"]]
        assert "in" in option_ids
        assert "us" in option_ids

        session_id = resp.json()["session_id"]
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id, "answer": "Indian", "option_ids": ["in"],
        })
        turn = resp.json()["turn"]
        assert turn["topic"] == "budget_level"
        option_ids = [o["id"] for o in turn["options"]]
        assert "budget" in option_ids
        assert "moderate" in option_ids
        assert "luxury" in option_ids


def test_multi_select_for_interests(client):
    """Verify multi_select=true for interests topic."""
    # start(1) + 3 profile responds + 2 discovery responds = 6 personality calls
    personality_responses = [_make_personality(None, "Q?") for _ in range(6)]
    p_idx = {"n": 0}

    def mock_personality(*args, **kwargs):
        resp = personality_responses[p_idx["n"]]
        p_idx["n"] += 1
        return resp

    with patch("api.routes.generate_personality", side_effect=mock_personality):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})
        session_id = resp.json()["session_id"]

        # Walk through profile
        for oid in ["in", "moderate", "adventure"]:
            resp = client.post("/api/discovery/respond", json={
                "session_id": session_id, "answer": "x", "option_ids": [oid],
            })

        # Walk through discovery to interests
        for oid in ["soon", "solo"]:
            resp = client.post("/api/discovery/respond", json={
                "session_id": session_id, "answer": "x", "option_ids": [oid],
            })

        turn = resp.json()["turn"]
        assert turn["topic"] == "interests"
        assert turn["multi_select"] is True
