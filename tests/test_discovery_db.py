import pytest
from db import HistoryDB


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test_discovery.db")
    db = HistoryDB(db_path)
    yield db
    db.close()


def test_save_and_get_profile(test_db):
    test_db.save_profile(
        user_id="test_user",
        travel_history=["Japan", "Thailand"],
        preferences={"climate": "warm"},
        budget_level="budget",
        passport_country="IN",
    )
    profile = test_db.get_profile("test_user")
    assert profile is not None
    assert profile["user_id"] == "test_user"
    assert profile["travel_history"] == ["Japan", "Thailand"]
    assert profile["preferences"] == {"climate": "warm"}
    assert profile["budget_level"] == "budget"
    assert profile["passport_country"] == "IN"


def test_get_profile_not_found(test_db):
    profile = test_db.get_profile("nonexistent")
    assert profile is None


def test_update_profile(test_db):
    test_db.save_profile("user1", ["Japan"], {}, "moderate", "IN")
    test_db.save_profile("user1", ["Japan", "Italy"], {"pace": "fast"}, "luxury", "IN")
    profile = test_db.get_profile("user1")
    assert profile["travel_history"] == ["Japan", "Italy"]
    assert profile["budget_level"] == "luxury"


def test_save_and_get_discovery_session(test_db):
    intent = {"month": "July", "duration": 7, "interests": ["beaches"]}
    suggestions = [{"destination": "Bali", "country": "Indonesia"}]
    test_db.save_discovery_session("user1", intent, suggestions, "Bali")
    sessions = test_db.get_discovery_sessions("user1")
    assert len(sessions) == 1
    assert sessions[0]["trip_intent"]["month"] == "July"
    assert sessions[0]["chosen_destination"] == "Bali"


def test_get_discovery_sessions_empty(test_db):
    sessions = test_db.get_discovery_sessions("nobody")
    assert sessions == []


def test_multiple_discovery_sessions(test_db):
    test_db.save_discovery_session("user1", {"trip": 1}, [], "Tokyo")
    test_db.save_discovery_session("user1", {"trip": 2}, [], "Paris")
    sessions = test_db.get_discovery_sessions("user1", limit=5)
    assert len(sessions) == 2
    # Most recent first
    assert sessions[0]["chosen_destination"] == "Paris"
