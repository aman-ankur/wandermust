from models import UserProfile, DestinationSuggestion, DiscoveryState


def test_user_profile_defaults():
    p = UserProfile()
    assert p.user_id == "default"
    assert p.travel_history == []
    assert p.budget_level == "moderate"
    assert p.passport_country == "IN"


def test_user_profile_with_data():
    p = UserProfile(
        user_id="test_user",
        travel_history=["Japan", "Thailand"],
        preferences={"climate": "warm", "pace": "relaxed"},
        budget_level="budget",
        passport_country="IN",
    )
    assert p.travel_history == ["Japan", "Thailand"]
    assert p.preferences["climate"] == "warm"


def test_destination_suggestion():
    s = DestinationSuggestion(
        destination="Tbilisi, Georgia",
        country="Georgia",
        reason="Visa-free for Indian passports",
        estimated_budget_per_day=4000.0,
        best_months=[5, 6, 9, 10],
        match_score=0.87,
        tags=["culture", "food"],
    )
    assert s.destination == "Tbilisi, Georgia"
    assert s.match_score == 0.87
    assert len(s.tags) == 2


def test_destination_suggestion_defaults():
    s = DestinationSuggestion(
        destination="Tokyo, Japan",
        country="Japan",
        reason="Great food scene",
    )
    assert s.estimated_budget_per_day == 0.0
    assert s.best_months == []
    assert s.tags == []


def test_discovery_state_accepts_fields():
    state: DiscoveryState = {
        "user_profile": {"user_id": "default"},
        "onboarding_complete": True,
        "onboarding_messages": [{"role": "assistant", "content": "Hello!"}],
        "discovery_messages": [],
        "discovery_complete": False,
        "trip_intent": {},
        "suggestions": [],
        "chosen_destination": None,
        "optimizer_state": None,
        "errors": [],
    }
    assert state["onboarding_complete"] is True
    assert state["chosen_destination"] is None
