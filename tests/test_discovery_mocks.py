from mock_data import (
    get_mock_onboarding_response,
    get_mock_discovery_response,
    get_mock_suggestions,
    get_mock_trip_intent,
    get_mock_user_profile,
)


def test_mock_onboarding_responses():
    for i in range(5):
        resp = get_mock_onboarding_response(i)
        assert isinstance(resp, str)
        assert len(resp) > 0


def test_mock_onboarding_out_of_range():
    resp = get_mock_onboarding_response(99)
    assert "defaults" in resp.lower() or "not sure" in resp.lower()


def test_mock_discovery_responses():
    for i in range(5):
        resp = get_mock_discovery_response(i)
        assert isinstance(resp, str)
        assert len(resp) > 0


def test_mock_discovery_out_of_range():
    resp = get_mock_discovery_response(99)
    assert "flexible" in resp.lower()


def test_mock_suggestions_all():
    suggestions = get_mock_suggestions()
    assert len(suggestions) >= 3
    for s in suggestions:
        assert "destination" in s
        assert "country" in s
        assert "reason" in s
        assert "match_score" in s


def test_mock_suggestions_filter():
    suggestions = get_mock_suggestions("Bali")
    assert len(suggestions) == 1
    assert "Bali" in suggestions[0]["destination"]


def test_mock_suggestions_filter_no_match():
    suggestions = get_mock_suggestions("Atlantis")
    assert suggestions == []


def test_mock_trip_intent():
    intent = get_mock_trip_intent()
    assert "travel_month" in intent
    assert "duration_days" in intent
    assert "interests" in intent
    assert isinstance(intent["interests"], list)


def test_mock_user_profile():
    profile = get_mock_user_profile()
    assert profile["user_id"] == "default"
    assert isinstance(profile["travel_history"], list)
    assert "budget_level" in profile
    assert "passport_country" in profile
