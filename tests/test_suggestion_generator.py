import pytest
from unittest.mock import patch, MagicMock
from agents.suggestion_generator import (
    suggestion_generator_node,
    generate_suggestions,
)


def test_generate_suggestions_with_mock_llm():
    profile = {
        "travel_history": ["Japan", "Thailand"],
        "preferences": {"climate": "warm"},
        "budget_level": "moderate",
        "passport_country": "IN",
    }
    intent = {
        "travel_month": "July",
        "duration_days": 7,
        "interests": ["beaches", "food"],
        "constraints": ["visa-free"],
        "travel_companions": "couple",
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"destination": "Bali, Indonesia", "country": "Indonesia", '
                '"reason": "Visa-free, great beaches", "estimated_budget_per_day": 5000, '
                '"best_months": [5, 6, 7, 8], "match_score": 0.9, "tags": ["beaches"]}]'
    )

    with patch("agents.suggestion_generator._get_llm", return_value=mock_llm):
        suggestions = generate_suggestions(profile, intent)

    assert len(suggestions) == 1
    assert suggestions[0]["destination"] == "Bali, Indonesia"
    assert suggestions[0]["match_score"] == 0.9


def test_generate_suggestions_llm_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")

    with patch("agents.suggestion_generator._get_llm", return_value=mock_llm):
        suggestions = generate_suggestions({}, {})

    assert suggestions == []


def test_suggestion_generator_node_success():
    state = {
        "user_profile": {"passport_country": "IN", "budget_level": "moderate",
                         "travel_history": ["Japan"]},
        "trip_intent": {"travel_month": "July", "interests": ["beaches"]},
        "errors": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"destination": "Bali, Indonesia", "country": "Indonesia", '
                '"reason": "Great match", "estimated_budget_per_day": 5000, '
                '"best_months": [7, 8], "match_score": 0.9, "tags": ["beaches"]}, '
                '{"destination": "Da Nang, Vietnam", "country": "Vietnam", '
                '"reason": "Affordable beaches", "estimated_budget_per_day": 3500, '
                '"best_months": [5, 6, 7], "match_score": 0.85, "tags": ["beaches", "food"]}]'
    )

    with patch("agents.suggestion_generator._get_llm", return_value=mock_llm):
        result = suggestion_generator_node(state)

    assert len(result["suggestions"]) == 2
    assert result["suggestions"][0]["destination"] == "Bali, Indonesia"


def test_suggestion_generator_node_no_suggestions():
    state = {
        "user_profile": {},
        "trip_intent": {},
        "errors": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")

    with patch("agents.suggestion_generator._get_llm", return_value=mock_llm):
        result = suggestion_generator_node(state)

    assert result["suggestions"] == []
    assert any("no suggestions" in e.lower() for e in result["errors"])


def test_generate_suggestions_invalid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not valid json at all")

    with patch("agents.suggestion_generator._get_llm", return_value=mock_llm):
        suggestions = generate_suggestions({}, {})

    assert suggestions == []
