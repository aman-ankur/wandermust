import pytest
from unittest.mock import patch, MagicMock
from agents.onboarding import (
    onboarding_node,
    extract_profile_from_conversation,
    ONBOARDING_QUESTIONS,
)


def test_onboarding_skips_if_profile_exists():
    """If user profile exists in DB, onboarding is skipped."""
    mock_db = MagicMock()
    mock_db.get_profile.return_value = {
        "user_id": "default",
        "travel_history": ["Japan"],
        "preferences": {"climate": "warm"},
        "budget_level": "moderate",
        "passport_country": "IN",
        "created_at": "2026-01-01",
    }

    state = {
        "onboarding_messages": [],
        "onboarding_complete": False,
        "errors": [],
    }

    with patch("agents.onboarding.HistoryDB", return_value=mock_db):
        result = onboarding_node(state)

    assert result["onboarding_complete"] is True
    assert result["user_profile"]["travel_history"] == ["Japan"]


def test_onboarding_questions_exist():
    assert len(ONBOARDING_QUESTIONS) == 5
    assert "visited" in ONBOARDING_QUESTIONS[0].lower() or "travel" in ONBOARDING_QUESTIONS[0].lower()


def test_extract_profile_with_mock_llm():
    messages = [
        {"role": "assistant", "content": "What countries have you visited?"},
        {"role": "user", "content": "Japan, Thailand, and Italy"},
        {"role": "assistant", "content": "What climate do you prefer?"},
        {"role": "user", "content": "Warm and tropical"},
        {"role": "assistant", "content": "Travel style?"},
        {"role": "user", "content": "Culture and food"},
        {"role": "assistant", "content": "Budget level?"},
        {"role": "user", "content": "Moderate"},
        {"role": "assistant", "content": "What passport?"},
        {"role": "user", "content": "Indian passport"},
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"travel_history": ["Japan", "Thailand", "Italy"], '
                '"preferences": {"climate": "warm", "pace": "relaxed", "style": "culture-foodie"}, '
                '"budget_level": "moderate", "passport_country": "IN"}'
    )

    with patch("agents.onboarding._get_llm", return_value=mock_llm):
        profile = extract_profile_from_conversation(messages)

    assert profile["travel_history"] == ["Japan", "Thailand", "Italy"]
    assert profile["budget_level"] == "moderate"
    assert profile["passport_country"] == "IN"


def test_extract_profile_llm_failure_returns_defaults():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")

    with patch("agents.onboarding._get_llm", return_value=mock_llm):
        profile = extract_profile_from_conversation([])

    assert profile["budget_level"] == "moderate"
    assert profile["passport_country"] == "IN"
    assert profile["travel_history"] == []


def test_onboarding_asks_first_question():
    """When no profile exists and no messages yet, onboarding asks first question via interrupt."""
    mock_db = MagicMock()
    mock_db.get_profile.return_value = None

    state = {
        "onboarding_messages": [],
        "onboarding_complete": False,
        "errors": [],
    }

    # interrupt() will be called — we mock it to return a user response
    with patch("agents.onboarding.HistoryDB", return_value=mock_db), \
         patch("agents.onboarding.interrupt", return_value="Japan and Thailand"):
        result = onboarding_node(state)

    # Should have added assistant question + user response
    assert result["onboarding_complete"] is False
    assert len(result["onboarding_messages"]) == 2
    assert result["onboarding_messages"][0]["role"] == "assistant"
    assert result["onboarding_messages"][1]["role"] == "user"
    assert result["onboarding_messages"][1]["content"] == "Japan and Thailand"
