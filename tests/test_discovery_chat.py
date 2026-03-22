import pytest
from unittest.mock import patch, MagicMock
from agents.discovery_chat import (
    discovery_chat_node,
    extract_trip_intent,
    get_next_question,
    BASE_QUESTIONS,
)


def test_base_questions_exist():
    assert len(BASE_QUESTIONS) == 5
    assert "travel" in BASE_QUESTIONS[0].lower() or "when" in BASE_QUESTIONS[0].lower()


def test_discovery_asks_first_question():
    """First invocation should ask the first base question via interrupt."""
    state = {
        "user_profile": {"budget_level": "moderate"},
        "discovery_messages": [],
        "discovery_complete": False,
        "errors": [],
    }

    with patch("agents.discovery_chat.interrupt", return_value="July or August"):
        result = discovery_chat_node(state)

    assert result["discovery_complete"] is False
    assert len(result["discovery_messages"]) == 2
    assert result["discovery_messages"][0]["role"] == "assistant"
    assert result["discovery_messages"][1]["content"] == "July or August"


def test_discovery_asks_second_question():
    """With one Q&A already done, should ask the second question."""
    state = {
        "user_profile": {"budget_level": "moderate"},
        "discovery_messages": [
            {"role": "assistant", "content": BASE_QUESTIONS[0]},
            {"role": "user", "content": "July or August"},
        ],
        "discovery_complete": False,
        "errors": [],
    }

    with patch("agents.discovery_chat.interrupt", return_value="About 7 days"):
        result = discovery_chat_node(state)

    assert result["discovery_complete"] is False
    assert len(result["discovery_messages"]) == 2
    assert result["discovery_messages"][0]["role"] == "assistant"
    assert BASE_QUESTIONS[1] in result["discovery_messages"][0]["content"]


def test_extract_trip_intent_with_mock_llm():
    messages = [
        {"role": "assistant", "content": "When are you traveling?"},
        {"role": "user", "content": "July"},
        {"role": "assistant", "content": "How many days?"},
        {"role": "user", "content": "7 days"},
        {"role": "assistant", "content": "What interests you?"},
        {"role": "user", "content": "Beaches and food"},
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"travel_month": "July", "duration_days": 7, '
                '"interests": ["beaches", "food"], "constraints": [], '
                '"travel_companions": "solo", "region_preference": "", "budget_total": 0}'
    )

    with patch("agents.discovery_chat._get_llm", return_value=mock_llm):
        intent = extract_trip_intent(messages)

    assert intent["travel_month"] == "July"
    assert intent["duration_days"] == 7
    assert "beaches" in intent["interests"]


def test_extract_trip_intent_llm_failure_returns_defaults():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")

    with patch("agents.discovery_chat._get_llm", return_value=mock_llm):
        intent = extract_trip_intent([])

    assert intent["duration_days"] == 7
    assert intent["travel_companions"] == "solo"


def test_get_next_question_fallback():
    """When LLM fails, should fall back to base questions."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")

    with patch("agents.discovery_chat._get_llm", return_value=mock_llm):
        result = get_next_question({}, [], 0)

    assert result["question"] == BASE_QUESTIONS[0]
    assert result["should_complete"] is False


def test_get_next_question_completion_when_past_base():
    """When asked_count exceeds base questions and LLM fails, should complete."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")

    with patch("agents.discovery_chat._get_llm", return_value=mock_llm):
        result = get_next_question({}, [], 10)

    assert result["should_complete"] is True


def test_discovery_completes_after_max_questions():
    """After max questions answered, should extract intent and complete."""
    # Build messages with 5 Q&A pairs (max_discovery_questions = 5)
    messages = []
    for i in range(5):
        messages.append({"role": "assistant", "content": f"Question {i+1}"})
        messages.append({"role": "user", "content": f"Answer {i+1}"})

    state = {
        "user_profile": {},
        "discovery_messages": messages,
        "discovery_complete": False,
        "errors": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"travel_month": "July", "duration_days": 7, '
                '"interests": ["beaches"], "constraints": [], '
                '"travel_companions": "solo", "region_preference": "", "budget_total": 0}'
    )

    with patch("agents.discovery_chat._get_llm", return_value=mock_llm):
        result = discovery_chat_node(state)

    assert result["discovery_complete"] is True
    assert result["trip_intent"]["travel_month"] == "July"
