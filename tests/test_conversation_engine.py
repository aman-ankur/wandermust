"""Tests for the LLM personality layer (conversation_engine).

Tests generate_personality() and generate_destinations() with mocked LLM.
Phase transition logic is now tested in test_conversation_controller.py.
"""
import pytest
from unittest.mock import patch, MagicMock
from api.conversation_engine import (
    generate_personality,
    generate_destinations,
    FALLBACK_TURNS,
)
from api.models import ConversationTurn


MOCK_PERSONALITY_JSON = '''{
    "reaction": "Oh nice, India has amazing visa-free options in SE Asia!",
    "question": "What passport do you hold? This helps me figure out visa-friendly spots.",
    "option_insights": ["Many visa-free options in SE Asia", "Visa-free almost everywhere", "Tell me and I'll check"]
}'''


def test_generate_personality_returns_dict():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=MOCK_PERSONALITY_JSON)

    with patch("api.conversation_engine._get_personality_llm", return_value=mock_llm):
        result = generate_personality(
            question_hint="what passport they hold",
            option_labels=["Indian", "US", "Other"],
            known_facts={},
        )

    assert isinstance(result, dict)
    assert "question" in result
    assert "option_insights" in result
    assert len(result["option_insights"]) == 3


def test_generate_personality_with_last_answer():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=MOCK_PERSONALITY_JSON)

    with patch("api.conversation_engine._get_personality_llm", return_value=mock_llm):
        result = generate_personality(
            question_hint="their budget comfort level",
            option_labels=["Budget-friendly", "Mid-range", "Luxury"],
            known_facts={"passport": "Indian"},
            last_answer="Indian",
        )

    assert result["reaction"] is not None


def test_generate_personality_fallback_on_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_personality_llm", return_value=mock_llm):
        result = generate_personality(
            question_hint="what passport they hold",
            option_labels=["Indian", "US"],
            known_facts={},
        )

    assert isinstance(result, dict)
    assert "question" in result
    assert result["reaction"] is None


MOCK_DESTINATIONS_JSON = '''{
    "reaction": "Based on your love of food and beaches...",
    "question": "What catches your eye?",
    "thinking": "I keep coming back to SE Asia for this profile",
    "destination_hints": [
        {"name": "Bali, Indonesia", "hook": "Perfect mix of beaches and food", "match_reason": "Budget fit, visa-free", "budget_hint": "~$50/day"}
    ],
    "options": [
        {"id": "more", "label": "Tell me more", "insight": "Explore deeper"},
        {"id": "different", "label": "Show me different options", "insight": "I'll try other regions"}
    ]
}'''


def test_generate_destinations_returns_dict():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=MOCK_DESTINATIONS_JSON)

    with patch("api.conversation_engine._get_destination_llm", return_value=mock_llm):
        result = generate_destinations(
            phase="narrowing",
            known_facts={"passport": "Indian", "budget_level": "Mid-range"},
            profile={"passport_country": "IN", "budget_level": "moderate"},
        )

    assert isinstance(result, dict)
    assert "destination_hints" in result
    assert len(result["destination_hints"]) == 1
    assert result["thinking"] is not None


def test_generate_destinations_fallback_on_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_destination_llm", return_value=mock_llm):
        result = generate_destinations(
            phase="narrowing",
            known_facts={},
            profile={},
        )

    assert isinstance(result, dict)
    assert "question" in result


def test_fallback_turns_exist():
    for phase in ("profile", "discovery", "narrowing", "reveal"):
        assert phase in FALLBACK_TURNS
        turn = FALLBACK_TURNS[phase]
        assert isinstance(turn, ConversationTurn)
        assert len(turn.options) >= 1


def test_fallback_profile_has_topic():
    turn = FALLBACK_TURNS["profile"]
    assert turn.topic == "passport"


def test_personality_uses_fast_model():
    """Personality generation should use the fast (mini) model."""
    with patch("api.conversation_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content=MOCK_PERSONALITY_JSON)
        mock_get_llm.return_value = mock_llm

        import api.conversation_engine as eng
        eng._personality_llm = None
        eng._destination_llm = None

        generate_personality(
            question_hint="what passport they hold",
            option_labels=["Indian", "US"],
            known_facts={},
        )
        mock_get_llm.assert_called_with("gpt-4o-mini")


def test_destinations_uses_smart_model():
    """Destination generation should use the smart (full) model."""
    with patch("api.conversation_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.bind.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content='{"reaction":"nice","question":"q","thinking":"t","destination_hints":[],"options":[]}')
        mock_get_llm.return_value = mock_llm

        import api.conversation_engine as eng
        eng._personality_llm = None
        eng._destination_llm = None

        generate_destinations(
            phase="narrowing",
            known_facts={"passport": "Indian"},
            profile={"passport_country": "IN"},
        )
        mock_get_llm.assert_called_with("gpt-4o")
