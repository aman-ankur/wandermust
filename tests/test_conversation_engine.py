import pytest
from unittest.mock import patch, MagicMock
from api.conversation_engine import generate_turn, decide_next_phase, extract_trip_intent_from_messages
from api.models import ConversationTurn


MOCK_PROFILE_TURN_JSON = '''{
    "phase": "profile",
    "reaction": null,
    "question": "What passport do you hold?",
    "options": [
        {"id": "in", "label": "Indian", "insight": "Many visa-free destinations in SE Asia"},
        {"id": "us", "label": "US", "insight": "Visa-free almost everywhere"},
        {"id": "other", "label": "Other", "insight": "Tell me and I'll check"}
    ],
    "multi_select": false,
    "can_free_text": true,
    "phase_complete": false
}'''


def test_generate_turn_returns_conversation_turn():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=MOCK_PROFILE_TURN_JSON)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="profile",
            messages=[],
            profile={},
            knowledge_context="",
        )

    assert isinstance(turn, ConversationTurn)
    assert turn.phase == "profile"
    assert len(turn.options) == 3


def test_generate_turn_includes_reaction():
    mock_json = '''{
        "phase": "discovery",
        "reaction": "India has great visa-free options in SE Asia!",
        "question": "When are you thinking of traveling?",
        "options": [
            {"id": "summer", "label": "Summer", "insight": "Monsoon in India, dry in Europe"}
        ],
        "phase_complete": false
    }'''
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=mock_json)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="discovery",
            messages=[
                {"role": "assistant", "content": "What passport?"},
                {"role": "user", "content": "Indian"},
            ],
            profile={"passport_country": "IN"},
            knowledge_context="TRAVEL INTELLIGENCE...",
        )

    assert turn.reaction is not None
    assert "India" in turn.reaction


def test_generate_turn_llm_failure_returns_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="profile",
            messages=[],
            profile={},
            knowledge_context="",
        )

    assert isinstance(turn, ConversationTurn)
    assert turn.phase == "profile"
    assert len(turn.options) >= 1


def test_generate_turn_narrowing_has_destination_hints():
    mock_json = '''{
        "phase": "narrowing",
        "reaction": "Let me think out loud...",
        "thinking": "I keep coming back to the Caucasus",
        "question": "What catches your eye?",
        "options": [
            {"id": "more", "label": "Tell me more", "insight": "Explore deeper"}
        ],
        "destination_hints": [
            {"name": "Tbilisi, Georgia", "hook": "Visa-free, wine country", "match_reason": "Budget fit"}
        ],
        "phase_complete": false
    }'''
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=mock_json)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="narrowing",
            messages=[],
            profile={"passport_country": "IN", "budget_level": "moderate"},
            knowledge_context="TRAVEL INTELLIGENCE...",
        )

    assert turn.destination_hints is not None
    assert len(turn.destination_hints) == 1
    assert turn.thinking is not None


def test_profile_stays_if_under_min_turns():
    phase = decide_next_phase(
        current_phase="profile",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "profile"


def test_profile_transitions_to_discovery():
    phase = decide_next_phase(
        current_phase="profile",
        turn_count=2,
        phase_complete=True,
        profile={"passport_country": "IN"},
    )
    assert phase == "discovery"


def test_discovery_transitions_to_narrowing():
    phase = decide_next_phase(
        current_phase="discovery",
        turn_count=3,
        phase_complete=True,
        profile={"passport_country": "IN"},
    )
    assert phase == "narrowing"


def test_narrowing_transitions_to_reveal():
    phase = decide_next_phase(
        current_phase="narrowing",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "reveal"


def test_discovery_stays_if_llm_says_not_complete():
    phase = decide_next_phase(
        current_phase="discovery",
        turn_count=5,
        phase_complete=False,
        profile={},
    )
    assert phase == "discovery"


def test_reveal_stays_reveal():
    phase = decide_next_phase(
        current_phase="reveal",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "reveal"


def test_extract_trip_intent_returns_structured_data():
    messages = [
        {"role": "assistant", "content": "When are you traveling?"},
        {"role": "user", "content": "July"},
        {"role": "assistant", "content": "How long?"},
        {"role": "user", "content": "7 days"},
        {"role": "assistant", "content": "What excites you?"},
        {"role": "user", "content": "Food and beaches"},
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"travel_month": "July", "duration_days": 7, '
                '"interests": ["food", "beaches"], "constraints": [], '
                '"travel_companions": "solo", "region_preference": "", "budget_total": 50000}'
    )

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        intent = extract_trip_intent_from_messages(messages)

    assert intent["travel_month"] == "July"
    assert intent["duration_days"] == 7
    assert "food" in intent["interests"]


def test_extract_trip_intent_fallback_on_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        intent = extract_trip_intent_from_messages([])

    assert intent["duration_days"] == 7
    assert intent["travel_companions"] == "solo"
