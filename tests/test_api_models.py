import pytest
from api.models import Option, DestinationHint, ConversationTurn, DiscoveryStartRequest, DiscoveryRespondRequest


def test_option_creation():
    opt = Option(id="beaches", label="Beaches & coast", insight="Great in summer")
    assert opt.id == "beaches"
    assert opt.emoji is None


def test_option_with_emoji():
    opt = Option(id="food", label="Foodie", insight="Street food paradise", emoji="🍜")
    assert opt.emoji == "🍜"


def test_destination_hint_creation():
    hint = DestinationHint(
        name="Tbilisi, Georgia",
        hook="Visa-free, ₹3k/day, wine country",
        match_reason="Budget + food + culture trifecta",
    )
    assert hint.name == "Tbilisi, Georgia"


def test_conversation_turn_minimal():
    turn = ConversationTurn(
        phase="profile",
        question="What passport do you hold?",
        options=[Option(id="in", label="Indian", insight="Many visa-free options")],
    )
    assert turn.phase == "profile"
    assert turn.reaction is None
    assert turn.destination_hints is None
    assert turn.multi_select is False
    assert turn.can_free_text is True
    assert turn.phase_complete is False


def test_conversation_turn_full():
    turn = ConversationTurn(
        phase="narrowing",
        reaction="Great choices!",
        question="What catches your eye?",
        options=[Option(id="more", label="Tell me more", insight="Explore deeper")],
        destination_hints=[
            DestinationHint(name="Georgia", hook="Wine country", match_reason="Budget fit")
        ],
        thinking="I keep coming back to the Caucasus...",
        phase_complete=False,
        multi_select=True,
    )
    assert turn.thinking is not None
    assert len(turn.destination_hints) == 1


def test_conversation_turn_valid_phases():
    for phase in ("profile", "discovery", "narrowing", "reveal"):
        turn = ConversationTurn(
            phase=phase,
            question="Test?",
            options=[Option(id="a", label="A", insight="a")],
        )
        assert turn.phase == phase


def test_start_request():
    req = DiscoveryStartRequest(user_id="default")
    assert req.user_id == "default"


def test_respond_request():
    req = DiscoveryRespondRequest(
        session_id="abc-123",
        answer="July",
    )
    assert req.session_id == "abc-123"
    assert req.answer == "July"
