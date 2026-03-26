import pytest
from api.models import ConversationTurn, Option, DestinationHint


def test_conversation_turn_renders_options():
    """Verify ConversationTurn has the fields needed for rendering."""
    turn = ConversationTurn(
        phase="profile",
        question="What passport?",
        options=[
            Option(id="in", label="Indian", insight="SE Asia visa-free", emoji="🇮🇳"),
            Option(id="us", label="US", insight="Global access", emoji="🇺🇸"),
        ],
    )
    assert len(turn.options) == 2
    assert turn.options[0].emoji == "🇮🇳"


def test_conversation_turn_renders_destination_hints():
    turn = ConversationTurn(
        phase="narrowing",
        reaction="Let me think...",
        thinking="I keep coming back to Georgia",
        question="What catches your eye?",
        options=[Option(id="more", label="Tell me more", insight="Dig deeper")],
        destination_hints=[
            DestinationHint(
                name="Tbilisi, Georgia",
                hook="Visa-free, ₹3k/day",
                match_reason="Budget fit",
            ),
        ],
    )
    assert turn.destination_hints is not None
    assert turn.destination_hints[0].name == "Tbilisi, Georgia"


def test_conversation_turn_reveal_phase():
    turn = ConversationTurn(
        phase="reveal",
        reaction="Here are my top picks!",
        question="Which destination excites you?",
        options=[Option(id="pick", label="Pick one below", insight="Click to select")],
        destination_hints=[
            DestinationHint(name="Georgia", hook="Wine + mountains", match_reason="Perfect match"),
            DestinationHint(name="Vietnam", hook="Street food + coast", match_reason="Budget winner"),
        ],
        phase_complete=True,
    )
    assert len(turn.destination_hints) == 2
    assert turn.phase_complete is True
