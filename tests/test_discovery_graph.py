import pytest
from unittest.mock import patch, MagicMock
from discovery_graph import (
    build_discovery_graph,
    _should_continue_onboarding,
    _should_continue_discovery,
)


def test_should_continue_onboarding_complete():
    state = {"onboarding_complete": True}
    assert _should_continue_onboarding(state) == "discovery_chat"


def test_should_continue_onboarding_not_complete():
    state = {"onboarding_complete": False}
    assert _should_continue_onboarding(state) == "onboarding"


def test_should_continue_discovery_complete():
    state = {"discovery_complete": True}
    assert _should_continue_discovery(state) == "suggestion_generator"


def test_should_continue_discovery_not_complete():
    state = {"discovery_complete": False}
    assert _should_continue_discovery(state) == "discovery_chat"


def test_discovery_graph_builds_demo_mode():
    """Demo mode graph should compile without error."""
    g = build_discovery_graph(demo=True)
    assert g is not None


def test_discovery_graph_builds_live_mode():
    """Live mode graph should compile without error."""
    g = build_discovery_graph(demo=False)
    assert g is not None


def test_demo_discovery_graph_full_run():
    """Integration test: run full discovery graph in demo mode."""
    g = build_discovery_graph(demo=True)
    config = {"configurable": {"thread_id": "test-demo-run"}}

    result = g.invoke(
        {
            "onboarding_complete": False,
            "onboarding_messages": [],
            "discovery_messages": [],
            "discovery_complete": False,
            "errors": [],
        },
        config=config,
    )

    # Should have completed onboarding
    assert result.get("onboarding_complete") is True
    assert result.get("user_profile") is not None

    # Should have completed discovery
    assert result.get("discovery_complete") is True
    assert result.get("trip_intent") is not None

    # Should have suggestions
    suggestions = result.get("suggestions", [])
    assert len(suggestions) >= 1
    assert "destination" in suggestions[0]


def test_demo_discovery_graph_skips_onboarding_if_profile():
    """If onboarding_complete is True and profile exists, should skip to discovery."""
    g = build_discovery_graph(demo=True)
    config = {"configurable": {"thread_id": "test-skip-onboard"}}

    result = g.invoke(
        {
            "user_profile": {"user_id": "default", "travel_history": ["Japan"]},
            "onboarding_complete": True,
            "onboarding_messages": [],
            "discovery_messages": [],
            "discovery_complete": False,
            "errors": [],
        },
        config=config,
    )

    assert result.get("discovery_complete") is True
    assert len(result.get("suggestions", [])) >= 1
