"""DEPRECATED: Discovery v2 uses FastAPI routes instead of LangGraph.

This module is kept for reference. New discovery flow is in api/routes.py.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models import DiscoveryState
from agents.onboarding import onboarding_node
from agents.discovery_chat import discovery_chat_node
from agents.suggestion_generator import suggestion_generator_node
from agents.discovery_bridge import bridge_node


def _should_continue_onboarding(state: DiscoveryState) -> str:
    """Route after onboarding: loop back if not complete, else proceed."""
    if state.get("onboarding_complete"):
        return "discovery_chat"
    return "onboarding"


def _should_continue_discovery(state: DiscoveryState) -> str:
    """Route after discovery chat: loop back if not complete, else proceed."""
    if state.get("discovery_complete"):
        return "suggestion_generator"
    return "discovery_chat"


def build_discovery_graph(demo: bool = False):
    """Build and compile the discovery LangGraph with checkpointer for interrupt/resume."""
    if demo:
        from agents.mock_agents import (
            mock_onboarding_node,
            mock_discovery_chat_node,
            mock_suggestion_generator_node,
            mock_bridge_node,
        )
        onboarding_fn = mock_onboarding_node
        discovery_fn = mock_discovery_chat_node
        suggestion_fn = mock_suggestion_generator_node
        bridge_fn = mock_bridge_node
    else:
        onboarding_fn = onboarding_node
        discovery_fn = discovery_chat_node
        suggestion_fn = suggestion_generator_node
        bridge_fn = bridge_node

    graph = StateGraph(DiscoveryState)

    graph.add_node("onboarding", onboarding_fn)
    graph.add_node("discovery_chat", discovery_fn)
    graph.add_node("suggestion_generator", suggestion_fn)
    graph.add_node("bridge", bridge_fn)

    graph.set_entry_point("onboarding")

    graph.add_conditional_edges(
        "onboarding",
        _should_continue_onboarding,
        {"onboarding": "onboarding", "discovery_chat": "discovery_chat"},
    )
    graph.add_conditional_edges(
        "discovery_chat",
        _should_continue_discovery,
        {"discovery_chat": "discovery_chat", "suggestion_generator": "suggestion_generator"},
    )
    graph.add_edge("suggestion_generator", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
