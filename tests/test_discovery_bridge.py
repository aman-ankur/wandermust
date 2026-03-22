import pytest
from unittest.mock import patch
from agents.discovery_bridge import (
    bridge_node,
    build_optimizer_state,
    _resolve_date_range,
)


def test_build_optimizer_state_basic():
    intent = {
        "travel_month": "July",
        "duration_days": 7,
        "travel_companions": "couple",
        "budget_total": 150000,
    }
    state = build_optimizer_state("Bali, Indonesia", intent)

    assert state["destination"] == "Bali, Indonesia"
    assert state["duration_days"] == 7
    assert state["num_travelers"] == 2
    assert state["budget_max"] == 150000.0
    assert "date_range" in state
    assert len(state["date_range"]) == 2


def test_build_optimizer_state_solo():
    intent = {
        "travel_month": "December",
        "duration_days": 10,
        "travel_companions": "solo",
    }
    state = build_optimizer_state("Tokyo, Japan", intent)

    assert state["num_travelers"] == 1
    assert state["budget_max"] is None


def test_build_optimizer_state_default_origin():
    intent = {"travel_month": "July", "duration_days": 7}
    state = build_optimizer_state("Bali", intent)
    assert state["origin"] == "Bangalore"


def test_build_optimizer_state_custom_origin():
    intent = {"travel_month": "July", "duration_days": 7}
    state = build_optimizer_state("Bali", intent, origin="Delhi")
    assert state["origin"] == "Delhi"


def test_build_optimizer_state_priorities():
    intent = {"travel_month": "July", "duration_days": 7}
    state = build_optimizer_state("Bali", intent)
    assert "weather" in state["priorities"]
    assert "flights" in state["priorities"]
    assert "hotels" in state["priorities"]
    assert "social" in state["priorities"]


def test_resolve_date_range_specific_month():
    start, end = _resolve_date_range("July", 7)
    assert "-07-01" in start
    assert "-07-31" in end


def test_resolve_date_range_season():
    start, end = _resolve_date_range("summer", 7)
    assert "-06-01" in start
    assert "-08-31" in end


def test_resolve_date_range_unknown():
    start, end = _resolve_date_range("sometime", 7)
    # Should return a fallback ~3 months out
    assert start is not None
    assert end is not None


def test_bridge_node_success():
    state = {
        "chosen_destination": "Bali, Indonesia",
        "trip_intent": {
            "travel_month": "July",
            "duration_days": 7,
            "travel_companions": "couple",
        },
        "errors": [],
    }
    result = bridge_node(state)
    assert result["optimizer_state"] is not None
    assert result["optimizer_state"]["destination"] == "Bali, Indonesia"
    assert result["optimizer_state"]["num_travelers"] == 2


def test_bridge_node_no_destination():
    state = {
        "chosen_destination": None,
        "trip_intent": {},
        "errors": [],
    }
    result = bridge_node(state)
    assert result["optimizer_state"] is None
    assert any("no destination" in e.lower() for e in result["errors"])
