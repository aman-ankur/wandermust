"""Tests for mock agents and mock data generator."""
from agents.mock_agents import (
    mock_weather_node, mock_flights_node,
    mock_hotels_node, mock_synthesizer_node,
)
from mock_data import get_mock_weather, get_mock_flight_price, get_mock_hotel_price
from graph import build_graph


# --- Mock data unit tests ---

def test_mock_weather_known_destination():
    """Tokyo in July should have warm temps."""
    data = get_mock_weather("Tokyo", "2026-07-01", "2026-07-07")
    assert "avg_temp" in data
    assert "rain_days" in data
    assert "avg_humidity" in data
    assert 20.0 < data["avg_temp"] < 35.0  # summer in Tokyo


def test_mock_weather_unknown_destination():
    """Unknown city still returns valid data."""
    data = get_mock_weather("Xyzzyville", "2026-07-01", "2026-07-07")
    assert isinstance(data["avg_temp"], float)
    assert isinstance(data["rain_days"], int)
    assert data["rain_days"] >= 0


def test_mock_weather_deterministic():
    """Same inputs produce same outputs."""
    d1 = get_mock_weather("Tokyo", "2026-07-01", "2026-07-07")
    d2 = get_mock_weather("Tokyo", "2026-07-01", "2026-07-07")
    assert d1 == d2


def test_mock_flight_price_known():
    prices = get_mock_flight_price("Tokyo", "2026-07-01")
    assert 20000 <= prices["min_price"] <= 50000
    assert prices["avg_price"] >= prices["min_price"]


def test_mock_hotel_price_known():
    prices = get_mock_hotel_price("Tokyo", "2026-07-01")
    assert 5000 <= prices["avg_nightly"] <= 15000


def test_mock_flight_price_unknown():
    prices = get_mock_flight_price("Xyzzyville", "2026-07-01")
    assert prices["min_price"] > 0
    assert prices["avg_price"] >= prices["min_price"]


# --- Mock agent node tests ---

def _make_state():
    return {
        "destination": "Tokyo",
        "origin": "Bangalore",
        "candidate_windows": [
            {"start": "2026-07-01", "end": "2026-07-07"},
            {"start": "2026-07-08", "end": "2026-07-14"},
        ],
        "num_travelers": 1,
        "errors": [],
    }


def test_mock_weather_node_shape():
    result = mock_weather_node(_make_state())
    assert "weather_data" in result
    assert len(result["weather_data"]) == 2
    for d in result["weather_data"]:
        assert "window" in d
        assert "score" in d
        assert 0.0 <= d["score"] <= 1.0


def test_mock_flights_node_shape():
    result = mock_flights_node(_make_state())
    assert "flight_data" in result
    assert len(result["flight_data"]) == 2
    for d in result["flight_data"]:
        assert "min_price" in d
        assert d["min_price"] > 0


def test_mock_hotels_node_shape():
    result = mock_hotels_node(_make_state())
    assert "hotel_data" in result
    assert len(result["hotel_data"]) == 2
    for d in result["hotel_data"]:
        assert "avg_nightly" in d
        assert d["avg_nightly"] > 0


def test_mock_synthesizer_node():
    state = {
        "destination": "Tokyo",
        "origin": "Bangalore",
        "ranked_windows": [
            {
                "window": {"start": "2026-07-01", "end": "2026-07-07"},
                "total_score": 0.85, "weather_score": 0.9,
                "flight_score": 0.8, "hotel_score": 0.75,
                "estimated_flight_cost": 30000,
                "estimated_hotel_cost": 8000,
                "has_historical_data": False,
            }
        ],
        "errors": [],
    }
    result = mock_synthesizer_node(state)
    assert "recommendation" in result
    assert "2026-07-01" in result["recommendation"]
    assert "demo" in result["recommendation"].lower()


# --- Full graph integration with demo mode ---

def test_demo_graph_full_run():
    """Run the full graph in demo mode — no API calls needed."""
    g = build_graph(demo=True)
    result = g.invoke({
        "destination": "Tokyo",
        "origin": "Bangalore",
        "date_range": ("2026-07-01", "2026-07-28"),
        "duration_days": 7,
        "num_travelers": 1,
        "budget_max": None,
        "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
        "errors": [],
    })
    assert "ranked_windows" in result
    assert len(result["ranked_windows"]) > 0
    assert "recommendation" in result
    assert len(result["recommendation"]) > 0
    # Should have no errors in demo mode
    assert len(result.get("errors", [])) == 0
