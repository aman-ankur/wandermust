from unittest.mock import patch, MagicMock
from agents.flights import flights_node, parse_flight_prices

def test_parse_prices():
    resp = {"best_flights": [{"price": 267}, {"price": 324}], "other_flights": [{"price": 278}]}
    result = parse_flight_prices(resp)
    assert result["min_price"] == 267
    assert result["avg_price"] == 289.67

def test_parse_empty():
    assert parse_flight_prices({"best_flights": [], "other_flights": []}) is None

def test_flights_node_mock():
    state = {"origin": "Bangalore", "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}],
        "num_travelers": 1, "errors": []}
    mock_client = MagicMock()
    mock_client.search_flights.return_value = {
        "best_flights": [{"price": 15000}], "other_flights": []}
    with patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.flights.HistoryDB"):
        result = flights_node(state)
    assert len(result["flight_data"]) == 1
    assert result["flight_data"][0]["min_price"] == 15000
