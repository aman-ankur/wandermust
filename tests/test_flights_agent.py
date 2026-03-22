from unittest.mock import patch, MagicMock
from agents.flights import flights_node, parse_flight_prices

def test_parse_prices():
    resp = {"data": [{"price": {"total": "15000.00"}}, {"price": {"total": "18000.00"}}, {"price": {"total": "20000.00"}}]}
    result = parse_flight_prices(resp)
    assert result["min_price"] == 15000.0
    assert result["avg_price"] == 17666.67

def test_parse_empty():
    assert parse_flight_prices({"data": []}) is None

def test_flights_node_mock():
    state = {"origin": "Bangalore", "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "num_travelers": 1, "errors": []}
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = {"data": [{"price": {"total": "15000.00"}}]}
    with patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.flights.HistoryDB"):
        result = flights_node(state)
    assert len(result["flight_data"]) == 1
