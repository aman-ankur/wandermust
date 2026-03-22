from unittest.mock import patch, MagicMock
from agents.hotels import hotels_node, parse_hotel_prices

def test_parse_prices():
    resp = {"properties": [
        {"rate_per_night": {"extracted_lowest": 150}},
        {"rate_per_night": {"extracted_lowest": 200}},
    ]}
    result = parse_hotel_prices(resp)
    assert result["avg_nightly"] == 175.0

def test_parse_empty():
    assert parse_hotel_prices({"properties": []}) is None

def test_parse_missing_rate():
    resp = {"properties": [
        {"rate_per_night": {"extracted_lowest": 100}},
        {"name": "No rate hotel"},  # missing rate_per_night
    ]}
    result = parse_hotel_prices(resp)
    assert result["avg_nightly"] == 100.0

def test_hotels_node_mock():
    state = {"destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    mock_client = MagicMock()
    mock_client.search_hotels.return_value = {
        "properties": [{"rate_per_night": {"extracted_lowest": 8000}}]}
    with patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.hotels.HistoryDB"):
        result = hotels_node(state)
    assert len(result["hotel_data"]) == 1
    assert result["hotel_data"][0]["avg_nightly"] == 8000.0
