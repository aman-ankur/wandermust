from unittest.mock import patch, MagicMock
from agents.hotels import hotels_node, parse_hotel_prices

def test_parse_prices():
    resp = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}, {"offers": [{"price": {"total": "12000.00"}}]}]}
    assert parse_hotel_prices(resp)["avg_nightly"] == 10000.0

def test_parse_empty():
    assert parse_hotel_prices({"data": []}) is None

def test_hotels_node_mock():
    state = {"destination": "Tokyo", "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    mock_client = MagicMock()
    mock_client.get_iata_code.return_value = "TYO"
    mock_client.search_hotels.return_value = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}]}
    with patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.hotels.HistoryDB"):
        result = hotels_node(state)
    assert len(result["hotel_data"]) == 1
