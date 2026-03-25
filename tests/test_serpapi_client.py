from unittest.mock import patch, MagicMock
from services.serpapi_client import SerpApiClient

def test_search_flights():
    mock_search = MagicMock()
    mock_search.get_dict.return_value = {
        "best_flights": [{"price": 267}, {"price": 324}],
        "other_flights": [{"price": 278}],
    }
    with patch("services.serpapi_client.GoogleSearch", return_value=mock_search) as MockGS:
        client = SerpApiClient(api_key="test")
        result = client.search_flights("BLR", "DEL", "2026-07-01", currency="USD", adults=2)
    assert result == {"best_flights": [{"price": 267}, {"price": 324}], "other_flights": [{"price": 278}]}
    call_args = MockGS.call_args[0][0]
    assert call_args["engine"] == "google_flights"
    assert call_args["departure_id"] == "BLR"
    assert call_args["adults"] == 2

def test_search_flights_empty():
    mock_search = MagicMock()
    mock_search.get_dict.return_value = {"best_flights": [], "other_flights": []}
    with patch("services.serpapi_client.GoogleSearch", return_value=mock_search):
        client = SerpApiClient(api_key="test")
        result = client.search_flights("BLR", "DEL", "2026-07-01")
    assert result == {"best_flights": [], "other_flights": []}

def test_search_hotels():
    mock_search = MagicMock()
    mock_search.get_dict.return_value = {
        "properties": [
            {"name": "Hotel A", "rate_per_night": {"extracted_lowest": 150}},
            {"name": "Hotel B", "rate_per_night": {"extracted_lowest": 200}},
        ]
    }
    with patch("services.serpapi_client.GoogleSearch", return_value=mock_search) as MockGS:
        client = SerpApiClient(api_key="test")
        result = client.search_hotels("Tokyo", "2026-07-01", "2026-07-07", currency="USD")
    assert len(result["properties"]) == 2
    call_args = MockGS.call_args[0][0]
    assert call_args["engine"] == "google_hotels"
    assert call_args["q"] == "Hotels in Tokyo"

def test_search_hotels_empty():
    mock_search = MagicMock()
    mock_search.get_dict.return_value = {"properties": []}
    with patch("services.serpapi_client.GoogleSearch", return_value=mock_search):
        client = SerpApiClient(api_key="test")
        result = client.search_hotels("Tokyo", "2026-07-01", "2026-07-07")
    assert result == {"properties": []}
