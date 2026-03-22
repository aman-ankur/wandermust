from unittest.mock import patch, MagicMock
from services.tavily_client import search_destination

def test_search_destination_returns_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "Best time to visit Tokyo", "content": "Cherry blossom season is March-April.",
             "url": "https://example.com/tokyo", "score": 0.9},
            {"title": "Tokyo travel tips", "content": "Avoid Golden Week in May.",
             "url": "https://example.com/tokyo2", "score": 0.8},
        ]
    }
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Tokyo", "July")
    assert len(results) == 4  # 2 results x 2 queries
    assert results[0]["title"] == "Best time to visit Tokyo"
    assert "content" in results[0]
    assert "url" in results[0]

def test_search_destination_empty():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Xyzzyville", "January")
    assert results == []

def test_search_destination_api_error():
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("API error")
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Tokyo", "July")
    assert results == []
