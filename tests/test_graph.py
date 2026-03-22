from unittest.mock import patch, MagicMock
from graph import build_graph

def test_graph_builds():
    assert build_graph() is not None

def test_graph_e2e_mocked():
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = {"data": [{"price": {"total": "15000.00"}}]}
    mock_client.search_hotels.return_value = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}]}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July is great for Tokyo.")

    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}), \
         patch("agents.weather.get_weather_for_window", return_value={"avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0}), \
         patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.synthesizer._get_llm", return_value=mock_llm), \
         patch("agents.flights.HistoryDB"), patch("agents.hotels.HistoryDB"), \
         patch("agents.social.search_destination", return_value=[]), \
         patch("agents.social.search_subreddits", return_value=[]), \
         patch("agents.social.HistoryDB") as mock_social_db:
        mock_social_db.return_value.get_social.return_value = None
        result = build_graph().invoke({
            "destination": "Tokyo", "origin": "Bangalore",
            "date_range": ("2026-07-01", "2026-07-28"), "duration_days": 7,
            "num_travelers": 1, "budget_max": None,
            "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15}, "errors": []})
    assert len(result["ranked_windows"]) > 0
    assert len(result["recommendation"]) > 0

def test_demo_graph_includes_social():
    """Demo graph should include social node and produce social data."""
    g = build_graph(demo=True)
    result = g.invoke({
        "destination": "Tokyo",
        "origin": "Bangalore",
        "date_range": ("2026-07-01", "2026-07-28"),
        "duration_days": 7,
        "num_travelers": 1,
        "budget_max": None,
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "errors": [],
    })
    assert "social_data" in result
    assert len(result["social_data"]) > 0
    assert "social_insights" in result
    assert len(result["social_insights"]) > 0
    # Ranked windows should have social_score
    assert "social_score" in result["ranked_windows"][0]
