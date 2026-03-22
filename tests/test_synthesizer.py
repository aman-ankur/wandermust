from unittest.mock import patch, MagicMock
from agents.synthesizer import synthesizer_node, format_ranked_data_fallback

SAMPLE_RANKED = [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "total_score": 0.85,
    "weather_score": 0.9, "flight_score": 0.8, "hotel_score": 0.75,
    "estimated_flight_cost": 15000, "estimated_hotel_cost": 8000, "has_historical_data": False}]

def test_fallback_format():
    result = format_ranked_data_fallback(SAMPLE_RANKED)
    assert "2026-07-01" in result

def test_synthesizer_uses_llm():
    state = {"destination": "Tokyo", "origin": "Bangalore", "ranked_windows": SAMPLE_RANKED, "errors": []}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July is great.")
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    assert "July" in result["recommendation"]

def test_synthesizer_fallback():
    state = {"destination": "Tokyo", "origin": "Bangalore", "ranked_windows": SAMPLE_RANKED, "errors": []}
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    assert "2026-07-01" in result["recommendation"]

def test_synthesizer_includes_social_insights():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Tokyo is best in spring. Crowds are moderate. Visit Senso-ji early.")
    state = {
        "destination": "Tokyo", "origin": "Bangalore",
        "ranked_windows": [{
            "window": {"start": "2026-04-01", "end": "2026-04-07"},
            "total_score": 0.9, "weather_score": 0.85, "flight_score": 0.8,
            "hotel_score": 0.7, "social_score": 0.9,
            "estimated_flight_cost": 30000, "estimated_hotel_cost": 8000,
        }],
        "social_insights": [{
            "destination": "Tokyo",
            "crowd_level": "moderate",
            "events": [{"name": "Cherry Blossom Festival", "period": "late March - mid April"}],
            "itinerary_tips": [{"tip": "Visit Senso-ji early morning", "source": "reddit"}],
            "sentiment": "highly recommended",
        }],
        "errors": [],
    }
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    # Verify the prompt sent to LLM includes social insights
    call_args = mock_llm.invoke.call_args[0][0]
    assert "crowd" in call_args.lower() or "social" in call_args.lower()
    assert "Cherry Blossom" in call_args or "itinerary" in call_args.lower()
