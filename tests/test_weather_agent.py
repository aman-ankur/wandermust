from unittest.mock import patch
from agents.weather import weather_node, score_weather

def test_score_ideal():
    assert score_weather(24.0, 0, 50.0) > 0.8

def test_score_bad():
    assert score_weather(40.0, 6, 90.0) < 0.3

def test_score_range():
    s = score_weather(24.0, 2, 60.0)
    assert 0.0 <= s <= 1.0

def test_weather_node_populates_state():
    state = {"destination": "Tokyo", "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}), \
         patch("agents.weather.get_weather_for_window", return_value={"avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0}):
        result = weather_node(state)
    assert len(result["weather_data"]) == 1
    assert 0.0 <= result["weather_data"][0]["score"] <= 1.0
