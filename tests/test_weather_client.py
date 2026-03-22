import pytest
from services.weather_client import get_weather_for_window

def test_weather_returns_expected_fields():
    result = get_weather_for_window(35.68, 139.69, "2026-07-01", "2026-07-07")
    assert "avg_temp" in result
    assert "rain_days" in result
    assert isinstance(result["avg_temp"], float)

def test_weather_invalid_coords():
    with pytest.raises(Exception):
        get_weather_for_window(999.0, 999.0, "2026-07-01", "2026-07-07")
