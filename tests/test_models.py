from models import CandidateWindow, WeatherResult, FlightResult

def test_candidate_window():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    assert w.start == "2026-06-01"

def test_weather_result_defaults():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    r = WeatherResult(window=w, avg_temp=25.0, rain_days=2, avg_humidity=60.0)
    assert r.score == 0.0
    assert r.is_historical is False

def test_flight_result_currency_default():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    r = FlightResult(window=w, min_price=15000.0, avg_price=18000.0)
    assert r.currency == "INR"
