from models import CandidateWindow, WeatherResult, FlightResult, TravelState, RankedWindow

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

def test_travel_state_accepts_social_fields():
    state: TravelState = {
        "destination": "Tokyo",
        "social_data": [{"window_start": "2026-07-01", "social_score": 0.8}],
        "social_insights": [{"destination": "Tokyo", "timing_score": 0.8, "crowd_level": "moderate"}],
        "errors": [],
    }
    assert state["social_data"][0]["social_score"] == 0.8
    assert state["social_insights"][0]["crowd_level"] == "moderate"

def test_ranked_window_has_social_score():
    rw = RankedWindow(
        window={"start": "2026-07-01", "end": "2026-07-07"},
        weather_score=0.8, flight_score=0.7, hotel_score=0.6,
        social_score=0.85, total_score=0.75,
    )
    assert rw.social_score == 0.85
