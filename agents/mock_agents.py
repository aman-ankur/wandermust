"""Mock agent nodes for demo mode.

Drop-in replacements for weather_node, flights_node, hotels_node, and
synthesizer_node that use mock_data instead of live API calls.
"""
from models import TravelState
from agents.weather import score_weather
from mock_data import (
    get_mock_weather,
    get_mock_flight_price,
    get_mock_hotel_price,
    get_mock_recommendation,
    get_mock_social_insights,
)
from config import settings


def mock_weather_node(state: TravelState) -> dict:
    """Mock weather agent — returns destination-aware simulated weather."""
    destination = state["destination"]
    results = []
    for window in state["candidate_windows"]:
        data = get_mock_weather(destination, window["start"], window["end"])
        score = score_weather(data["avg_temp"], data["rain_days"], data["avg_humidity"])
        results.append({
            "window": window,
            "avg_temp": data["avg_temp"],
            "rain_days": data["rain_days"],
            "avg_humidity": data["avg_humidity"],
            "score": score,
            "is_historical": False,
        })
    return {"weather_data": results, "errors": []}


def mock_flights_node(state: TravelState) -> dict:
    """Mock flights agent — returns destination-aware simulated prices."""
    destination = state["destination"]
    currency = settings.default_currency
    results = []
    for window in state["candidate_windows"]:
        prices = get_mock_flight_price(destination, window["start"])
        results.append({
            "window": window,
            "min_price": prices["min_price"],
            "avg_price": prices["avg_price"],
            "currency": currency,
            "score": 0.0,
            "is_historical": False,
        })
    return {"flight_data": results, "errors": []}


def mock_hotels_node(state: TravelState) -> dict:
    """Mock hotels agent — returns destination-aware simulated rates."""
    destination = state["destination"]
    currency = settings.default_currency
    results = []
    for window in state["candidate_windows"]:
        prices = get_mock_hotel_price(destination, window["start"])
        results.append({
            "window": window,
            "avg_nightly": prices["avg_nightly"],
            "currency": currency,
            "score": 0.0,
            "is_historical": False,
        })
    return {"hotel_data": results, "errors": []}


def mock_synthesizer_node(state: TravelState) -> dict:
    """Mock synthesizer — returns a static recommendation without LLM."""
    ranked = state.get("ranked_windows", [])
    destination = state.get("destination", "the destination")
    origin = state.get("origin", "your city")
    recommendation = get_mock_recommendation(destination, origin, ranked)
    return {"recommendation": recommendation, "errors": []}


def mock_social_node(state: TravelState) -> dict:
    """Mock social agent — returns destination-aware simulated insights."""
    from datetime import date as _date
    destination = state["destination"]
    windows = state.get("candidate_windows", [])
    if not windows:
        return {"social_data": [], "social_insights": [], "errors": []}
    insights_data = get_mock_social_insights(destination, windows[0]["start"])
    social_data = []
    for w in windows:
        month = _date.fromisoformat(w["start"]).month
        score = insights_data["timing_score"]
        best = insights_data.get("best_months", [])
        if best and month in best:
            score = min(1.0, score + 0.1)
        elif best:
            score = max(0.0, score - 0.1)
        social_data.append({
            "window_start": w["start"],
            "window_end": w["end"],
            "social_score": round(score, 3),
        })
    social_insights = [{
        "destination": destination,
        "timing_score": insights_data["timing_score"],
        "crowd_level": insights_data["crowd_level"],
        "events": insights_data["events"],
        "itinerary_tips": insights_data["itinerary_tips"],
        "sentiment": insights_data["sentiment"],
        "sources": [],
    }]
    return {"social_data": social_data, "social_insights": social_insights, "errors": []}
