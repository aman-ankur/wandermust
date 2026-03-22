from models import TravelState
from services.geocoding import geocode_city
from services.weather_client import get_weather_for_window

def score_weather(avg_temp, rain_days, avg_humidity,
                  ideal_temp_min=20.0, ideal_temp_max=28.0):
    """Score 0-1. Higher = better weather."""
    # Temperature
    if ideal_temp_min <= avg_temp <= ideal_temp_max:
        temp_score = 1.0
    else:
        distance = min(abs(avg_temp - ideal_temp_min), abs(avg_temp - ideal_temp_max))
        temp_score = max(0.0, 1.0 - distance / 20.0)
    # Rain: 0 days = 1.0, 7 days = 0.0
    rain_score = max(0.0, 1.0 - rain_days / 7.0)
    # Humidity
    if avg_humidity <= 60:
        humidity_score = 1.0
    elif avg_humidity <= 80:
        humidity_score = 1.0 - (avg_humidity - 60) / 40.0
    else:
        humidity_score = max(0.0, 0.5 - (avg_humidity - 80) / 40.0)
    return round(temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25, 3)

def weather_node(state: TravelState) -> dict:
    errors = list(state.get("errors", []))
    try:
        geo = geocode_city(state["destination"])
    except ValueError as e:
        errors.append(f"Weather: geocoding failed — {e}")
        return {"weather_data": [], "errors": errors}
    results = []
    for window in state["candidate_windows"]:
        try:
            data = get_weather_for_window(geo["latitude"], geo["longitude"],
                                          window["start"], window["end"])
            score = score_weather(data["avg_temp"], data["rain_days"], data["avg_humidity"])
            results.append({
                "window": window, "avg_temp": data["avg_temp"],
                "rain_days": data["rain_days"], "avg_humidity": data["avg_humidity"],
                "score": score, "is_historical": False,
            })
        except Exception as e:
            errors.append(f"Weather: failed for {window['start']} — {e}")
    return {"weather_data": results, "errors": errors}
