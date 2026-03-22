import httpx
from datetime import date
from config import settings

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

def get_weather_for_window(latitude, longitude, start_date, end_date):
    """Returns: {avg_temp, rain_days, avg_humidity}. Uses Climate API for future, Historical for past."""
    is_future = date.fromisoformat(end_date) >= date.today()
    url = CLIMATE_URL if is_future else HISTORICAL_URL
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
    }
    if is_future:
        params["models"] = "EC_Earth3P_HR"
    response = httpx.get(url, params=params, timeout=settings.api_timeout_seconds)
    response.raise_for_status()
    daily = response.json().get("daily", {})
    temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    humidity = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
    return {
        "avg_temp": round(sum(temps)/len(temps), 1) if temps else 0.0,
        "rain_days": sum(1 for p in precip if p > 1.0),
        "avg_humidity": round(sum(humidity)/len(humidity), 1) if humidity else 0.0,
    }
