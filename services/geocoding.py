import logging
import httpx
from config import settings

logger = logging.getLogger("wandermust.geocoding")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

def geocode_city(city_name: str) -> dict:
    """Returns: {name, latitude, longitude, country, country_code}. Raises ValueError if not found."""
    response = httpx.get(
        GEOCODING_URL,
        params={"name": city_name, "count": 1, "language": "en"},
        timeout=settings.api_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("results"):
        logger.error(f"Geocoding failed: city '{city_name}' not found")
        raise ValueError(f"City '{city_name}' not found")
    result = data["results"][0]
    logger.info(f"Geocoded '{city_name}' → {result['name']} ({result.get('country', '')}) lat={result['latitude']:.2f} lon={result['longitude']:.2f}")
    return {
        "name": result["name"],
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "country": result.get("country", ""),
        "country_code": result.get("country_code", ""),
    }
