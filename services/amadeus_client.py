import time
import httpx
from config import settings

AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
FLIGHTS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
HOTEL_LIST_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
HOTEL_OFFERS_URL = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
IATA_URL = "https://test.api.amadeus.com/v1/reference-data/locations"

class AmadeusClient:
    def __init__(self, client_id="", client_secret=""):
        self._client_id = client_id or settings.amadeus_client_id
        self._client_secret = client_secret or settings.amadeus_client_secret
        self._token = ""
        self._token_expiry = 0.0

    def _authenticate(self):
        response = httpx.post(AUTH_URL, data={
            "grant_type": "client_credentials",
            "client_id": self._client_id, "client_secret": self._client_secret,
        }, timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"] - 60

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _headers(self):
        return {"Authorization": f"Bearer {self._token}"}

    def search_flights(self, origin, destination, departure_date, currency="INR", adults=1):
        self._ensure_auth()
        response = httpx.get(FLIGHTS_URL, params={
            "originLocationCode": origin, "destinationLocationCode": destination,
            "departureDate": departure_date, "adults": adults,
            "currencyCode": currency, "max": 5,
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def search_hotels(self, city_code, checkin, checkout, currency="INR", adults=1):
        self._ensure_auth()
        response = httpx.get(HOTEL_LIST_URL, params={"cityCode": city_code},
            headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        hotels = response.json().get("data", [])[:10]
        if not hotels:
            return {"data": []}
        hotel_ids = [h["hotelId"] for h in hotels]
        response = httpx.get(HOTEL_OFFERS_URL, params={
            "hotelIds": ",".join(hotel_ids), "checkInDate": checkin,
            "checkOutDate": checkout, "adults": adults,
            "currency": currency, "bestRateOnly": "true",
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_iata_code(self, city_name):
        self._ensure_auth()
        response = httpx.get(IATA_URL, params={
            "keyword": city_name, "subType": "CITY,AIRPORT",
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data[0].get("iataCode") if data else None
