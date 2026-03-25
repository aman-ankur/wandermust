import logging
from serpapi import GoogleSearch

logger = logging.getLogger("wandermust.serpapi")

class SerpApiClient:
    def __init__(self, api_key=""):
        self._api_key = api_key

    def search_flights(self, origin, destination, departure_date, currency="INR", adults=1):
        logger.info(f"SerpApi flights: {origin} → {destination}, date={departure_date}, currency={currency}, adults={adults}")
        search = GoogleSearch({
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date,
            "currency": currency,
            "adults": adults,
            "type": "2",  # one-way
            "api_key": self._api_key,
        })
        results = search.get_dict()
        n_best = len(results.get("best_flights", []))
        n_other = len(results.get("other_flights", []))
        logger.info(f"SerpApi flights response: {n_best} best, {n_other} other flights")
        if not n_best and not n_other:
            logger.warning(f"SerpApi flights: no results for {origin} → {destination} on {departure_date}")
        return {
            "best_flights": results.get("best_flights", []),
            "other_flights": results.get("other_flights", []),
        }

    def search_hotels(self, city_name, checkin, checkout, currency="INR", adults=1):
        logger.info(f"SerpApi hotels: {city_name}, checkin={checkin}, checkout={checkout}")
        search = GoogleSearch({
            "engine": "google_hotels",
            "q": f"Hotels in {city_name}",
            "check_in_date": checkin,
            "check_out_date": checkout,
            "currency": currency,
            "adults": adults,
            "api_key": self._api_key,
        })
        results = search.get_dict()
        n_props = len(results.get("properties", []))
        logger.info(f"SerpApi hotels response: {n_props} properties")
        if not n_props:
            logger.warning(f"SerpApi hotels: no properties for {city_name} on {checkin}")
        return {
            "properties": results.get("properties", []),
        }
