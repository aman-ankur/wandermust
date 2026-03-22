from serpapi import GoogleSearch

class SerpApiClient:
    def __init__(self, api_key=""):
        self._api_key = api_key

    def search_flights(self, origin, destination, departure_date, currency="INR", adults=1):
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
        return {
            "best_flights": results.get("best_flights", []),
            "other_flights": results.get("other_flights", []),
        }

    def search_hotels(self, city_name, checkin, checkout, currency="INR", adults=1):
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
        return {
            "properties": results.get("properties", []),
        }
