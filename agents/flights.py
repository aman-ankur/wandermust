from models import TravelState
from services.serpapi_client import SerpApiClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = SerpApiClient(api_key=settings.serpapi_api_key)
    return _client

def parse_flight_prices(api_response):
    all_flights = api_response.get("best_flights", []) + api_response.get("other_flights", [])
    if not all_flights: return None
    prices = [f["price"] for f in all_flights if "price" in f]
    if not prices: return None
    return {"min_price": min(prices), "avg_price": round(sum(prices)/len(prices), 2)}

def flights_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    origin = state["origin"]
    destination = state["destination"]

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_flights(origin, destination, window["start"],
                                             currency=currency, adults=state.get("num_travelers", 1))
            parsed = parse_flight_prices(response)
            if parsed:
                db.save_flight(origin, destination, window["start"], parsed["min_price"], currency)
                results.append({"window": window, "min_price": parsed["min_price"],
                    "avg_price": parsed["avg_price"], "currency": currency, "score": 0.0, "is_historical": False})
            else:
                hist = db.get_flight(origin, destination, window["start"], tolerance_days=7)
                if hist:
                    results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                        "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
        except Exception as e:
            hist = db.get_flight(origin, destination, window["start"], tolerance_days=7)
            if hist:
                results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                    "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
            else:
                errors.append(f"Flights: failed for {window['start']} — {e}")
    return {"flight_data": results, "errors": errors}
