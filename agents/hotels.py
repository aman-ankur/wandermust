from models import TravelState
from services.serpapi_client import SerpApiClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = SerpApiClient(api_key=settings.serpapi_api_key)
    return _client

def parse_hotel_prices(api_response):
    properties = api_response.get("properties", [])
    if not properties: return None
    prices = []
    for p in properties:
        try:
            prices.append(float(p["rate_per_night"]["extracted_lowest"]))
        except (KeyError, TypeError):
            continue
    if not prices: return None
    return {"avg_nightly": round(sum(prices)/len(prices), 2)}

def hotels_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    destination = state["destination"]

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_hotels(destination, window["start"], window["end"], currency=currency)
            parsed = parse_hotel_prices(response)
            if parsed:
                db.save_hotel(destination, window["start"], parsed["avg_nightly"], currency)
                results.append({"window": window, "avg_nightly": parsed["avg_nightly"],
                    "currency": currency, "score": 0.0, "is_historical": False})
            else:
                hist = db.get_hotel(destination, window["start"], tolerance_days=7)
                if hist:
                    results.append({"window": window, "avg_nightly": hist["avg_nightly"],
                        "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
        except Exception as e:
            hist = db.get_hotel(destination, window["start"], tolerance_days=7)
            if hist:
                results.append({"window": window, "avg_nightly": hist["avg_nightly"],
                    "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
            else:
                errors.append(f"Hotels: failed for {window['start']} — {e}")
    return {"hotel_data": results, "errors": errors}
