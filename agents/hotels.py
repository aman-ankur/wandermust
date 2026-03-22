from models import TravelState
from services.amadeus_client import AmadeusClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = AmadeusClient()
    return _client

def parse_hotel_prices(api_response):
    hotels = api_response.get("data", [])
    if not hotels: return None
    prices = []
    for h in hotels:
        offers = h.get("offers", [])
        if offers: prices.append(float(offers[0]["price"]["total"]))
    if not prices: return None
    return {"avg_nightly": round(sum(prices)/len(prices), 2)}

def hotels_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    destination = state["destination"]
    try:
        city_code = client.get_iata_code(destination)
        if not city_code: raise ValueError(f"IATA not found for {destination}")
    except Exception as e:
        errors.append(f"Hotels: IATA lookup failed — {e}")
        return {"hotel_data": [], "errors": errors}

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_hotels(city_code, window["start"], window["end"], currency=currency)
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
