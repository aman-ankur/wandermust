from models import TravelState
from services.amadeus_client import AmadeusClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = AmadeusClient()
    return _client

def parse_flight_prices(api_response):
    offers = api_response.get("data", [])
    if not offers: return None
    prices = [float(o["price"]["total"]) for o in offers]
    return {"min_price": min(prices), "avg_price": round(sum(prices)/len(prices), 2)}

def flights_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    try:
        origin_iata = client.get_iata_code(state["origin"])
        dest_iata = client.get_iata_code(state["destination"])
        if not origin_iata or not dest_iata:
            raise ValueError(f"IATA not found: origin={origin_iata}, dest={dest_iata}")
    except Exception as e:
        errors.append(f"Flights: IATA lookup failed — {e}")
        return {"flight_data": [], "errors": errors}

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_flights(origin_iata, dest_iata, window["start"],
                                             currency=currency, adults=state.get("num_travelers", 1))
            parsed = parse_flight_prices(response)
            if parsed:
                db.save_flight(origin_iata, dest_iata, window["start"], parsed["min_price"], currency)
                results.append({"window": window, "min_price": parsed["min_price"],
                    "avg_price": parsed["avg_price"], "currency": currency, "score": 0.0, "is_historical": False})
            else:
                hist = db.get_flight(origin_iata, dest_iata, window["start"], tolerance_days=7)
                if hist:
                    results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                        "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
        except Exception as e:
            hist = db.get_flight(origin_iata, dest_iata, window["start"], tolerance_days=7)
            if hist:
                results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                    "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
            else:
                errors.append(f"Flights: failed for {window['start']} — {e}")
    return {"flight_data": results, "errors": errors}
