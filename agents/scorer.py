from models import TravelState

def normalize_scores(values, lower_is_better=False):
    if len(values) <= 1: return [1.0] * len(values)
    min_v, max_v = min(values), max(values)
    if max_v == min_v: return [1.0] * len(values)
    if lower_is_better:
        return [round((max_v - v) / (max_v - min_v), 3) for v in values]
    return [round((v - min_v) / (max_v - min_v), 3) for v in values]

def scorer_node(state: TravelState) -> dict:
    priorities = state.get("priorities", {"weather": 0.4, "flights": 0.3, "hotels": 0.3})
    weather_data = state.get("weather_data", [])
    flight_data = state.get("flight_data", [])
    hotel_data = state.get("hotel_data", [])
    social_data = state.get("social_data", [])

    weather_by_start = {d["window"]["start"]: d for d in weather_data}
    flight_by_start = {d["window"]["start"]: d for d in flight_data}
    hotel_by_start = {d["window"]["start"]: d for d in hotel_data}
    social_by_start = {d["window_start"]: d for d in social_data}

    all_starts = sorted(set(
        [d["window"]["start"] for d in weather_data] +
        [d["window"]["start"] for d in flight_data] +
        [d["window"]["start"] for d in hotel_data] +
        [d["window_start"] for d in social_data]))

    # Normalize prices (lower = better)
    flight_starts = [s for s in all_starts if s in flight_by_start]
    hotel_starts = [s for s in all_starts if s in hotel_by_start]
    flight_norm = dict(zip(flight_starts,
        normalize_scores([flight_by_start[s]["min_price"] for s in flight_starts], lower_is_better=True))) if flight_starts else {}
    hotel_norm = dict(zip(hotel_starts,
        normalize_scores([hotel_by_start[s]["avg_nightly"] for s in hotel_starts], lower_is_better=True))) if hotel_starts else {}

    # Reweight if dimension missing
    active = {}
    if weather_data: active["weather"] = priorities.get("weather", 0.35)
    if flight_data: active["flights"] = priorities.get("flights", 0.25)
    if hotel_data: active["hotels"] = priorities.get("hotels", 0.25)
    if social_data: active["social"] = priorities.get("social", 0.15)
    total_w = sum(active.values()) or 1.0
    norm_w = {k: v/total_w for k, v in active.items()}

    ranked = []
    for start in all_starts:
        ws = weather_by_start.get(start, {}).get("score", 0.0)
        fs = flight_norm.get(start, 0.0)
        hs = hotel_norm.get(start, 0.0)
        ss = social_by_start.get(start, {}).get("social_score", 0.0)
        total = (ws * norm_w.get("weather", 0) + fs * norm_w.get("flights", 0) +
                 hs * norm_w.get("hotels", 0) + ss * norm_w.get("social", 0))
        window = (weather_by_start.get(start) or flight_by_start.get(start) or hotel_by_start.get(start, {}))
        ranked.append({
            "window": window.get("window", {"start": start, "end": ""}),
            "weather_score": round(ws, 3), "flight_score": round(fs, 3),
            "hotel_score": round(hs, 3), "social_score": round(ss, 3),
            "total_score": round(total, 3),
            "estimated_flight_cost": flight_by_start.get(start, {}).get("min_price", 0.0),
            "estimated_hotel_cost": hotel_by_start.get(start, {}).get("avg_nightly", 0.0),
            "has_historical_data": any([
                weather_by_start.get(start, {}).get("is_historical", False),
                flight_by_start.get(start, {}).get("is_historical", False),
                hotel_by_start.get(start, {}).get("is_historical", False)]),
        })
    ranked.sort(key=lambda x: x["total_score"], reverse=True)
    return {"ranked_windows": ranked, "errors": state.get("errors", [])}
