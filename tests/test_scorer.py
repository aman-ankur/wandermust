from agents.scorer import scorer_node, normalize_scores

def test_normalize_lower_is_better():
    result = normalize_scores([100, 200, 300], lower_is_better=True)
    assert result[0] == 1.0 and result[2] == 0.0

def test_normalize_higher_is_better():
    result = normalize_scores([0.2, 0.5, 0.9], lower_is_better=False)
    assert result[2] == 1.0

def test_normalize_single():
    assert normalize_scores([42], True) == [1.0]

def test_scorer_ranks():
    state = {
        "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
        "weather_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.9},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "score": 0.5}],
        "flight_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "min_price": 25000}],
        "hotel_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "avg_nightly": 8000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "avg_nightly": 6000}],
        "errors": []}
    result = scorer_node(state)
    assert result["ranked_windows"][0]["total_score"] >= result["ranked_windows"][1]["total_score"]

def test_scorer_missing_dimension():
    state = {"priorities": {"weather": 0.5, "flights": 0.3, "hotels": 0.2},
        "weather_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.8}],
        "flight_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000}],
        "hotel_data": [], "errors": []}
    result = scorer_node(state)
    assert len(result["ranked_windows"]) == 1
