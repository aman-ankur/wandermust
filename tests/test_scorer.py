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

def test_scorer_with_social_dimension():
    state = {
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "weather_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.9},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "score": 0.5}],
        "flight_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "min_price": 25000}],
        "hotel_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "avg_nightly": 8000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "avg_nightly": 6000}],
        "social_data": [
            {"window_start": "2026-07-01", "window_end": "2026-07-07", "social_score": 0.85},
            {"window_start": "2026-07-08", "window_end": "2026-07-14", "social_score": 0.6}],
        "errors": []}
    result = scorer_node(state)
    assert result["ranked_windows"][0]["total_score"] >= result["ranked_windows"][1]["total_score"]
    # social_score should be present in ranked output
    assert "social_score" in result["ranked_windows"][0]
    assert result["ranked_windows"][0]["social_score"] >= 0

def test_scorer_social_missing_reweights():
    """Without social_data, scorer should redistribute weights."""
    state = {
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "weather_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.8}],
        "flight_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000}],
        "hotel_data": [],
        "social_data": [],
        "errors": []}
    result = scorer_node(state)
    assert len(result["ranked_windows"]) == 1
