import pytest
from db import HistoryDB

@pytest.fixture
def test_db(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    yield db
    db.close()

def test_save_and_query_flight(test_db):
    test_db.save_flight("BLR", "NRT", "2026-06-01", 15000.0, "INR")
    result = test_db.get_flight("BLR", "NRT", "2026-06-01")
    assert result is not None
    assert result["price"] == 15000.0

def test_query_flight_not_found(test_db):
    assert test_db.get_flight("BLR", "NRT", "2026-06-01") is None

def test_save_and_query_hotel(test_db):
    test_db.save_hotel("Tokyo", "2026-06-01", 8000.0, "INR")
    result = test_db.get_hotel("Tokyo", "2026-06-01")
    assert result["avg_nightly"] == 8000.0

def test_query_similar_date_flight(test_db):
    test_db.save_flight("BLR", "NRT", "2026-06-03", 15000.0, "INR")
    result = test_db.get_flight("BLR", "NRT", "2026-06-01", tolerance_days=7)
    assert result["price"] == 15000.0

def test_save_and_get_social(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    db.save_social(
        destination="Tokyo", month=7, timing_score=0.85,
        crowd_level="moderate", events='[{"name":"Tanabata","period":"July 7"}]',
        itinerary_tips='[{"tip":"Visit Meiji Shrine early","source":"reddit"}]',
        sentiment="highly recommended", source="both",
    )
    result = db.get_social("Tokyo", 7)
    assert result is not None
    assert result["timing_score"] == 0.85
    assert result["crowd_level"] == "moderate"

def test_get_social_tolerance(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    db.save_social("Tokyo", 7, 0.85, "moderate", "[]", "[]", "good", "both")
    result = db.get_social("Tokyo", 8, tolerance_months=1)
    assert result is not None

def test_get_social_missing(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    result = db.get_social("Tokyo", 7)
    assert result is None
