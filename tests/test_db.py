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
