import pytest
from services.geocoding import geocode_city

def test_geocode_city_returns_lat_lon():
    result = geocode_city("Tokyo")
    assert "latitude" in result
    assert abs(result["latitude"] - 35.68) < 1.0

def test_geocode_city_not_found():
    with pytest.raises(ValueError, match="not found"):
        geocode_city("Xyzzyville123")

def test_geocode_returns_name():
    result = geocode_city("Bangalore")
    assert "name" in result
