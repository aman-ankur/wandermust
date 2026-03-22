# Travel Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a LangGraph multi-agent system with Streamlit UI that finds optimal travel windows based on weather, flight prices, and hotel costs.

**Architecture:** A supervisor agent generates candidate date windows, dispatches 3 data-fetching agents in parallel (weather, flights, hotels), a scorer ranks windows by weighted score, and a synthesizer generates natural language recommendations. SQLite stores historical data as fallback.

**Tech Stack:** Python 3.11+, LangGraph, Streamlit, Amadeus API, Open-Meteo API, SQLite, Pydantic, OpenRouter (LLM)

**Spec:** `docs/superpowers/specs/2026-03-22-travel-optimizer-design.md`

---

## File Structure

```
travel-optimizer/
├── app.py                        # Streamlit entry point
├── graph.py                      # LangGraph graph definition
├── agents/
│   ├── __init__.py
│   ├── supervisor.py             # Input parsing + window generation
│   ├── weather.py                # Open-Meteo data fetcher
│   ├── flights.py                # Amadeus flights fetcher
│   ├── hotels.py                 # Amadeus hotels fetcher
│   ├── scorer.py                 # Normalization + weighted ranking
│   └── synthesizer.py            # LLM recommendation writer
├── services/
│   ├── __init__.py
│   ├── amadeus_client.py         # Amadeus API wrapper + token management
│   ├── weather_client.py         # Open-Meteo API wrapper
│   └── geocoding.py              # City → lat/lon + IATA code
├── cache.py                      # In-memory TTL cache
├── db.py                         # SQLite historical store + fallback queries
├── models.py                     # Pydantic models (TravelState, API responses)
├── config.py                     # Settings from .env
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_geocoding.py
│   ├── test_weather_client.py
│   ├── test_amadeus_client.py
│   ├── test_db.py
│   ├── test_cache.py
│   ├── test_supervisor.py
│   ├── test_weather_agent.py
│   ├── test_flights_agent.py
│   ├── test_hotels_agent.py
│   ├── test_scorer.py
│   ├── test_synthesizer.py
│   └── test_graph.py
├── .env.example                  # Template for API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `config.py`, `models.py`, `tests/__init__.py`, `agents/__init__.py`, `services/__init__.py`

- [x] **Step 1: Initialize git repo and create .gitignore**

```bash
cd /Users/aankur/workspace/travel-optimizer
git init
```

```gitignore
# .gitignore
__pycache__/
*.pyc
.env
*.db
.venv/
.streamlit/
```

- [x] **Step 2: Create requirements.txt**

```txt
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
streamlit>=1.40.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
httpx>=0.27.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

- [x] **Step 3: Create .env.example**

```env
# Amadeus API (https://developers.amadeus.com)
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret

# LLM via OpenRouter (https://openrouter.ai)
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514

# Defaults
DEFAULT_ORIGIN=Bangalore
DEFAULT_CURRENCY=INR
```

- [x] **Step 4: Create config.py**

```python
# config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"
    default_origin: str = "Bangalore"
    default_currency: str = "INR"
    db_path: str = "travel_history.db"
    cache_ttl_seconds: int = 86400  # 24 hours
    api_timeout_seconds: int = 10
    api_max_retries: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
```

- [x] **Step 5: Create models.py with Pydantic models**

```python
# models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import TypedDict


class CandidateWindow(BaseModel):
    start: str  # "2026-06-01"
    end: str    # "2026-06-07"


class WeatherResult(BaseModel):
    window: CandidateWindow
    avg_temp: float
    rain_days: int
    avg_humidity: float
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None


class FlightResult(BaseModel):
    window: CandidateWindow
    min_price: float
    avg_price: float
    currency: str = "INR"
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None


class HotelResult(BaseModel):
    window: CandidateWindow
    avg_nightly: float
    currency: str = "INR"
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None


class RankedWindow(BaseModel):
    window: CandidateWindow
    weather_score: float
    flight_score: float
    hotel_score: float
    total_score: float
    estimated_flight_cost: float = 0.0
    estimated_hotel_cost: float = 0.0
    weather_summary: str = ""
    has_historical_data: bool = False


class TravelState(TypedDict, total=False):
    # Input
    destination: str
    origin: str
    date_range: tuple[str, str]
    duration_days: int
    num_travelers: int
    budget_max: float | None
    priorities: dict[str, float]

    # Generated by Supervisor
    candidate_windows: list[dict]

    # Filled by data agents
    weather_data: list[dict]
    flight_data: list[dict]
    hotel_data: list[dict]

    # Filled by scorer
    ranked_windows: list[dict]

    # Filled by synthesizer
    recommendation: str

    # Error tracking
    errors: list[str]
```

- [x] **Step 6: Create empty __init__.py files**

Create empty `__init__.py` in `agents/`, `services/`, `tests/`.

- [x] **Step 7: Set up venv and install deps**

```bash
cd /Users/aankur/workspace/travel-optimizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [x] **Step 8: Write test for models and run**

```python
# tests/test_models.py
from models import CandidateWindow, WeatherResult, FlightResult, HotelResult, RankedWindow


def test_candidate_window():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    assert w.start == "2026-06-01"
    assert w.end == "2026-06-07"


def test_weather_result_defaults():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    r = WeatherResult(window=w, avg_temp=25.0, rain_days=2, avg_humidity=60.0)
    assert r.score == 0.0
    assert r.is_historical is False


def test_flight_result_currency_default():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    r = FlightResult(window=w, min_price=15000.0, avg_price=18000.0)
    assert r.currency == "INR"
```

Run: `pytest tests/test_models.py -v`
Expected: 3 PASS

- [x] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with models, config, and deps"
```

---

## Task 2: Geocoding Service

**Files:**
- Create: `services/geocoding.py`, `tests/test_geocoding.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_geocoding.py
import pytest
from services.geocoding import geocode_city, get_iata_code


def test_geocode_city_returns_lat_lon():
    """Uses live Open-Meteo API (free, no key)."""
    result = geocode_city("Tokyo")
    assert "latitude" in result
    assert "longitude" in result
    assert abs(result["latitude"] - 35.68) < 1.0  # roughly Tokyo


def test_geocode_city_not_found():
    with pytest.raises(ValueError, match="not found"):
        geocode_city("Xyzzyville123")


def test_geocode_returns_name():
    result = geocode_city("Bangalore")
    assert "name" in result
```

Run: `pytest tests/test_geocoding.py -v`
Expected: FAIL (module not found)

- [x] **Step 2: Implement geocoding service**

```python
# services/geocoding.py
import httpx
from config import settings

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode_city(city_name: str) -> dict:
    """Geocode a city name to lat/lon using Open-Meteo.

    Returns dict with keys: name, latitude, longitude, country, country_code.
    Raises ValueError if city not found.
    """
    response = httpx.get(
        GEOCODING_URL,
        params={"name": city_name, "count": 1, "language": "en"},
        timeout=settings.api_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
        raise ValueError(f"City '{city_name}' not found")

    result = data["results"][0]
    return {
        "name": result["name"],
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "country": result.get("country", ""),
        "country_code": result.get("country_code", ""),
    }
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_geocoding.py -v`
Expected: 3 PASS

- [x] **Step 4: Commit**

```bash
git add services/geocoding.py tests/test_geocoding.py
git commit -m "feat: geocoding service using Open-Meteo API"
```

---

## Task 3: In-Memory Cache + SQLite Historical Store

**Files:**
- Create: `cache.py`, `db.py`, `tests/test_cache.py`, `tests/test_db.py`

- [x] **Step 1: Write failing cache tests**

```python
# tests/test_cache.py
import time
from cache import TTLCache


def test_cache_set_and_get():
    c = TTLCache(ttl_seconds=60)
    c.set("key1", {"data": 42})
    assert c.get("key1") == {"data": 42}


def test_cache_miss():
    c = TTLCache(ttl_seconds=60)
    assert c.get("missing") is None


def test_cache_expiry():
    c = TTLCache(ttl_seconds=1)
    c.set("key1", "value")
    time.sleep(1.1)
    assert c.get("key1") is None
```

- [x] **Step 2: Implement cache**

```python
# cache.py
import time
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int = 86400):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()
```

- [x] **Step 3: Run cache tests**

Run: `pytest tests/test_cache.py -v`
Expected: 3 PASS

- [x] **Step 4: Write failing db tests**

```python
# tests/test_db.py
import os
import pytest
from db import HistoryDB


@pytest.fixture
def test_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = HistoryDB(db_path)
    yield db
    db.close()


def test_save_and_query_flight(test_db):
    test_db.save_flight("BLR", "NRT", "2026-06-01", 15000.0, "INR")
    result = test_db.get_flight("BLR", "NRT", "2026-06-01")
    assert result is not None
    assert result["price"] == 15000.0


def test_query_flight_not_found(test_db):
    result = test_db.get_flight("BLR", "NRT", "2026-06-01")
    assert result is None


def test_save_and_query_hotel(test_db):
    test_db.save_hotel("Tokyo", "2026-06-01", 8000.0, "INR")
    result = test_db.get_hotel("Tokyo", "2026-06-01")
    assert result is not None
    assert result["avg_nightly"] == 8000.0


def test_query_similar_date_flight(test_db):
    """Fallback should find data within 7 days of requested date."""
    test_db.save_flight("BLR", "NRT", "2026-06-03", 15000.0, "INR")
    result = test_db.get_flight("BLR", "NRT", "2026-06-01", tolerance_days=7)
    assert result is not None
    assert result["price"] == 15000.0
```

- [x] **Step 5: Implement db.py**

```python
# db.py
import sqlite3
from datetime import datetime


class HistoryDB:
    def __init__(self, db_path: str = "travel_history.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS flight_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                departure_date TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS hotel_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                checkin_date TEXT NOT NULL,
                avg_nightly REAL NOT NULL,
                currency TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    def save_flight(self, origin: str, destination: str, departure_date: str,
                    price: float, currency: str) -> None:
        self._conn.execute(
            "INSERT INTO flight_prices (origin, destination, departure_date, price, currency) "
            "VALUES (?, ?, ?, ?, ?)",
            (origin, destination, departure_date, price, currency),
        )
        self._conn.commit()

    def get_flight(self, origin: str, destination: str, departure_date: str,
                   tolerance_days: int = 0) -> dict | None:
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin = ? AND destination = ? "
                "AND ABS(julianday(departure_date) - julianday(?)) <= ? "
                "ORDER BY ABS(julianday(departure_date) - julianday(?)) ASC, "
                "fetched_at DESC LIMIT 1",
                (origin, destination, departure_date, tolerance_days, departure_date),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin = ? AND destination = ? AND departure_date = ? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (origin, destination, departure_date),
            ).fetchone()
        if row is None:
            return None
        return {"price": row["price"], "currency": row["currency"],
                "fetched_at": row["fetched_at"]}

    def save_hotel(self, city: str, checkin_date: str, avg_nightly: float,
                   currency: str) -> None:
        self._conn.execute(
            "INSERT INTO hotel_prices (city, checkin_date, avg_nightly, currency) "
            "VALUES (?, ?, ?, ?)",
            (city, checkin_date, avg_nightly, currency),
        )
        self._conn.commit()

    def get_hotel(self, city: str, checkin_date: str,
                  tolerance_days: int = 0) -> dict | None:
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city = ? "
                "AND ABS(julianday(checkin_date) - julianday(?)) <= ? "
                "ORDER BY ABS(julianday(checkin_date) - julianday(?)) ASC, "
                "fetched_at DESC LIMIT 1",
                (city, checkin_date, tolerance_days, checkin_date),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city = ? AND checkin_date = ? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (city, checkin_date),
            ).fetchone()
        if row is None:
            return None
        return {"avg_nightly": row["avg_nightly"], "currency": row["currency"],
                "fetched_at": row["fetched_at"]}

    def close(self):
        self._conn.close()
```

- [x] **Step 6: Run db tests**

Run: `pytest tests/test_db.py -v`
Expected: 4 PASS

- [x] **Step 7: Commit**

```bash
git add cache.py db.py tests/test_cache.py tests/test_db.py
git commit -m "feat: TTL cache and SQLite historical data store"
```

---

## Task 4: Weather Client Service

**Files:**
- Create: `services/weather_client.py`, `tests/test_weather_client.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_weather_client.py
import pytest
from services.weather_client import get_weather_for_window


def test_weather_returns_expected_fields():
    """Live test against Open-Meteo Climate API (free)."""
    result = get_weather_for_window(
        latitude=35.68, longitude=139.69,  # Tokyo
        start_date="2026-07-01", end_date="2026-07-07",
    )
    assert "avg_temp" in result
    assert "rain_days" in result
    assert "avg_humidity" in result
    assert isinstance(result["avg_temp"], float)
    assert isinstance(result["rain_days"], int)


def test_weather_invalid_coords():
    with pytest.raises(Exception):
        get_weather_for_window(
            latitude=999.0, longitude=999.0,
            start_date="2026-07-01", end_date="2026-07-07",
        )
```

Run: `pytest tests/test_weather_client.py -v`
Expected: FAIL (module not found)

- [x] **Step 2: Implement weather client**

```python
# services/weather_client.py
import httpx
from datetime import date
from config import settings

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_weather_for_window(
    latitude: float, longitude: float,
    start_date: str, end_date: str,
) -> dict:
    """Fetch weather data for a date window.

    Uses Climate API for future dates (30-year normals),
    Historical API for past dates.

    Returns: {avg_temp, rain_days, avg_humidity}
    """
    is_future = date.fromisoformat(end_date) >= date.today()

    if is_future:
        return _fetch_climate(latitude, longitude, start_date, end_date)
    else:
        return _fetch_historical(latitude, longitude, start_date, end_date)


def _fetch_climate(lat: float, lon: float, start: str, end: str) -> dict:
    # Climate API uses month-day ranges with a model year range
    response = httpx.get(
        CLIMATE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
            "models": "EC_Earth3P_HR",
        },
        timeout=settings.api_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    return _parse_daily(data)


def _fetch_historical(lat: float, lon: float, start: str, end: str) -> dict:
    response = httpx.get(
        HISTORICAL_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        },
        timeout=settings.api_timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    return _parse_daily(data)


def _parse_daily(data: dict) -> dict:
    daily = data.get("daily", {})
    temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    humidity = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]

    avg_temp = sum(temps) / len(temps) if temps else 0.0
    rain_days = sum(1 for p in precip if p > 1.0)  # >1mm = rain day
    avg_humidity = sum(humidity) / len(humidity) if humidity else 0.0

    return {
        "avg_temp": round(avg_temp, 1),
        "rain_days": rain_days,
        "avg_humidity": round(avg_humidity, 1),
    }
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_weather_client.py -v`
Expected: 2 PASS

- [x] **Step 4: Commit**

```bash
git add services/weather_client.py tests/test_weather_client.py
git commit -m "feat: weather client with climate/historical API selection"
```

---

## Task 5: Amadeus Client Service

**Files:**
- Create: `services/amadeus_client.py`, `tests/test_amadeus_client.py`

- [x] **Step 1: Write failing tests (mocked — Amadeus needs real keys)**

```python
# tests/test_amadeus_client.py
import pytest
from unittest.mock import patch, MagicMock
from services.amadeus_client import AmadeusClient


@pytest.fixture
def client():
    return AmadeusClient(client_id="test_id", client_secret="test_secret")


def test_auth_token_request(client):
    """Test that auth fetches a token."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "tok123", "expires_in": 1799}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        client._authenticate()
        assert client._token == "tok123"


def test_search_flights_builds_correct_params(client):
    """Test flight search constructs proper API request."""
    client._token = "tok123"
    client._token_expiry = 9999999999.0

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"price": {"total": "15000"}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get:
        result = client.search_flights("BLR", "NRT", "2026-07-01", "INR")
        call_params = mock_get.call_args[1]["params"]
        assert call_params["originLocationCode"] == "BLR"
        assert call_params["destinationLocationCode"] == "NRT"
        assert call_params["currencyCode"] == "INR"


def test_search_hotels_builds_correct_params(client):
    """Test hotel search constructs proper API request."""
    client._token = "tok123"
    client._token_expiry = 9999999999.0

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": []}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get:
        client.search_hotels("TYO", "2026-07-01", "2026-07-07", currency="INR")
        call_params = mock_get.call_args[1]["params"]
        assert call_params["cityCode"] == "TYO"
```

Run: `pytest tests/test_amadeus_client.py -v`
Expected: FAIL (module not found)

- [x] **Step 2: Implement Amadeus client**

```python
# services/amadeus_client.py
import time
import httpx
from config import settings

AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
FLIGHTS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
HOTEL_LIST_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
HOTEL_OFFERS_URL = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
IATA_URL = "https://test.api.amadeus.com/v1/reference-data/locations"


class AmadeusClient:
    def __init__(self, client_id: str = "", client_secret: str = ""):
        self._client_id = client_id or settings.amadeus_client_id
        self._client_secret = client_secret or settings.amadeus_client_secret
        self._token: str = ""
        self._token_expiry: float = 0.0

    def _authenticate(self) -> None:
        response = httpx.post(
            AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"] - 60  # refresh 1min early

    def _ensure_auth(self) -> None:
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def search_flights(self, origin: str, destination: str,
                       departure_date: str, currency: str = "INR",
                       adults: int = 1) -> dict:
        self._ensure_auth()
        response = httpx.get(
            FLIGHTS_URL,
            params={
                "originLocationCode": origin,
                "destinationLocationCode": destination,
                "departureDate": departure_date,
                "adults": adults,
                "currencyCode": currency,
                "max": 5,
            },
            headers=self._headers(),
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def search_hotels(self, city_code: str, checkin: str, checkout: str,
                      currency: str = "INR", adults: int = 1) -> dict:
        self._ensure_auth()
        # Step 1: get hotel IDs for city (cached externally)
        response = httpx.get(
            HOTEL_LIST_URL,
            params={"cityCode": city_code},
            headers=self._headers(),
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        hotels = response.json().get("data", [])[:10]  # top 10
        if not hotels:
            return {"data": []}

        hotel_ids = [h["hotelId"] for h in hotels]

        # Step 2: get offers
        response = httpx.get(
            HOTEL_OFFERS_URL,
            params={
                "hotelIds": ",".join(hotel_ids),
                "checkInDate": checkin,
                "checkOutDate": checkout,
                "adults": adults,
                "currency": currency,
                "bestRateOnly": "true",
            },
            headers=self._headers(),
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_iata_code(self, city_name: str) -> str | None:
        """Look up IATA code for a city."""
        self._ensure_auth()
        response = httpx.get(
            IATA_URL,
            params={
                "keyword": city_name,
                "subType": "CITY,AIRPORT",
            },
            headers=self._headers(),
            timeout=settings.api_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if data:
            return data[0].get("iataCode")
        return None
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_amadeus_client.py -v`
Expected: 3 PASS

- [x] **Step 4: Commit**

```bash
git add services/amadeus_client.py tests/test_amadeus_client.py
git commit -m "feat: Amadeus API client with auth, flights, hotels, and IATA lookup"
```

---

## Task 6: Supervisor Agent

**Files:**
- Create: `agents/supervisor.py`, `tests/test_supervisor.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_supervisor.py
from agents.supervisor import generate_candidate_windows


def test_generates_correct_number_of_windows():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", duration_days=7)
    # 30 days, 7 day windows, rolling weekly = ~4 windows
    assert len(windows) >= 3
    assert len(windows) <= 5


def test_window_duration_matches():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", duration_days=7)
    from datetime import date
    for w in windows:
        start = date.fromisoformat(w["start"])
        end = date.fromisoformat(w["end"])
        assert (end - start).days == 7


def test_windows_stay_within_range():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", duration_days=7)
    for w in windows:
        assert w["start"] >= "2026-06-01"
        assert w["end"] <= "2026-06-30"


def test_wide_range_more_windows():
    windows = generate_candidate_windows("2026-06-01", "2026-09-30", duration_days=7)
    assert len(windows) >= 10
```

Run: `pytest tests/test_supervisor.py -v`
Expected: FAIL

- [x] **Step 2: Implement supervisor**

```python
# agents/supervisor.py
from datetime import date, timedelta
from models import TravelState


def generate_candidate_windows(
    start_date: str, end_date: str, duration_days: int = 7
) -> list[dict]:
    """Generate rolling candidate travel windows within a date range."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows = []

    current = start
    while current + timedelta(days=duration_days) <= end:
        windows.append({
            "start": current.isoformat(),
            "end": (current + timedelta(days=duration_days)).isoformat(),
        })
        current += timedelta(days=7)  # roll weekly

    return windows


def supervisor_node(state: TravelState) -> dict:
    """LangGraph node: parse input and generate candidate windows."""
    windows = generate_candidate_windows(
        state["date_range"][0],
        state["date_range"][1],
        state.get("duration_days", 7),
    )
    return {"candidate_windows": windows, "errors": state.get("errors", [])}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_supervisor.py -v`
Expected: 4 PASS

- [x] **Step 4: Commit**

```bash
git add agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: supervisor agent with candidate window generation"
```

---

## Task 7: Weather Agent

**Files:**
- Create: `agents/weather.py`, `tests/test_weather_agent.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_weather_agent.py
from unittest.mock import patch
from agents.weather import weather_node, score_weather


def test_score_weather_ideal():
    score = score_weather(avg_temp=24.0, rain_days=0, avg_humidity=50.0)
    assert score > 0.8


def test_score_weather_bad():
    score = score_weather(avg_temp=40.0, rain_days=6, avg_humidity=90.0)
    assert score < 0.3


def test_score_weather_range():
    score = score_weather(avg_temp=24.0, rain_days=2, avg_humidity=60.0)
    assert 0.0 <= score <= 1.0


def test_weather_node_populates_state():
    mock_weather = {"avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0}

    state = {
        "destination": "Tokyo",
        "candidate_windows": [
            {"start": "2026-07-01", "end": "2026-07-07"},
        ],
        "errors": [],
    }

    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}):
        with patch("agents.weather.get_weather_for_window", return_value=mock_weather):
            result = weather_node(state)

    assert len(result["weather_data"]) == 1
    assert result["weather_data"][0]["avg_temp"] == 25.0
    assert 0.0 <= result["weather_data"][0]["score"] <= 1.0
```

Run: `pytest tests/test_weather_agent.py -v`
Expected: FAIL

- [x] **Step 2: Implement weather agent**

```python
# agents/weather.py
from models import TravelState
from services.geocoding import geocode_city
from services.weather_client import get_weather_for_window


def score_weather(avg_temp: float, rain_days: int, avg_humidity: float,
                  ideal_temp_min: float = 20.0, ideal_temp_max: float = 28.0) -> float:
    """Score weather 0-1. Higher is better."""
    # Temperature score: 1.0 if in ideal range, drops off outside
    if ideal_temp_min <= avg_temp <= ideal_temp_max:
        temp_score = 1.0
    else:
        distance = min(abs(avg_temp - ideal_temp_min), abs(avg_temp - ideal_temp_max))
        temp_score = max(0.0, 1.0 - distance / 20.0)

    # Rain score: 0 rain days = 1.0, 7 = 0.0
    rain_score = max(0.0, 1.0 - rain_days / 7.0)

    # Humidity score: 40-60% ideal, penalize above 75%
    if avg_humidity <= 60:
        humidity_score = 1.0
    elif avg_humidity <= 80:
        humidity_score = 1.0 - (avg_humidity - 60) / 40.0
    else:
        humidity_score = max(0.0, 0.5 - (avg_humidity - 80) / 40.0)

    # Weighted combination
    return round(temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25, 3)


def weather_node(state: TravelState) -> dict:
    """LangGraph node: fetch weather data for all candidate windows."""
    destination = state["destination"]
    windows = state["candidate_windows"]
    errors = list(state.get("errors", []))

    try:
        geo = geocode_city(destination)
    except ValueError as e:
        errors.append(f"Weather: geocoding failed — {e}")
        return {"weather_data": [], "errors": errors}

    results = []
    for window in windows:
        try:
            data = get_weather_for_window(
                geo["latitude"], geo["longitude"],
                window["start"], window["end"],
            )
            score = score_weather(data["avg_temp"], data["rain_days"], data["avg_humidity"])
            results.append({
                "window": window,
                "avg_temp": data["avg_temp"],
                "rain_days": data["rain_days"],
                "avg_humidity": data["avg_humidity"],
                "score": score,
                "is_historical": False,
            })
        except Exception as e:
            errors.append(f"Weather: failed for {window['start']} — {e}")

    return {"weather_data": results, "errors": errors}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_weather_agent.py -v`
Expected: 4 PASS

- [x] **Step 4: Commit**

```bash
git add agents/weather.py tests/test_weather_agent.py
git commit -m "feat: weather agent with scoring and Open-Meteo integration"
```

---

## Task 8: Flights Agent

**Files:**
- Create: `agents/flights.py`, `tests/test_flights_agent.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_flights_agent.py
from unittest.mock import patch, MagicMock
from agents.flights import flights_node, parse_flight_prices


def test_parse_flight_prices_extracts_min_and_avg():
    api_response = {
        "data": [
            {"price": {"total": "15000.00"}},
            {"price": {"total": "18000.00"}},
            {"price": {"total": "20000.00"}},
        ]
    }
    result = parse_flight_prices(api_response)
    assert result["min_price"] == 15000.0
    assert result["avg_price"] == 17666.67


def test_parse_flight_prices_empty():
    result = parse_flight_prices({"data": []})
    assert result is None


def test_flights_node_with_mock_client():
    state = {
        "origin": "Bangalore",
        "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}],
        "num_travelers": 1,
        "errors": [],
    }
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = {
        "data": [{"price": {"total": "15000.00"}}]
    }

    with patch("agents.flights._get_client", return_value=mock_client):
        result = flights_node(state)

    assert len(result["flight_data"]) == 1
    assert result["flight_data"][0]["min_price"] == 15000.0
```

Run: `pytest tests/test_flights_agent.py -v`
Expected: FAIL

- [x] **Step 2: Implement flights agent**

```python
# agents/flights.py
from models import TravelState
from services.amadeus_client import AmadeusClient
from db import HistoryDB
from config import settings

_client: AmadeusClient | None = None


def _get_client() -> AmadeusClient:
    global _client
    if _client is None:
        _client = AmadeusClient()
    return _client


def parse_flight_prices(api_response: dict) -> dict | None:
    """Extract min and avg price from Amadeus flight offers response."""
    offers = api_response.get("data", [])
    if not offers:
        return None
    prices = [float(o["price"]["total"]) for o in offers]
    return {
        "min_price": min(prices),
        "avg_price": round(sum(prices) / len(prices), 2),
    }


def flights_node(state: TravelState) -> dict:
    """LangGraph node: fetch flight prices for all candidate windows."""
    client = _get_client()
    windows = state["candidate_windows"]
    errors = list(state.get("errors", []))
    currency = settings.default_currency

    # Resolve IATA codes
    try:
        origin_iata = client.get_iata_code(state["origin"])
        dest_iata = client.get_iata_code(state["destination"])
        if not origin_iata or not dest_iata:
            raise ValueError(f"Could not resolve IATA: origin={origin_iata}, dest={dest_iata}")
    except Exception as e:
        errors.append(f"Flights: IATA lookup failed — {e}")
        return {"flight_data": [], "errors": errors}

    db = HistoryDB(settings.db_path)
    results = []

    for window in windows:
        try:
            response = client.search_flights(
                origin_iata, dest_iata, window["start"],
                currency=currency,
                adults=state.get("num_travelers", 1),
            )
            parsed = parse_flight_prices(response)
            if parsed:
                # Save to history
                db.save_flight(origin_iata, dest_iata, window["start"],
                               parsed["min_price"], currency)
                results.append({
                    "window": window,
                    "min_price": parsed["min_price"],
                    "avg_price": parsed["avg_price"],
                    "currency": currency,
                    "score": 0.0,  # scored later by scorer
                    "is_historical": False,
                })
            else:
                # Try historical fallback
                hist = db.get_flight(origin_iata, dest_iata, window["start"],
                                     tolerance_days=7)
                if hist:
                    results.append({
                        "window": window,
                        "min_price": hist["price"],
                        "avg_price": hist["price"],
                        "currency": hist["currency"],
                        "score": 0.0,
                        "is_historical": True,
                        "fetched_at": hist["fetched_at"],
                    })
        except Exception as e:
            # Fallback to historical
            hist = db.get_flight(origin_iata, dest_iata, window["start"],
                                 tolerance_days=7)
            if hist:
                results.append({
                    "window": window,
                    "min_price": hist["price"],
                    "avg_price": hist["price"],
                    "currency": hist["currency"],
                    "score": 0.0,
                    "is_historical": True,
                    "fetched_at": hist["fetched_at"],
                })
            else:
                errors.append(f"Flights: failed for {window['start']} — {e}")

    return {"flight_data": results, "errors": errors}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_flights_agent.py -v`
Expected: 3 PASS

- [x] **Step 4: Commit**

```bash
git add agents/flights.py tests/test_flights_agent.py
git commit -m "feat: flights agent with Amadeus integration and historical fallback"
```

---

## Task 9: Hotels Agent

**Files:**
- Create: `agents/hotels.py`, `tests/test_hotels_agent.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_hotels_agent.py
from unittest.mock import patch, MagicMock
from agents.hotels import hotels_node, parse_hotel_prices


def test_parse_hotel_prices():
    api_response = {
        "data": [
            {"offers": [{"price": {"total": "8000.00"}}]},
            {"offers": [{"price": {"total": "12000.00"}}]},
        ]
    }
    result = parse_hotel_prices(api_response)
    assert result["avg_nightly"] == 10000.0


def test_parse_hotel_prices_empty():
    result = parse_hotel_prices({"data": []})
    assert result is None


def test_hotels_node_with_mock():
    state = {
        "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}],
        "errors": [],
    }
    mock_client = MagicMock()
    mock_client.get_iata_code.return_value = "TYO"
    mock_client.search_hotels.return_value = {
        "data": [{"offers": [{"price": {"total": "8000.00"}}]}]
    }

    with patch("agents.hotels._get_client", return_value=mock_client):
        result = hotels_node(state)

    assert len(result["hotel_data"]) == 1
    assert result["hotel_data"][0]["avg_nightly"] == 8000.0
```

Run: `pytest tests/test_hotels_agent.py -v`
Expected: FAIL

- [x] **Step 2: Implement hotels agent**

```python
# agents/hotels.py
from models import TravelState
from services.amadeus_client import AmadeusClient
from db import HistoryDB
from config import settings

_client: AmadeusClient | None = None


def _get_client() -> AmadeusClient:
    global _client
    if _client is None:
        _client = AmadeusClient()
    return _client


def parse_hotel_prices(api_response: dict) -> dict | None:
    """Extract average nightly rate from Amadeus hotel offers."""
    hotels = api_response.get("data", [])
    if not hotels:
        return None
    prices = []
    for hotel in hotels:
        offers = hotel.get("offers", [])
        if offers:
            prices.append(float(offers[0]["price"]["total"]))
    if not prices:
        return None
    return {"avg_nightly": round(sum(prices) / len(prices), 2)}


def hotels_node(state: TravelState) -> dict:
    """LangGraph node: fetch hotel prices for all candidate windows."""
    client = _get_client()
    windows = state["candidate_windows"]
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    destination = state["destination"]

    try:
        city_code = client.get_iata_code(destination)
        if not city_code:
            raise ValueError(f"Could not resolve IATA for {destination}")
    except Exception as e:
        errors.append(f"Hotels: IATA lookup failed — {e}")
        return {"hotel_data": [], "errors": errors}

    db = HistoryDB(settings.db_path)
    results = []

    for window in windows:
        try:
            response = client.search_hotels(
                city_code, window["start"], window["end"], currency=currency,
            )
            parsed = parse_hotel_prices(response)
            if parsed:
                db.save_hotel(destination, window["start"],
                              parsed["avg_nightly"], currency)
                results.append({
                    "window": window,
                    "avg_nightly": parsed["avg_nightly"],
                    "currency": currency,
                    "score": 0.0,
                    "is_historical": False,
                })
            else:
                hist = db.get_hotel(destination, window["start"], tolerance_days=7)
                if hist:
                    results.append({
                        "window": window,
                        "avg_nightly": hist["avg_nightly"],
                        "currency": hist["currency"],
                        "score": 0.0,
                        "is_historical": True,
                        "fetched_at": hist["fetched_at"],
                    })
        except Exception as e:
            hist = db.get_hotel(destination, window["start"], tolerance_days=7)
            if hist:
                results.append({
                    "window": window,
                    "avg_nightly": hist["avg_nightly"],
                    "currency": hist["currency"],
                    "score": 0.0,
                    "is_historical": True,
                    "fetched_at": hist["fetched_at"],
                })
            else:
                errors.append(f"Hotels: failed for {window['start']} — {e}")

    return {"hotel_data": results, "errors": errors}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_hotels_agent.py -v`
Expected: 3 PASS

- [x] **Step 4: Commit**

```bash
git add agents/hotels.py tests/test_hotels_agent.py
git commit -m "feat: hotels agent with Amadeus integration and historical fallback"
```

---

## Task 10: Scorer Agent

**Files:**
- Create: `agents/scorer.py`, `tests/test_scorer.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
from agents.scorer import scorer_node, normalize_scores


def test_normalize_scores():
    values = [100.0, 200.0, 300.0]
    # Lower price is better for flights/hotels → invert
    result = normalize_scores(values, lower_is_better=True)
    assert result[0] == 1.0  # cheapest = best
    assert result[2] == 0.0  # most expensive = worst


def test_normalize_scores_higher_is_better():
    values = [0.2, 0.5, 0.9]
    result = normalize_scores(values, lower_is_better=False)
    assert result[2] == 1.0
    assert result[0] == 0.0


def test_normalize_single_value():
    result = normalize_scores([42.0], lower_is_better=True)
    assert result == [1.0]


def test_scorer_node_ranks_windows():
    state = {
        "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
        "weather_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.9},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "score": 0.5},
        ],
        "flight_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "min_price": 25000},
        ],
        "hotel_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "avg_nightly": 8000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "avg_nightly": 6000},
        ],
        "errors": [],
    }
    result = scorer_node(state)
    ranked = result["ranked_windows"]
    assert len(ranked) == 2
    # First window should rank higher (better weather + cheaper flights)
    assert ranked[0]["total_score"] >= ranked[1]["total_score"]


def test_scorer_handles_missing_dimension():
    """If hotel data is empty, scorer should still work with remaining data."""
    state = {
        "priorities": {"weather": 0.5, "flights": 0.3, "hotels": 0.2},
        "weather_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.8},
        ],
        "flight_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000},
        ],
        "hotel_data": [],
        "errors": [],
    }
    result = scorer_node(state)
    assert len(result["ranked_windows"]) == 1
    assert result["ranked_windows"][0]["hotel_score"] == 0.0
```

Run: `pytest tests/test_scorer.py -v`
Expected: FAIL

- [x] **Step 2: Implement scorer**

```python
# agents/scorer.py
from models import TravelState


def normalize_scores(values: list[float], lower_is_better: bool = False) -> list[float]:
    """Normalize a list of values to 0-1 range."""
    if len(values) <= 1:
        return [1.0] * len(values)
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0] * len(values)
    if lower_is_better:
        return [round((max_v - v) / (max_v - min_v), 3) for v in values]
    return [round((v - min_v) / (max_v - min_v), 3) for v in values]


def scorer_node(state: TravelState) -> dict:
    """LangGraph node: normalize scores and rank candidate windows."""
    priorities = state.get("priorities", {"weather": 0.4, "flights": 0.3, "hotels": 0.3})
    weather_data = state.get("weather_data", [])
    flight_data = state.get("flight_data", [])
    hotel_data = state.get("hotel_data", [])

    # Build lookup by window start date
    weather_by_start = {d["window"]["start"]: d for d in weather_data}
    flight_by_start = {d["window"]["start"]: d for d in flight_data}
    hotel_by_start = {d["window"]["start"]: d for d in hotel_data}

    # Collect all unique window starts
    all_starts = set()
    for d in weather_data:
        all_starts.add(d["window"]["start"])
    for d in flight_data:
        all_starts.add(d["window"]["start"])
    for d in hotel_data:
        all_starts.add(d["window"]["start"])

    all_starts = sorted(all_starts)

    # Normalize flight and hotel prices (lower is better)
    flight_prices = [flight_by_start[s]["min_price"] for s in all_starts if s in flight_by_start]
    hotel_prices = [hotel_by_start[s]["avg_nightly"] for s in all_starts if s in hotel_by_start]

    flight_norm = normalize_scores(flight_prices, lower_is_better=True) if flight_prices else []
    hotel_norm = normalize_scores(hotel_prices, lower_is_better=True) if hotel_prices else []

    # Map normalized scores back
    flight_starts = [s for s in all_starts if s in flight_by_start]
    hotel_starts = [s for s in all_starts if s in hotel_by_start]
    flight_score_map = dict(zip(flight_starts, flight_norm))
    hotel_score_map = dict(zip(hotel_starts, hotel_norm))

    # Reweight priorities if a dimension is missing
    active_weights = {}
    if weather_data:
        active_weights["weather"] = priorities.get("weather", 0.4)
    if flight_data:
        active_weights["flights"] = priorities.get("flights", 0.3)
    if hotel_data:
        active_weights["hotels"] = priorities.get("hotels", 0.3)

    total_weight = sum(active_weights.values()) or 1.0
    norm_weights = {k: v / total_weight for k, v in active_weights.items()}

    # Score each window
    ranked = []
    for start in all_starts:
        w_score = weather_by_start.get(start, {}).get("score", 0.0)
        f_score = flight_score_map.get(start, 0.0)
        h_score = hotel_score_map.get(start, 0.0)

        total = (
            w_score * norm_weights.get("weather", 0.0)
            + f_score * norm_weights.get("flights", 0.0)
            + h_score * norm_weights.get("hotels", 0.0)
        )

        window = weather_by_start.get(start, flight_by_start.get(start, hotel_by_start.get(start, {})))
        window_obj = window.get("window", {"start": start, "end": ""})

        has_hist = any([
            weather_by_start.get(start, {}).get("is_historical", False),
            flight_by_start.get(start, {}).get("is_historical", False),
            hotel_by_start.get(start, {}).get("is_historical", False),
        ])

        ranked.append({
            "window": window_obj,
            "weather_score": round(w_score, 3),
            "flight_score": round(f_score, 3),
            "hotel_score": round(h_score, 3),
            "total_score": round(total, 3),
            "estimated_flight_cost": flight_by_start.get(start, {}).get("min_price", 0.0),
            "estimated_hotel_cost": hotel_by_start.get(start, {}).get("avg_nightly", 0.0),
            "has_historical_data": has_hist,
        })

    ranked.sort(key=lambda x: x["total_score"], reverse=True)
    return {"ranked_windows": ranked, "errors": state.get("errors", [])}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_scorer.py -v`
Expected: 5 PASS

- [x] **Step 4: Commit**

```bash
git add agents/scorer.py tests/test_scorer.py
git commit -m "feat: scorer agent with normalization and priority reweighting"
```

---

## Task 11: Synthesizer Agent

**Files:**
- Create: `agents/synthesizer.py`, `tests/test_synthesizer.py`

- [x] **Step 1: Write failing tests**

```python
# tests/test_synthesizer.py
from unittest.mock import patch, MagicMock
from agents.synthesizer import synthesizer_node, format_ranked_data_fallback


def test_format_fallback_produces_readable_text():
    ranked = [
        {
            "window": {"start": "2026-07-01", "end": "2026-07-07"},
            "total_score": 0.85,
            "weather_score": 0.9,
            "flight_score": 0.8,
            "hotel_score": 0.75,
            "estimated_flight_cost": 15000,
            "estimated_hotel_cost": 8000,
            "has_historical_data": False,
        }
    ]
    result = format_ranked_data_fallback(ranked)
    assert "2026-07-01" in result
    assert "15000" in result or "15,000" in result


def test_synthesizer_node_uses_llm():
    state = {
        "destination": "Tokyo",
        "origin": "Bangalore",
        "ranked_windows": [
            {
                "window": {"start": "2026-07-01", "end": "2026-07-07"},
                "total_score": 0.85,
                "weather_score": 0.9,
                "flight_score": 0.8,
                "hotel_score": 0.75,
                "estimated_flight_cost": 15000,
                "estimated_hotel_cost": 8000,
                "has_historical_data": False,
            }
        ],
        "errors": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July 1-7 is the best time to visit Tokyo.")

    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)

    assert "recommendation" in result
    assert len(result["recommendation"]) > 0


def test_synthesizer_falls_back_on_llm_failure():
    state = {
        "destination": "Tokyo",
        "origin": "Bangalore",
        "ranked_windows": [
            {
                "window": {"start": "2026-07-01", "end": "2026-07-07"},
                "total_score": 0.85,
                "weather_score": 0.9,
                "flight_score": 0.8,
                "hotel_score": 0.75,
                "estimated_flight_cost": 15000,
                "estimated_hotel_cost": 8000,
                "has_historical_data": False,
            }
        ],
        "errors": [],
    }

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")

    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)

    assert "recommendation" in result
    assert "2026-07-01" in result["recommendation"]
```

Run: `pytest tests/test_synthesizer.py -v`
Expected: FAIL

- [x] **Step 2: Implement synthesizer**

```python
# agents/synthesizer.py
from langchain_openai import ChatOpenAI
from models import TravelState
from config import settings

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _llm


def format_ranked_data_fallback(ranked: list[dict], top_n: int = 3) -> str:
    """Format ranked windows as plain text (fallback when LLM unavailable)."""
    lines = []
    for i, r in enumerate(ranked[:top_n], 1):
        w = r["window"]
        lines.append(
            f"#{i}: {w['start']} to {w['end']} "
            f"(score: {r['total_score']:.2f}) — "
            f"Weather: {r['weather_score']:.2f}, "
            f"Flights: ~{r['estimated_flight_cost']:.0f} {r.get('currency', 'INR')}, "
            f"Hotels: ~{r['estimated_hotel_cost']:.0f}/night"
            + (" [some data estimated from history]" if r.get("has_historical_data") else "")
        )
    return "\n".join(lines)


def synthesizer_node(state: TravelState) -> dict:
    """LangGraph node: generate natural language recommendation."""
    ranked = state.get("ranked_windows", [])
    errors = list(state.get("errors", []))
    destination = state.get("destination", "the destination")
    origin = state.get("origin", "your city")

    if not ranked:
        return {"recommendation": "No data available to make a recommendation.", "errors": errors}

    top = ranked[:5]
    data_summary = format_ranked_data_fallback(top, top_n=5)

    prompt = (
        f"You are a travel advisor. Based on the following ranked travel windows "
        f"for a trip from {origin} to {destination}, write a concise recommendation "
        f"(3-5 sentences) explaining which dates are best and why. Mention weather, "
        f"flight cost, and hotel cost. If any data is estimated from history, note that.\n\n"
        f"{data_summary}"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        return {"recommendation": response.content, "errors": errors}
    except Exception as e:
        errors.append(f"Synthesizer: LLM failed — {e}")
        return {"recommendation": data_summary, "errors": errors}
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_synthesizer.py -v`
Expected: 3 PASS

- [x] **Step 4: Commit**

```bash
git add agents/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: synthesizer agent with LLM recommendation and fallback"
```

---

## Task 12: LangGraph Definition

**Files:**
- Create: `graph.py`, `tests/test_graph.py`

- [x] **Step 1: Write failing test**

```python
# tests/test_graph.py
from unittest.mock import patch, MagicMock
from graph import build_graph


def test_graph_builds_without_error():
    g = build_graph()
    assert g is not None


def test_graph_full_run_with_mocks():
    """Integration test with all external calls mocked."""
    mock_weather = {
        "avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0,
    }
    mock_flights_response = {
        "data": [{"price": {"total": "15000.00"}}]
    }
    mock_hotels_response = {
        "data": [{"offers": [{"price": {"total": "8000.00"}}]}]
    }
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = mock_flights_response
    mock_client.search_hotels.return_value = mock_hotels_response

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July is great for Tokyo.")

    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}), \
         patch("agents.weather.get_weather_for_window", return_value=mock_weather), \
         patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.synthesizer._get_llm", return_value=mock_llm), \
         patch("agents.flights.HistoryDB") as mock_fdb, \
         patch("agents.hotels.HistoryDB") as mock_hdb:

        mock_fdb.return_value = MagicMock()
        mock_hdb.return_value = MagicMock()

        g = build_graph()
        result = g.invoke({
            "destination": "Tokyo",
            "origin": "Bangalore",
            "date_range": ("2026-07-01", "2026-07-28"),
            "duration_days": 7,
            "num_travelers": 1,
            "budget_max": None,
            "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
            "errors": [],
        })

    assert "ranked_windows" in result
    assert len(result["ranked_windows"]) > 0
    assert "recommendation" in result
    assert len(result["recommendation"]) > 0
```

Run: `pytest tests/test_graph.py -v`
Expected: FAIL

- [x] **Step 2: Implement graph**

```python
# graph.py
from langgraph.graph import StateGraph, END
from models import TravelState
from agents.supervisor import supervisor_node
from agents.weather import weather_node
from agents.flights import flights_node
from agents.hotels import hotels_node
from agents.scorer import scorer_node
from agents.synthesizer import synthesizer_node


def build_graph() -> StateGraph:
    """Build and compile the travel optimizer LangGraph."""
    graph = StateGraph(TravelState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather", weather_node)
    graph.add_node("flights", flights_node)
    graph.add_node("hotels", hotels_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Supervisor → parallel data agents
    graph.add_edge("supervisor", "weather")
    graph.add_edge("supervisor", "flights")
    graph.add_edge("supervisor", "hotels")

    # Data agents → scorer
    graph.add_edge("weather", "scorer")
    graph.add_edge("flights", "scorer")
    graph.add_edge("hotels", "scorer")

    # Scorer → synthesizer → end
    graph.add_edge("scorer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
```

- [x] **Step 3: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: 2 PASS

- [x] **Step 4: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: LangGraph definition with parallel fan-out and sequential scoring"
```

---

## Task 13: Streamlit UI

**Files:**
- Create: `app.py`

- [x] **Step 1: Implement Streamlit app**

```python
# app.py
import streamlit as st
import pandas as pd
from graph import build_graph
from config import settings

st.set_page_config(page_title="Travel Optimizer", layout="wide")
st.title("Travel Optimizer")
st.caption("Find the best time to visit any destination")

# --- Sidebar: Inputs ---
with st.sidebar:
    st.header("Trip Details")
    destination = st.text_input("Destination", placeholder="Tokyo, Japan")
    origin = st.text_input("Origin", value=settings.default_origin)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From")
    with col2:
        end_date = st.date_input("To")

    duration = st.slider("Trip duration (days)", 3, 21, 7)
    travelers = st.number_input("Travelers", 1, 10, 1)
    budget = st.number_input("Budget ceiling (optional, 0 = no limit)", 0, 1000000, 0)

    st.header("Priorities")
    w_weather = st.slider("Weather importance", 0.0, 1.0, 0.4, 0.05)
    w_flights = st.slider("Flight cost importance", 0.0, 1.0, 0.3, 0.05)
    w_hotels = st.slider("Hotel cost importance", 0.0, 1.0, 0.3, 0.05)

    search = st.button("Find Best Time", type="primary", use_container_width=True)

# --- Main: Results ---
if search:
    if not destination:
        st.error("Please enter a destination.")
    else:
        # Normalize priorities
        total_w = w_weather + w_flights + w_hotels
        if total_w == 0:
            total_w = 1.0
        priorities = {
            "weather": w_weather / total_w,
            "flights": w_flights / total_w,
            "hotels": w_hotels / total_w,
        }

        state = {
            "destination": destination,
            "origin": origin,
            "date_range": (start_date.isoformat(), end_date.isoformat()),
            "duration_days": duration,
            "num_travelers": travelers,
            "budget_max": budget if budget > 0 else None,
            "priorities": priorities,
            "errors": [],
        }

        with st.spinner("Searching for the best travel windows..."):
            graph = build_graph()
            result = graph.invoke(state)

        # Show errors if any
        if result.get("errors"):
            with st.expander("Warnings / Errors", expanded=False):
                for err in result["errors"]:
                    st.warning(err)

        # Recommendation
        ranked = result.get("ranked_windows", [])
        recommendation = result.get("recommendation", "")

        if not ranked:
            st.error("No results found. Try a wider date range or different destination.")
        else:
            st.header("Recommendation")
            st.write(recommendation)

            # Top 3 cards
            st.header("Top Windows")
            cols = st.columns(min(3, len(ranked)))
            for i, r in enumerate(ranked[:3]):
                with cols[i]:
                    w = r["window"]
                    hist_badge = " (estimated)" if r.get("has_historical_data") else ""
                    st.metric(
                        label=f"#{i+1}: {w['start']} → {w['end']}",
                        value=f"Score: {r['total_score']:.2f}",
                    )
                    st.caption(
                        f"Weather: {r['weather_score']:.2f} | "
                        f"Flight: ~{r['estimated_flight_cost']:,.0f} | "
                        f"Hotel: ~{r['estimated_hotel_cost']:,.0f}/night"
                        f"{hist_badge}"
                    )

            # Charts
            st.header("Comparison")
            if ranked:
                df = pd.DataFrame([
                    {
                        "Window": r["window"]["start"],
                        "Weather Score": r["weather_score"],
                        "Flight Score": r["flight_score"],
                        "Hotel Score": r["hotel_score"],
                        "Total Score": r["total_score"],
                        "Flight Cost": r["estimated_flight_cost"],
                        "Hotel Cost/Night": r["estimated_hotel_cost"],
                    }
                    for r in ranked
                ])
                df = df.set_index("Window")

                tab1, tab2, tab3 = st.tabs(["Scores", "Flight Prices", "Hotel Prices"])
                with tab1:
                    st.bar_chart(df[["Weather Score", "Flight Score", "Hotel Score", "Total Score"]])
                with tab2:
                    st.line_chart(df["Flight Cost"])
                with tab3:
                    st.line_chart(df["Hotel Cost/Night"])
```

- [x] **Step 2: Test manually**

```bash
cd /Users/aankur/workspace/travel-optimizer
source .venv/bin/activate
streamlit run app.py
```

Expected: Browser opens, sidebar shows inputs, search button triggers the graph.

- [x] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI with sidebar inputs, result cards, and comparison charts"
```

---

## Task 14: README and Final Polish

**Files:**
- Create: `README.md`

- [x] **Step 1: Write README**

```markdown
# Travel Optimizer

Find the best time to visit any destination based on weather, flight prices, and hotel costs.

Built with LangGraph (multi-agent orchestration), Streamlit (UI), Amadeus API (flights + hotels), and Open-Meteo (weather).

## Setup

1. Clone and install:
   ```bash
   cd travel-optimizer
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create `.env` from template:
   ```bash
   cp .env.example .env
   ```

3. Add your API keys to `.env`:
   - **Amadeus**: Sign up at https://developers.amadeus.com (free tier)
   - **OpenRouter**: Sign up at https://openrouter.ai (for LLM recommendations)

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest tests/ -v
```

## Architecture

```
User Input → Supervisor → [Weather | Flights | Hotels] (parallel) → Scorer → Synthesizer → UI
```

- **Supervisor**: Generates candidate date windows
- **Weather Agent**: Open-Meteo climate/historical data
- **Flight Agent**: Amadeus flight offers + SQLite fallback
- **Hotel Agent**: Amadeus hotel offers + SQLite fallback
- **Scorer**: Normalizes and ranks by weighted score
- **Synthesizer**: LLM-generated recommendation (falls back to raw data)
```

- [x] **Step 2: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [x] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, run, and architecture overview"
```
