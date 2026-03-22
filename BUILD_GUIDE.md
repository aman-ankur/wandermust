# Travel Optimizer — Complete Build Guide

> **For any coding agent:** This document contains everything needed to build the project end-to-end. Read this entire file before starting. Implement tasks in order. Each task has complete code, test commands, and commit instructions.

---

## Project Overview

A **LangGraph multi-agent system** with **Streamlit UI** that finds the best time to visit a destination by analyzing weather, flight prices, and hotel costs across candidate date windows.

**Key principle:** Supervisor generates candidate windows → 3 data agents fetch in parallel → scorer ranks → synthesizer explains results.

## Prerequisites

Before starting:
1. **Python 3.11+** installed
2. **Amadeus API key** — sign up free at https://developers.amadeus.com (self-service, 500 calls/month)
3. **OpenRouter API key** — sign up at https://openrouter.ai (for LLM synthesis)
4. Project root: `/Users/aankur/workspace/travel-optimizer/`

## Architecture

```
[User Input via Streamlit]
     │
[Supervisor] → generates N candidate windows (rolling 7-day blocks)
     │
     ├── [Weather Agent]  ── Open-Meteo API (free, no key)
     ├── [Flight Agent]   ── Amadeus Flight Offers API
     └── [Hotel Agent]    ── Amadeus Hotel Offers API
                               │
                          (all 3 run in parallel via LangGraph fan-out)
                               │
                     [Scorer] → normalize 0-1, weighted rank
                               │
                     [Synthesizer] → LLM explains top picks
                               │
                     [Streamlit UI] → cards, charts, recommendation
```

**LLM is only used by:** Synthesizer (recommendation text). All other agents are deterministic.

**Fallback:** Every successful API response is saved to SQLite. When APIs fail or quota is exhausted, agents fall back to historical data and flag it in the UI.

## API Reference

| API | Base URL | Auth | Free Tier |
|-----|----------|------|-----------|
| Open-Meteo Geocoding | `geocoding-api.open-meteo.com/v1/search` | None | Unlimited |
| Open-Meteo Climate | `climate-api.open-meteo.com/v1/climate` | None | Unlimited |
| Open-Meteo Historical | `archive-api.open-meteo.com/v1/archive` | None | Unlimited |
| Amadeus Auth | `test.api.amadeus.com/v1/security/oauth2/token` | client_credentials | — |
| Amadeus Flights | `test.api.amadeus.com/v2/shopping/flight-offers` | Bearer token | 500/mo shared |
| Amadeus Hotel List | `test.api.amadeus.com/v1/reference-data/locations/hotels/by-city` | Bearer token | 500/mo shared |
| Amadeus Hotel Offers | `test.api.amadeus.com/v3/shopping/hotel-offers` | Bearer token | 500/mo shared |
| Amadeus IATA Lookup | `test.api.amadeus.com/v1/reference-data/locations` | Bearer token | 500/mo shared |
| OpenRouter | `openrouter.ai/api/v1` | API key | Pay per token |

## File Structure

```
travel-optimizer/
├── app.py                        # Streamlit entry point
├── graph.py                      # LangGraph graph definition (nodes + edges)
├── agents/
│   ├── __init__.py
│   ├── supervisor.py             # Input parsing + candidate window generation
│   ├── weather.py                # Open-Meteo data fetcher + weather scoring
│   ├── flights.py                # Amadeus flights fetcher + SQLite fallback
│   ├── hotels.py                 # Amadeus hotels fetcher + SQLite fallback
│   ├── scorer.py                 # 0-1 normalization + weighted ranking
│   └── synthesizer.py            # LLM recommendation (OpenRouter) + fallback
├── services/
│   ├── __init__.py
│   ├── amadeus_client.py         # Amadeus API wrapper + OAuth token management
│   ├── weather_client.py         # Open-Meteo Climate/Historical API wrapper
│   └── geocoding.py              # City name → lat/lon via Open-Meteo
├── cache.py                      # In-memory TTL cache (24h default)
├── db.py                         # SQLite historical data store + tolerance-based queries
├── models.py                     # Pydantic models for state, API responses
├── config.py                     # Settings loaded from .env via pydantic-settings
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
├── .env.example
├── .env                          # (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## State Model (shared across all agents)

```python
class TravelState(TypedDict, total=False):
    # User input
    destination: str              # "Tokyo, Japan"
    origin: str                   # "Bangalore, India" (default)
    date_range: tuple[str, str]   # ("2026-06-01", "2026-09-30")
    duration_days: int            # 7
    num_travelers: int            # 1
    budget_max: float | None      # optional
    priorities: dict[str, float]  # {"weather": 0.4, "flights": 0.3, "hotels": 0.3}

    # Supervisor output
    candidate_windows: list[dict] # [{"start": "2026-06-01", "end": "2026-06-07"}, ...]

    # Data agent outputs
    weather_data: list[dict]      # [{window, avg_temp, rain_days, avg_humidity, score, is_historical}]
    flight_data: list[dict]       # [{window, min_price, avg_price, currency, score, is_historical}]
    hotel_data: list[dict]        # [{window, avg_nightly, currency, score, is_historical}]

    # Scorer output
    ranked_windows: list[dict]    # [{window, weather_score, flight_score, hotel_score, total_score, ...}]

    # Synthesizer output
    recommendation: str           # natural language summary

    # Errors from any agent
    errors: list[str]
```

## Error Handling Rules

1. If one data agent fails, others still complete. Scorer reweights priorities across available dimensions.
2. Amadeus tokens expire every 30 min — auto-refresh 1 min early.
3. Network: 10s timeout, 3 retries with backoff.
4. Geocoding failure → fail fast with clear error.
5. LLM failure → Synthesizer returns raw ranked data as plain text.
6. Every successful API response → saved to SQLite for future fallback.

## SQLite Schema

```sql
CREATE TABLE flight_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    departure_date TEXT NOT NULL,
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE hotel_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    checkin_date TEXT NOT NULL,
    avg_nightly REAL NOT NULL,
    currency TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Fallback query:** When live API fails, query SQLite for same route + date within 7-day tolerance, ordered by closest date then most recent fetch.

---

# Implementation Tasks

Build in this exact order. Each task is independent from the next and should be committed separately.

---

## Task 1: Project Scaffolding

Create all boilerplate files.

**Files to create:** `.gitignore`, `requirements.txt`, `.env.example`, `config.py`, `models.py`, `agents/__init__.py`, `services/__init__.py`, `tests/__init__.py`

### .gitignore
```
__pycache__/
*.pyc
.env
*.db
.venv/
.streamlit/
```

### requirements.txt
```
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

### .env.example
```
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514
DEFAULT_ORIGIN=Bangalore
DEFAULT_CURRENCY=INR
```

### config.py
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"
    default_origin: str = "Bangalore"
    default_currency: str = "INR"
    db_path: str = "travel_history.db"
    cache_ttl_seconds: int = 86400
    api_timeout_seconds: int = 10
    api_max_retries: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
```

### models.py
```python
from __future__ import annotations
from pydantic import BaseModel
from typing import TypedDict

class CandidateWindow(BaseModel):
    start: str
    end: str

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
    has_historical_data: bool = False

class TravelState(TypedDict, total=False):
    destination: str
    origin: str
    date_range: tuple[str, str]
    duration_days: int
    num_travelers: int
    budget_max: float | None
    priorities: dict[str, float]
    candidate_windows: list[dict]
    weather_data: list[dict]
    flight_data: list[dict]
    hotel_data: list[dict]
    ranked_windows: list[dict]
    recommendation: str
    errors: list[str]
```

### Tests: tests/test_models.py
```python
from models import CandidateWindow, WeatherResult, FlightResult

def test_candidate_window():
    w = CandidateWindow(start="2026-06-01", end="2026-06-07")
    assert w.start == "2026-06-01"

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

### Setup commands
```bash
cd /Users/aankur/workspace/travel-optimizer
git init
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_models.py -v  # should pass
git add -A && git commit -m "feat: project scaffolding with models, config, and deps"
```

---

## Task 2: Geocoding Service

**File:** `services/geocoding.py`

```python
import httpx
from config import settings

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

def geocode_city(city_name: str) -> dict:
    """Returns: {name, latitude, longitude, country, country_code}. Raises ValueError if not found."""
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

### Tests: tests/test_geocoding.py
```python
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
```

```bash
pytest tests/test_geocoding.py -v
git add services/geocoding.py tests/test_geocoding.py && git commit -m "feat: geocoding service using Open-Meteo API"
```

---

## Task 3: Cache + SQLite Store

**Files:** `cache.py`, `db.py`

### cache.py
```python
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

### db.py
```python
import sqlite3

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

    def save_flight(self, origin, destination, departure_date, price, currency):
        self._conn.execute(
            "INSERT INTO flight_prices (origin, destination, departure_date, price, currency) VALUES (?, ?, ?, ?, ?)",
            (origin, destination, departure_date, price, currency))
        self._conn.commit()

    def get_flight(self, origin, destination, departure_date, tolerance_days=0):
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin=? AND destination=? AND ABS(julianday(departure_date)-julianday(?))<=? "
                "ORDER BY ABS(julianday(departure_date)-julianday(?)) ASC, fetched_at DESC LIMIT 1",
                (origin, destination, departure_date, tolerance_days, departure_date)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin=? AND destination=? AND departure_date=? ORDER BY fetched_at DESC LIMIT 1",
                (origin, destination, departure_date)).fetchone()
        return dict(row) if row else None

    def save_hotel(self, city, checkin_date, avg_nightly, currency):
        self._conn.execute(
            "INSERT INTO hotel_prices (city, checkin_date, avg_nightly, currency) VALUES (?, ?, ?, ?)",
            (city, checkin_date, avg_nightly, currency))
        self._conn.commit()

    def get_hotel(self, city, checkin_date, tolerance_days=0):
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city=? AND ABS(julianday(checkin_date)-julianday(?))<=? "
                "ORDER BY ABS(julianday(checkin_date)-julianday(?)) ASC, fetched_at DESC LIMIT 1",
                (city, checkin_date, tolerance_days, checkin_date)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city=? AND checkin_date=? ORDER BY fetched_at DESC LIMIT 1",
                (city, checkin_date)).fetchone()
        return dict(row) if row else None

    def close(self):
        self._conn.close()
```

### Tests: tests/test_cache.py
```python
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

### Tests: tests/test_db.py
```python
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
```

```bash
pytest tests/test_cache.py tests/test_db.py -v
git add cache.py db.py tests/test_cache.py tests/test_db.py && git commit -m "feat: TTL cache and SQLite historical data store"
```

---

## Task 4: Weather Client

**File:** `services/weather_client.py`

```python
import httpx
from datetime import date
from config import settings

CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

def get_weather_for_window(latitude, longitude, start_date, end_date):
    """Returns: {avg_temp, rain_days, avg_humidity}. Uses Climate API for future, Historical for past."""
    is_future = date.fromisoformat(end_date) >= date.today()
    url = CLIMATE_URL if is_future else HISTORICAL_URL
    params = {
        "latitude": latitude, "longitude": longitude,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
    }
    if is_future:
        params["models"] = "EC_Earth3P_HR"
    response = httpx.get(url, params=params, timeout=settings.api_timeout_seconds)
    response.raise_for_status()
    daily = response.json().get("daily", {})
    temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    humidity = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
    return {
        "avg_temp": round(sum(temps)/len(temps), 1) if temps else 0.0,
        "rain_days": sum(1 for p in precip if p > 1.0),
        "avg_humidity": round(sum(humidity)/len(humidity), 1) if humidity else 0.0,
    }
```

### Tests: tests/test_weather_client.py (live — Open-Meteo is free)
```python
import pytest
from services.weather_client import get_weather_for_window

def test_weather_returns_expected_fields():
    result = get_weather_for_window(35.68, 139.69, "2026-07-01", "2026-07-07")
    assert "avg_temp" in result
    assert "rain_days" in result
    assert isinstance(result["avg_temp"], float)

def test_weather_invalid_coords():
    with pytest.raises(Exception):
        get_weather_for_window(999.0, 999.0, "2026-07-01", "2026-07-07")
```

```bash
pytest tests/test_weather_client.py -v
git add services/weather_client.py tests/test_weather_client.py && git commit -m "feat: weather client with climate/historical API selection"
```

---

## Task 5: Amadeus Client

**File:** `services/amadeus_client.py`

```python
import time
import httpx
from config import settings

AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
FLIGHTS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
HOTEL_LIST_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
HOTEL_OFFERS_URL = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
IATA_URL = "https://test.api.amadeus.com/v1/reference-data/locations"

class AmadeusClient:
    def __init__(self, client_id="", client_secret=""):
        self._client_id = client_id or settings.amadeus_client_id
        self._client_secret = client_secret or settings.amadeus_client_secret
        self._token = ""
        self._token_expiry = 0.0

    def _authenticate(self):
        response = httpx.post(AUTH_URL, data={
            "grant_type": "client_credentials",
            "client_id": self._client_id, "client_secret": self._client_secret,
        }, timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data["expires_in"] - 60

    def _ensure_auth(self):
        if time.time() >= self._token_expiry:
            self._authenticate()

    def _headers(self):
        return {"Authorization": f"Bearer {self._token}"}

    def search_flights(self, origin, destination, departure_date, currency="INR", adults=1):
        self._ensure_auth()
        response = httpx.get(FLIGHTS_URL, params={
            "originLocationCode": origin, "destinationLocationCode": destination,
            "departureDate": departure_date, "adults": adults,
            "currencyCode": currency, "max": 5,
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def search_hotels(self, city_code, checkin, checkout, currency="INR", adults=1):
        self._ensure_auth()
        response = httpx.get(HOTEL_LIST_URL, params={"cityCode": city_code},
            headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        hotels = response.json().get("data", [])[:10]
        if not hotels:
            return {"data": []}
        hotel_ids = [h["hotelId"] for h in hotels]
        response = httpx.get(HOTEL_OFFERS_URL, params={
            "hotelIds": ",".join(hotel_ids), "checkInDate": checkin,
            "checkOutDate": checkout, "adults": adults,
            "currency": currency, "bestRateOnly": "true",
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def get_iata_code(self, city_name):
        self._ensure_auth()
        response = httpx.get(IATA_URL, params={
            "keyword": city_name, "subType": "CITY,AIRPORT",
        }, headers=self._headers(), timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data[0].get("iataCode") if data else None
```

### Tests: tests/test_amadeus_client.py (mocked — needs real keys)
```python
from unittest.mock import patch, MagicMock
from services.amadeus_client import AmadeusClient
import pytest

@pytest.fixture
def client():
    return AmadeusClient(client_id="test", client_secret="test")

def test_auth_token_request(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok123", "expires_in": 1799}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.post", return_value=mock_resp):
        client._authenticate()
        assert client._token == "tok123"

def test_search_flights_params(client):
    client._token = "tok123"
    client._token_expiry = 9999999999.0
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"price": {"total": "15000"}}]}
    mock_resp.raise_for_status = MagicMock()
    with patch("httpx.get", return_value=mock_resp) as mock_get:
        client.search_flights("BLR", "NRT", "2026-07-01", "INR")
        params = mock_get.call_args[1]["params"]
        assert params["originLocationCode"] == "BLR"
        assert params["currencyCode"] == "INR"
```

```bash
pytest tests/test_amadeus_client.py -v
git add services/amadeus_client.py tests/test_amadeus_client.py && git commit -m "feat: Amadeus API client with auth, flights, hotels, IATA lookup"
```

---

## Task 6: Supervisor Agent

**File:** `agents/supervisor.py`

```python
from datetime import date, timedelta
from models import TravelState

def generate_candidate_windows(start_date, end_date, duration_days=7):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows = []
    current = start
    while current + timedelta(days=duration_days) <= end:
        windows.append({
            "start": current.isoformat(),
            "end": (current + timedelta(days=duration_days)).isoformat(),
        })
        current += timedelta(days=7)
    return windows

def supervisor_node(state: TravelState) -> dict:
    windows = generate_candidate_windows(
        state["date_range"][0], state["date_range"][1],
        state.get("duration_days", 7))
    return {"candidate_windows": windows, "errors": state.get("errors", [])}
```

### Tests: tests/test_supervisor.py
```python
from agents.supervisor import generate_candidate_windows
from datetime import date

def test_generates_correct_number():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    assert 3 <= len(windows) <= 5

def test_window_duration():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    for w in windows:
        d = (date.fromisoformat(w["end"]) - date.fromisoformat(w["start"])).days
        assert d == 7

def test_windows_within_range():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    for w in windows:
        assert w["start"] >= "2026-06-01"
        assert w["end"] <= "2026-06-30"

def test_wide_range():
    windows = generate_candidate_windows("2026-06-01", "2026-09-30", 7)
    assert len(windows) >= 10
```

```bash
pytest tests/test_supervisor.py -v
git add agents/supervisor.py tests/test_supervisor.py && git commit -m "feat: supervisor agent with candidate window generation"
```

---

## Task 7: Weather Agent

**File:** `agents/weather.py`

```python
from models import TravelState
from services.geocoding import geocode_city
from services.weather_client import get_weather_for_window

def score_weather(avg_temp, rain_days, avg_humidity,
                  ideal_temp_min=20.0, ideal_temp_max=28.0):
    """Score 0-1. Higher = better weather."""
    # Temperature
    if ideal_temp_min <= avg_temp <= ideal_temp_max:
        temp_score = 1.0
    else:
        distance = min(abs(avg_temp - ideal_temp_min), abs(avg_temp - ideal_temp_max))
        temp_score = max(0.0, 1.0 - distance / 20.0)
    # Rain: 0 days = 1.0, 7 days = 0.0
    rain_score = max(0.0, 1.0 - rain_days / 7.0)
    # Humidity
    if avg_humidity <= 60:
        humidity_score = 1.0
    elif avg_humidity <= 80:
        humidity_score = 1.0 - (avg_humidity - 60) / 40.0
    else:
        humidity_score = max(0.0, 0.5 - (avg_humidity - 80) / 40.0)
    return round(temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25, 3)

def weather_node(state: TravelState) -> dict:
    errors = list(state.get("errors", []))
    try:
        geo = geocode_city(state["destination"])
    except ValueError as e:
        errors.append(f"Weather: geocoding failed — {e}")
        return {"weather_data": [], "errors": errors}
    results = []
    for window in state["candidate_windows"]:
        try:
            data = get_weather_for_window(geo["latitude"], geo["longitude"],
                                          window["start"], window["end"])
            score = score_weather(data["avg_temp"], data["rain_days"], data["avg_humidity"])
            results.append({
                "window": window, "avg_temp": data["avg_temp"],
                "rain_days": data["rain_days"], "avg_humidity": data["avg_humidity"],
                "score": score, "is_historical": False,
            })
        except Exception as e:
            errors.append(f"Weather: failed for {window['start']} — {e}")
    return {"weather_data": results, "errors": errors}
```

### Tests: tests/test_weather_agent.py
```python
from unittest.mock import patch
from agents.weather import weather_node, score_weather

def test_score_ideal():
    assert score_weather(24.0, 0, 50.0) > 0.8

def test_score_bad():
    assert score_weather(40.0, 6, 90.0) < 0.3

def test_score_range():
    s = score_weather(24.0, 2, 60.0)
    assert 0.0 <= s <= 1.0

def test_weather_node_populates_state():
    state = {"destination": "Tokyo", "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}), \
         patch("agents.weather.get_weather_for_window", return_value={"avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0}):
        result = weather_node(state)
    assert len(result["weather_data"]) == 1
    assert 0.0 <= result["weather_data"][0]["score"] <= 1.0
```

```bash
pytest tests/test_weather_agent.py -v
git add agents/weather.py tests/test_weather_agent.py && git commit -m "feat: weather agent with scoring"
```

---

## Task 8: Flights Agent

**File:** `agents/flights.py`

```python
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
```

### Tests: tests/test_flights_agent.py
```python
from unittest.mock import patch, MagicMock
from agents.flights import flights_node, parse_flight_prices

def test_parse_prices():
    resp = {"data": [{"price": {"total": "15000.00"}}, {"price": {"total": "18000.00"}}, {"price": {"total": "20000.00"}}]}
    result = parse_flight_prices(resp)
    assert result["min_price"] == 15000.0
    assert result["avg_price"] == 17666.67

def test_parse_empty():
    assert parse_flight_prices({"data": []}) is None

def test_flights_node_mock():
    state = {"origin": "Bangalore", "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "num_travelers": 1, "errors": []}
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = {"data": [{"price": {"total": "15000.00"}}]}
    with patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.flights.HistoryDB"):
        result = flights_node(state)
    assert len(result["flight_data"]) == 1
```

```bash
pytest tests/test_flights_agent.py -v
git add agents/flights.py tests/test_flights_agent.py && git commit -m "feat: flights agent with Amadeus + historical fallback"
```

---

## Task 9: Hotels Agent

**File:** `agents/hotels.py`

```python
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
```

### Tests: tests/test_hotels_agent.py
```python
from unittest.mock import patch, MagicMock
from agents.hotels import hotels_node, parse_hotel_prices

def test_parse_prices():
    resp = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}, {"offers": [{"price": {"total": "12000.00"}}]}]}
    assert parse_hotel_prices(resp)["avg_nightly"] == 10000.0

def test_parse_empty():
    assert parse_hotel_prices({"data": []}) is None

def test_hotels_node_mock():
    state = {"destination": "Tokyo", "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    mock_client = MagicMock()
    mock_client.get_iata_code.return_value = "TYO"
    mock_client.search_hotels.return_value = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}]}
    with patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.hotels.HistoryDB"):
        result = hotels_node(state)
    assert len(result["hotel_data"]) == 1
```

```bash
pytest tests/test_hotels_agent.py -v
git add agents/hotels.py tests/test_hotels_agent.py && git commit -m "feat: hotels agent with Amadeus + historical fallback"
```

---

## Task 10: Scorer Agent

**File:** `agents/scorer.py`

```python
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

    weather_by_start = {d["window"]["start"]: d for d in weather_data}
    flight_by_start = {d["window"]["start"]: d for d in flight_data}
    hotel_by_start = {d["window"]["start"]: d for d in hotel_data}

    all_starts = sorted(set(
        [d["window"]["start"] for d in weather_data] +
        [d["window"]["start"] for d in flight_data] +
        [d["window"]["start"] for d in hotel_data]))

    # Normalize prices (lower = better)
    flight_starts = [s for s in all_starts if s in flight_by_start]
    hotel_starts = [s for s in all_starts if s in hotel_by_start]
    flight_norm = dict(zip(flight_starts,
        normalize_scores([flight_by_start[s]["min_price"] for s in flight_starts], lower_is_better=True))) if flight_starts else {}
    hotel_norm = dict(zip(hotel_starts,
        normalize_scores([hotel_by_start[s]["avg_nightly"] for s in hotel_starts], lower_is_better=True))) if hotel_starts else {}

    # Reweight if dimension missing
    active = {}
    if weather_data: active["weather"] = priorities.get("weather", 0.4)
    if flight_data: active["flights"] = priorities.get("flights", 0.3)
    if hotel_data: active["hotels"] = priorities.get("hotels", 0.3)
    total_w = sum(active.values()) or 1.0
    norm_w = {k: v/total_w for k, v in active.items()}

    ranked = []
    for start in all_starts:
        ws = weather_by_start.get(start, {}).get("score", 0.0)
        fs = flight_norm.get(start, 0.0)
        hs = hotel_norm.get(start, 0.0)
        total = ws * norm_w.get("weather", 0) + fs * norm_w.get("flights", 0) + hs * norm_w.get("hotels", 0)
        window = (weather_by_start.get(start) or flight_by_start.get(start) or hotel_by_start.get(start, {}))
        ranked.append({
            "window": window.get("window", {"start": start, "end": ""}),
            "weather_score": round(ws, 3), "flight_score": round(fs, 3),
            "hotel_score": round(hs, 3), "total_score": round(total, 3),
            "estimated_flight_cost": flight_by_start.get(start, {}).get("min_price", 0.0),
            "estimated_hotel_cost": hotel_by_start.get(start, {}).get("avg_nightly", 0.0),
            "has_historical_data": any([
                weather_by_start.get(start, {}).get("is_historical", False),
                flight_by_start.get(start, {}).get("is_historical", False),
                hotel_by_start.get(start, {}).get("is_historical", False)]),
        })
    ranked.sort(key=lambda x: x["total_score"], reverse=True)
    return {"ranked_windows": ranked, "errors": state.get("errors", [])}
```

### Tests: tests/test_scorer.py
```python
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
```

```bash
pytest tests/test_scorer.py -v
git add agents/scorer.py tests/test_scorer.py && git commit -m "feat: scorer agent with normalization and priority reweighting"
```

---

## Task 11: Synthesizer Agent

**File:** `agents/synthesizer.py`

```python
from langchain_openai import ChatOpenAI
from models import TravelState
from config import settings

_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=settings.openrouter_model,
            api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    return _llm

def format_ranked_data_fallback(ranked, top_n=3):
    lines = []
    for i, r in enumerate(ranked[:top_n], 1):
        w = r["window"]
        lines.append(f"#{i}: {w['start']} to {w['end']} (score: {r['total_score']:.2f}) — "
            f"Weather: {r['weather_score']:.2f}, Flights: ~{r['estimated_flight_cost']:.0f}, "
            f"Hotels: ~{r['estimated_hotel_cost']:.0f}/night"
            + (" [estimated from history]" if r.get("has_historical_data") else ""))
    return "\n".join(lines)

def synthesizer_node(state: TravelState) -> dict:
    ranked = state.get("ranked_windows", [])
    errors = list(state.get("errors", []))
    if not ranked:
        return {"recommendation": "No data available.", "errors": errors}
    data_summary = format_ranked_data_fallback(ranked[:5], top_n=5)
    prompt = (f"You are a travel advisor. Based on these ranked travel windows for a trip from "
        f"{state.get('origin', 'your city')} to {state.get('destination', 'the destination')}, "
        f"write a concise recommendation (3-5 sentences) about which dates are best and why. "
        f"Mention weather, flight cost, and hotel cost. If data is estimated from history, note that.\n\n{data_summary}")
    try:
        response = _get_llm().invoke(prompt)
        return {"recommendation": response.content, "errors": errors}
    except Exception as e:
        errors.append(f"Synthesizer: LLM failed — {e}")
        return {"recommendation": data_summary, "errors": errors}
```

### Tests: tests/test_synthesizer.py
```python
from unittest.mock import patch, MagicMock
from agents.synthesizer import synthesizer_node, format_ranked_data_fallback

SAMPLE_RANKED = [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "total_score": 0.85,
    "weather_score": 0.9, "flight_score": 0.8, "hotel_score": 0.75,
    "estimated_flight_cost": 15000, "estimated_hotel_cost": 8000, "has_historical_data": False}]

def test_fallback_format():
    result = format_ranked_data_fallback(SAMPLE_RANKED)
    assert "2026-07-01" in result

def test_synthesizer_uses_llm():
    state = {"destination": "Tokyo", "origin": "Bangalore", "ranked_windows": SAMPLE_RANKED, "errors": []}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July is great.")
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    assert "July" in result["recommendation"]

def test_synthesizer_fallback():
    state = {"destination": "Tokyo", "origin": "Bangalore", "ranked_windows": SAMPLE_RANKED, "errors": []}
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    assert "2026-07-01" in result["recommendation"]
```

```bash
pytest tests/test_synthesizer.py -v
git add agents/synthesizer.py tests/test_synthesizer.py && git commit -m "feat: synthesizer agent with LLM + fallback"
```

---

## Task 12: LangGraph Wiring

**File:** `graph.py`

```python
from langgraph.graph import StateGraph, END
from models import TravelState
from agents.supervisor import supervisor_node
from agents.weather import weather_node
from agents.flights import flights_node
from agents.hotels import hotels_node
from agents.scorer import scorer_node
from agents.synthesizer import synthesizer_node

def build_graph():
    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather", weather_node)
    graph.add_node("flights", flights_node)
    graph.add_node("hotels", hotels_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("supervisor")
    # Fan-out: supervisor → 3 data agents in parallel
    graph.add_edge("supervisor", "weather")
    graph.add_edge("supervisor", "flights")
    graph.add_edge("supervisor", "hotels")
    # Fan-in: all 3 → scorer
    graph.add_edge("weather", "scorer")
    graph.add_edge("flights", "scorer")
    graph.add_edge("hotels", "scorer")
    # Sequential: scorer → synthesizer → end
    graph.add_edge("scorer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
```

### Tests: tests/test_graph.py
```python
from unittest.mock import patch, MagicMock
from graph import build_graph

def test_graph_builds():
    assert build_graph() is not None

def test_graph_e2e_mocked():
    mock_client = MagicMock()
    mock_client.get_iata_code.side_effect = lambda x: "BLR" if "Bangalore" in x else "NRT"
    mock_client.search_flights.return_value = {"data": [{"price": {"total": "15000.00"}}]}
    mock_client.search_hotels.return_value = {"data": [{"offers": [{"price": {"total": "8000.00"}}]}]}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="July is great for Tokyo.")

    with patch("agents.weather.geocode_city", return_value={"latitude": 35.68, "longitude": 139.69}), \
         patch("agents.weather.get_weather_for_window", return_value={"avg_temp": 25.0, "rain_days": 1, "avg_humidity": 55.0}), \
         patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.synthesizer._get_llm", return_value=mock_llm), \
         patch("agents.flights.HistoryDB"), patch("agents.hotels.HistoryDB"):
        result = build_graph().invoke({
            "destination": "Tokyo", "origin": "Bangalore",
            "date_range": ("2026-07-01", "2026-07-28"), "duration_days": 7,
            "num_travelers": 1, "budget_max": None,
            "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3}, "errors": []})
    assert len(result["ranked_windows"]) > 0
    assert len(result["recommendation"]) > 0
```

```bash
pytest tests/test_graph.py -v
git add graph.py tests/test_graph.py && git commit -m "feat: LangGraph with parallel fan-out"
```

---

## Task 13: Streamlit UI

**File:** `app.py`

```python
import streamlit as st
import pandas as pd
from graph import build_graph
from config import settings

st.set_page_config(page_title="Travel Optimizer", layout="wide")
st.title("Travel Optimizer")
st.caption("Find the best time to visit any destination")

with st.sidebar:
    st.header("Trip Details")
    destination = st.text_input("Destination", placeholder="Tokyo, Japan")
    origin = st.text_input("Origin", value=settings.default_origin)
    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("From")
    with col2: end_date = st.date_input("To")
    duration = st.slider("Trip duration (days)", 3, 21, 7)
    travelers = st.number_input("Travelers", 1, 10, 1)
    budget = st.number_input("Budget ceiling (0 = no limit)", 0, 1000000, 0)
    st.header("Priorities")
    w_weather = st.slider("Weather", 0.0, 1.0, 0.4, 0.05)
    w_flights = st.slider("Flight cost", 0.0, 1.0, 0.3, 0.05)
    w_hotels = st.slider("Hotel cost", 0.0, 1.0, 0.3, 0.05)
    search = st.button("Find Best Time", type="primary", use_container_width=True)

if search:
    if not destination:
        st.error("Enter a destination.")
    else:
        total_w = w_weather + w_flights + w_hotels or 1.0
        state = {
            "destination": destination, "origin": origin,
            "date_range": (start_date.isoformat(), end_date.isoformat()),
            "duration_days": duration, "num_travelers": travelers,
            "budget_max": budget if budget > 0 else None,
            "priorities": {"weather": w_weather/total_w, "flights": w_flights/total_w, "hotels": w_hotels/total_w},
            "errors": []}
        with st.spinner("Searching..."):
            result = build_graph().invoke(state)
        if result.get("errors"):
            with st.expander("Warnings", expanded=False):
                for e in result["errors"]: st.warning(e)
        ranked = result.get("ranked_windows", [])
        if not ranked:
            st.error("No results. Try wider dates or different destination.")
        else:
            st.header("Recommendation")
            st.write(result.get("recommendation", ""))
            st.header("Top Windows")
            cols = st.columns(min(3, len(ranked)))
            for i, r in enumerate(ranked[:3]):
                with cols[i]:
                    w = r["window"]
                    st.metric(f"#{i+1}: {w['start']} → {w['end']}", f"Score: {r['total_score']:.2f}")
                    st.caption(f"Weather: {r['weather_score']:.2f} | Flight: ~{r['estimated_flight_cost']:,.0f} | Hotel: ~{r['estimated_hotel_cost']:,.0f}/night"
                        + (" (estimated)" if r.get("has_historical_data") else ""))
            st.header("Comparison")
            df = pd.DataFrame([{"Window": r["window"]["start"], "Weather": r["weather_score"],
                "Flights": r["flight_score"], "Hotels": r["hotel_score"], "Total": r["total_score"],
                "Flight Cost": r["estimated_flight_cost"], "Hotel/Night": r["estimated_hotel_cost"]} for r in ranked]).set_index("Window")
            t1, t2, t3 = st.tabs(["Scores", "Flight Prices", "Hotel Prices"])
            with t1: st.bar_chart(df[["Weather", "Flights", "Hotels", "Total"]])
            with t2: st.line_chart(df["Flight Cost"])
            with t3: st.line_chart(df["Hotel/Night"])
```

```bash
streamlit run app.py  # manual test
git add app.py && git commit -m "feat: Streamlit UI"
```

---

## Task 14: README

**File:** `README.md`

```markdown
# Travel Optimizer

Find the best time to visit any destination — based on weather, flights, and hotels.

Built with LangGraph, Streamlit, Amadeus API, and Open-Meteo.

## Setup

\```bash
cd travel-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
\```

## Run

\```bash
streamlit run app.py
\```

## Test

\```bash
pytest tests/ -v
\```

## Architecture

\```
User → Supervisor → [Weather | Flights | Hotels] parallel → Scorer → Synthesizer → UI
\```
```

```bash
pytest tests/ -v  # full suite
git add README.md && git commit -m "docs: README"
```

---

## Next Steps (after MVP)

1. **Add Kiwi/Tequila as secondary flight source** — natural "add another agent" exercise
2. **Price trend charts** — query SQLite history to show how prices change over months
3. **Budget filtering** — filter out windows where total cost exceeds budget ceiling
4. **Multiple destinations** — compare "Tokyo vs Bangkok vs Lisbon" side by side
5. **Deployment** — Streamlit Cloud (free) for personal access from anywhere
6. **Async agents** — convert to async httpx calls for true parallelism (LangGraph supports async nodes)
