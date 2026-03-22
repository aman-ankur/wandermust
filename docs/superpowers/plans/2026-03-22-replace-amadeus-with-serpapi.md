# Replace Amadeus with SerpApi (Google Flights + Google Hotels) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the defunct Amadeus API with SerpApi for both flight search (Google Flights engine) and hotel search (Google Hotels engine).

**Architecture:** One new service client (`services/serpapi_client.py`) replaces `AmadeusClient`, exposing `search_flights()` and `search_hotels()`. The flight/hotel agents get updated imports and parsers. No IATA lookup needed — SerpApi accepts IATA codes or city names directly. Hotels use a text query (`q` param) instead of `dest_id`. Single API key for everything.

**Tech Stack:** `serpapi` Python package (`google-search-results`), existing Pydantic settings, existing SQLite fallback

**Context:** Amadeus self-service developer portal no longer accepts registrations and will be decommissioned July 17, 2026. Kiwi.com Tequila API is now invitation-only. SerpApi provides Google Flights + Google Hotels via a single API key with 100 free searches/month.

---

## API Reference

### SerpApi Google Flights

**Endpoint:** `engine=google_flights`

**Key params:** `departure_id` (IATA or city), `arrival_id`, `outbound_date` (YYYY-MM-DD), `currency`, `adults`, `type` (2=one-way)

**Response shape:**
```json
{
  "best_flights": [{"price": 267, "flights": [...], "total_duration": 455}],
  "other_flights": [{"price": 278, "flights": [...]}],
  "price_insights": {"lowest_price": 267}
}
```

**We extract:** `best_flights[].price` + `other_flights[].price` → min_price, avg_price

### SerpApi Google Hotels

**Endpoint:** `engine=google_hotels`

**Key params:** `q` (search query like "Hotels in Tokyo"), `check_in_date`, `check_out_date` (YYYY-MM-DD), `currency`, `adults`

**Response shape:**
```json
{
  "properties": [
    {
      "name": "Hotel Name",
      "rate_per_night": {"lowest": "$150", "extracted_lowest": 150},
      "total_rate": {"lowest": "$750", "extracted_lowest": 750},
      "overall_rating": 4.6,
      "hotel_class": "5-star hotel"
    }
  ]
}
```

**We extract:** `properties[].rate_per_night.extracted_lowest` → avg_nightly

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `services/serpapi_client.py` | Create | SerpApi client for flights + hotels |
| `services/amadeus_client.py` | Delete | No longer needed |
| `agents/flights.py` | Modify | Use SerpApi client, new parser |
| `agents/hotels.py` | Modify | Use SerpApi client, new parser, no IATA lookup |
| `config.py` | Modify | Add `serpapi_api_key`, remove `amadeus_*` |
| `.env.example` | Modify | Replace Amadeus vars with `SERPAPI_API_KEY` |
| `requirements.txt` | Modify | Add `google-search-results` |
| `tests/test_serpapi_client.py` | Create | Unit tests for SerpApi client |
| `tests/test_flights_agent.py` | Modify | Update mocks for new client/response shape |
| `tests/test_hotels_agent.py` | Modify | Update mocks for new client/response shape |
| `tests/test_amadeus_client.py` | Delete | No longer needed |

---

### Task 1: Create SerpApi Client

**Files:**
- Create: `services/serpapi_client.py`
- Test: `tests/test_serpapi_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_serpapi_client.py
from unittest.mock import patch, MagicMock
from services.serpapi_client import SerpApiClient

def test_search_flights():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "best_flights": [{"price": 267}, {"price": 324}],
        "other_flights": [{"price": 278}],
    }
    with patch("services.serpapi_client.serpapi.Client", return_value=mock_client):
        client = SerpApiClient(api_key="test")
        result = client.search_flights("BLR", "DEL", "2026-07-01", currency="USD", adults=2)
    assert result == {"best_flights": [{"price": 267}, {"price": 324}], "other_flights": [{"price": 278}]}
    call_args = mock_client.search.call_args[0][0]
    assert call_args["engine"] == "google_flights"
    assert call_args["departure_id"] == "BLR"
    assert call_args["adults"] == 2

def test_search_flights_empty():
    mock_client = MagicMock()
    mock_client.search.return_value = {"best_flights": [], "other_flights": []}
    with patch("services.serpapi_client.serpapi.Client", return_value=mock_client):
        client = SerpApiClient(api_key="test")
        result = client.search_flights("BLR", "DEL", "2026-07-01")
    assert result == {"best_flights": [], "other_flights": []}

def test_search_hotels():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "properties": [
            {"name": "Hotel A", "rate_per_night": {"extracted_lowest": 150}},
            {"name": "Hotel B", "rate_per_night": {"extracted_lowest": 200}},
        ]
    }
    with patch("services.serpapi_client.serpapi.Client", return_value=mock_client):
        client = SerpApiClient(api_key="test")
        result = client.search_hotels("Tokyo", "2026-07-01", "2026-07-07", currency="USD")
    assert len(result["properties"]) == 2
    call_args = mock_client.search.call_args[0][0]
    assert call_args["engine"] == "google_hotels"
    assert call_args["q"] == "Hotels in Tokyo"

def test_search_hotels_empty():
    mock_client = MagicMock()
    mock_client.search.return_value = {"properties": []}
    with patch("services.serpapi_client.serpapi.Client", return_value=mock_client):
        client = SerpApiClient(api_key="test")
        result = client.search_hotels("Tokyo", "2026-07-01", "2026-07-07")
    assert result == {"properties": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_serpapi_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.serpapi_client'`

- [ ] **Step 3: Write implementation**

```python
# services/serpapi_client.py
import serpapi

class SerpApiClient:
    def __init__(self, api_key=""):
        self._api_key = api_key

    def _client(self):
        return serpapi.Client(api_key=self._api_key)

    def search_flights(self, origin, destination, departure_date, currency="INR", adults=1):
        results = self._client().search({
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id": destination,
            "outbound_date": departure_date,
            "currency": currency,
            "adults": adults,
            "type": "2",  # one-way
        })
        return {
            "best_flights": results.get("best_flights", []),
            "other_flights": results.get("other_flights", []),
        }

    def search_hotels(self, city_name, checkin, checkout, currency="INR", adults=1):
        results = self._client().search({
            "engine": "google_hotels",
            "q": f"Hotels in {city_name}",
            "check_in_date": checkin,
            "check_out_date": checkout,
            "currency": currency,
            "adults": adults,
        })
        return {
            "properties": results.get("properties", []),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_serpapi_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/serpapi_client.py tests/test_serpapi_client.py
git commit -m "feat: add SerpApi client for Google Flights + Hotels"
```

---

### Task 2: Update Config and Dependencies

**Files:**
- Modify: `config.py:3-5`
- Modify: `.env.example:10-12`
- Modify: `requirements.txt`

- [ ] **Step 1: Update config.py — replace amadeus fields with serpapi**

In `config.py`, remove:
```python
    amadeus_client_id: str = ""
    amadeus_client_secret: str = ""
```
Add in their place:
```python
    serpapi_api_key: str = ""
```

- [ ] **Step 2: Update .env.example**

Replace:
```
# Amadeus API (https://developers.amadeus.com) — for flights + hotels
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret
```
With:
```
# SerpApi (https://serpapi.com) — for flights + hotels via Google
SERPAPI_API_KEY=your_serpapi_key
```

- [ ] **Step 3: Add serpapi to requirements.txt**

Add line: `google-search-results`

- [ ] **Step 4: Install and verify**

Run: `pip install google-search-results && python -c "import serpapi; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add config.py .env.example requirements.txt
git commit -m "feat: update config for SerpApi, remove Amadeus settings"
```

---

### Task 3: Update Flight Agent

**Files:**
- Modify: `agents/flights.py` (full rewrite)
- Modify: `tests/test_flights_agent.py` (full rewrite)

- [ ] **Step 1: Write updated tests**

```python
# tests/test_flights_agent.py
from unittest.mock import patch, MagicMock
from agents.flights import flights_node, parse_flight_prices

def test_parse_prices():
    resp = {"best_flights": [{"price": 267}, {"price": 324}], "other_flights": [{"price": 278}]}
    result = parse_flight_prices(resp)
    assert result["min_price"] == 267
    assert result["avg_price"] == 289.67

def test_parse_empty():
    assert parse_flight_prices({"best_flights": [], "other_flights": []}) is None

def test_flights_node_mock():
    state = {"origin": "Bangalore", "destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}],
        "num_travelers": 1, "errors": []}
    mock_client = MagicMock()
    mock_client.search_flights.return_value = {
        "best_flights": [{"price": 15000}], "other_flights": []}
    with patch("agents.flights._get_client", return_value=mock_client), \
         patch("agents.flights.HistoryDB"):
        result = flights_node(state)
    assert len(result["flight_data"]) == 1
    assert result["flight_data"][0]["min_price"] == 15000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_flights_agent.py -v`
Expected: FAIL (old parse format, old import)

- [ ] **Step 3: Rewrite flights.py**

```python
# agents/flights.py
from models import TravelState
from services.serpapi_client import SerpApiClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = SerpApiClient(api_key=settings.serpapi_api_key)
    return _client

def parse_flight_prices(api_response):
    all_flights = api_response.get("best_flights", []) + api_response.get("other_flights", [])
    if not all_flights: return None
    prices = [f["price"] for f in all_flights if "price" in f]
    if not prices: return None
    return {"min_price": min(prices), "avg_price": round(sum(prices)/len(prices), 2)}

def flights_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    origin = state["origin"]
    destination = state["destination"]

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_flights(origin, destination, window["start"],
                                             currency=currency, adults=state.get("num_travelers", 1))
            parsed = parse_flight_prices(response)
            if parsed:
                db.save_flight(origin, destination, window["start"], parsed["min_price"], currency)
                results.append({"window": window, "min_price": parsed["min_price"],
                    "avg_price": parsed["avg_price"], "currency": currency, "score": 0.0, "is_historical": False})
            else:
                hist = db.get_flight(origin, destination, window["start"], tolerance_days=7)
                if hist:
                    results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                        "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
        except Exception as e:
            hist = db.get_flight(origin, destination, window["start"], tolerance_days=7)
            if hist:
                results.append({"window": window, "min_price": hist["price"], "avg_price": hist["price"],
                    "currency": hist["currency"], "score": 0.0, "is_historical": True, "fetched_at": hist["fetched_at"]})
            else:
                errors.append(f"Flights: failed for {window['start']} — {e}")
    return {"flight_data": results, "errors": errors}
```

**Note:** SerpApi accepts both IATA codes and city names for `departure_id`/`arrival_id`, so no separate IATA lookup needed. DB now stores whatever the user typed (city name) instead of IATA codes. Old historical rows with IATA codes become unreachable — accepted trade-off.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_flights_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/flights.py tests/test_flights_agent.py
git commit -m "feat: switch flight agent from Amadeus to SerpApi Google Flights"
```

---

### Task 4: Update Hotel Agent

**Files:**
- Modify: `agents/hotels.py` (full rewrite)
- Modify: `tests/test_hotels_agent.py` (full rewrite)

- [ ] **Step 1: Write updated tests**

```python
# tests/test_hotels_agent.py
from unittest.mock import patch, MagicMock
from agents.hotels import hotels_node, parse_hotel_prices

def test_parse_prices():
    resp = {"properties": [
        {"rate_per_night": {"extracted_lowest": 150}},
        {"rate_per_night": {"extracted_lowest": 200}},
    ]}
    result = parse_hotel_prices(resp)
    assert result["avg_nightly"] == 175.0

def test_parse_empty():
    assert parse_hotel_prices({"properties": []}) is None

def test_parse_missing_rate():
    resp = {"properties": [
        {"rate_per_night": {"extracted_lowest": 100}},
        {"name": "No rate hotel"},  # missing rate_per_night
    ]}
    result = parse_hotel_prices(resp)
    assert result["avg_nightly"] == 100.0

def test_hotels_node_mock():
    state = {"destination": "Tokyo",
        "candidate_windows": [{"start": "2026-07-01", "end": "2026-07-07"}], "errors": []}
    mock_client = MagicMock()
    mock_client.search_hotels.return_value = {
        "properties": [{"rate_per_night": {"extracted_lowest": 8000}}]}
    with patch("agents.hotels._get_client", return_value=mock_client), \
         patch("agents.hotels.HistoryDB"):
        result = hotels_node(state)
    assert len(result["hotel_data"]) == 1
    assert result["hotel_data"][0]["avg_nightly"] == 8000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hotels_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite hotels.py**

```python
# agents/hotels.py
from models import TravelState
from services.serpapi_client import SerpApiClient
from db import HistoryDB
from config import settings

_client = None
def _get_client():
    global _client
    if _client is None: _client = SerpApiClient(api_key=settings.serpapi_api_key)
    return _client

def parse_hotel_prices(api_response):
    properties = api_response.get("properties", [])
    if not properties: return None
    prices = []
    for p in properties:
        try:
            prices.append(float(p["rate_per_night"]["extracted_lowest"]))
        except (KeyError, TypeError):
            continue
    if not prices: return None
    return {"avg_nightly": round(sum(prices)/len(prices), 2)}

def hotels_node(state: TravelState) -> dict:
    client = _get_client()
    errors = list(state.get("errors", []))
    currency = settings.default_currency
    destination = state["destination"]

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            response = client.search_hotels(destination, window["start"], window["end"], currency=currency)
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hotels_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/hotels.py tests/test_hotels_agent.py
git commit -m "feat: switch hotel agent from Amadeus to SerpApi Google Hotels"
```

---

### Task 5: Delete Amadeus Client, Update Docs

**Files:**
- Delete: `services/amadeus_client.py`
- Delete: `tests/test_amadeus_client.py`
- Modify: `README.md`
- Modify: `docs/BUILD_GUIDE.md` (if exists — update Amadeus references)
- Modify: `docs/learning/project-walkthrough/03-data-agents.md` (if exists — update Amadeus references)
- Modify: `docs/learning/project-walkthrough/05-end-to-end-flow.md` (if exists — update Amadeus references)

**Note on historical DB data:** Old `flight_prices` rows store IATA codes (e.g., "BLR", "NRT") while new rows store city names (e.g., "Bangalore", "Tokyo"). Old historical fallback data becomes unreachable — accepted trade-off. Fresh data accumulates after migration.

- [ ] **Step 1: Delete Amadeus files**

```bash
git rm services/amadeus_client.py tests/test_amadeus_client.py
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS — no remaining Amadeus imports

- [ ] **Step 3: Update README.md**

Key changes:
- Prerequisites: replace "Amadeus API credentials" → "SerpApi API key (https://serpapi.com) — 100 free searches/month"
- Tech Stack: "Amadeus API — Flight and hotel data" → "SerpApi — Flight + hotel search via Google Flights & Google Hotels"
- Agent table: Flights source → "SerpApi Google Flights", Hotels source → "SerpApi Google Hotels"
- Config `.env` section: replace `AMADEUS_CLIENT_ID`/`AMADEUS_CLIENT_SECRET` → `SERPAPI_API_KEY`
- Roadmap: remove "Add multiple flight data sources (Kiwi, Skyscanner)" — Kiwi is also invite-only now

- [ ] **Step 4: Update other docs with Amadeus references**

Search for "amadeus" (case-insensitive) in `docs/` and update references to describe SerpApi instead.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove Amadeus client, update docs for SerpApi"
```

---

### Task 6: Full Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Test demo mode**

Run: `streamlit run app.py`
- Enable demo mode toggle
- Run a search — should use mock data, no API calls
Expected: Results display correctly (mock agents are independent of API client)

- [ ] **Step 3: Test with real API key**

Set in `.env`:
```
SERPAPI_API_KEY=<real key>
```
Run: `streamlit run app.py`
- Disable demo mode
- Search: Bangalore → Tokyo, July 2026
Expected: Real flight prices and hotel rates returned

- [ ] **Step 4: Test SQLite fallback**

Clear `SERPAPI_API_KEY` from `.env`, run again.
Expected: If previous data exists, historical data served with `is_historical: True`. Otherwise, error messages in results.
