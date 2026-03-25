# Data Agents — Tools in Action

Our three data agents (weather, flights, hotels) are the workhorses of the system. They don't use LLMs — they're pure Python functions that call APIs and return structured data. In agent terminology, they ARE the tools.

This chapter opens up each agent, explains the shared patterns, and shows why deterministic code beats LLM reasoning for data fetching. If you have not read [Foundations Chapter 02 — Tools](../foundations/02-tools-giving-llms-hands.md) and [Chapter 07 — Reliability](../foundations/07-reliability-and-production.md), those provide the conceptual foundation for what follows.

---

## The Pattern All Three Share

Before looking at each agent individually, let's extract the pattern they all follow. Every data agent in our system has the same five-step structure:

```python
def data_agent_node(state: TravelState) -> dict:
    errors = list(state.get("errors", []))
    try:
        # 1. Resolve identifiers (geocode city, look up IATA codes)
        identifier = resolve(state["destination"])
    except Exception as e:
        errors.append(f"Agent: resolution failed — {e}")
        return {"my_data": [], "errors": errors}

    db = HistoryDB(settings.db_path)
    results = []
    for window in state["candidate_windows"]:
        try:
            # 2. Call external API
            raw = api_client.fetch(identifier, window)
            parsed = parse_response(raw)
            if parsed:
                # 3. Save to SQLite for future fallback
                db.save(parsed)
                results.append(parsed)
            else:
                # 4. No data from API — try historical fallback
                hist = db.get_similar(identifier, window, tolerance_days=7)
                if hist:
                    results.append({**hist, "is_historical": True})
        except Exception as e:
            # 5. API error — try historical fallback
            hist = db.get_similar(identifier, window, tolerance_days=7)
            if hist:
                results.append({**hist, "is_historical": True})
            else:
                errors.append(f"Agent: failed for {window['start']} — {e}")
    return {"my_data": results, "errors": errors}
```

The key behaviors:

- **Errors are accumulated, not raised.** The agent copies the existing errors list, appends any new problems, and returns everything. The system never crashes from a single agent failure.
- **Each window is independent.** If the API fails for July 8 but works for July 15, you still get July 15 data.
- **Every successful response is saved to SQLite.** This builds a historical database automatically over time.
- **On failure, SQLite is the fallback.** If the API is down or quota is exhausted, the agent looks for similar historical data.
- **Historical data is flagged.** The `is_historical` field lets downstream nodes (and the UI) know when data is estimated rather than live.

Now let's see how each agent implements this pattern.

---

## Weather Agent in Detail

The weather agent is in `agents/weather.py`. It is the simplest of the three because Open-Meteo is free, requires no authentication, and has no rate limits.

### Step 1: Geocode the destination

The weather API needs latitude and longitude, not a city name. The agent starts by geocoding:

```python
from services.geocoding import geocode_city

def weather_node(state: TravelState) -> dict:
    errors = list(state.get("errors", []))
    try:
        geo = geocode_city(state["destination"])
    except ValueError as e:
        errors.append(f"Weather: geocoding failed — {e}")
        return {"weather_data": [], "errors": errors}
```

The geocoding service calls Open-Meteo's geocoding API:

```python
def geocode_city(city_name: str) -> dict:
    """Returns: {name, latitude, longitude, country, country_code}."""
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

If geocoding fails (typo in city name, API down), the weather agent returns immediately with an empty `weather_data` list and an error message. The system continues without weather data — the scorer will reweight priorities across the remaining dimensions.

### Step 2: Fetch weather for each window

The weather client chooses between two Open-Meteo endpoints based on whether the dates are in the future:

```python
def get_weather_for_window(latitude, longitude, start_date, end_date):
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

The Climate API uses the `EC_Earth3P_HR` model for projections. The Historical API uses actual recorded data. Both return the same shape — daily temperature, precipitation, and humidity — so the calling code does not need to know which endpoint was used.

Notice the `None` filtering: `[t for t in daily.get("temperature_2m_mean", []) if t is not None]`. API responses sometimes contain null values for days where data is unavailable. Silently skipping them and averaging the rest is more useful than crashing.

### Step 3: Score weather deterministically

This is where the design philosophy becomes concrete. We need a 0-to-1 score for each weather window. We could ask an LLM: "On a scale of 0 to 1, how good is 27.3C with 3 rain days and 72% humidity for tourism?" But that would give different scores every time, cost tokens, and be impossible to debug.

Instead, we use a formula:

```python
def score_weather(avg_temp, rain_days, avg_humidity,
                  ideal_temp_min=20.0, ideal_temp_max=28.0):
    """Score 0-1. Higher = better weather."""
    # Temperature: 1.0 if in ideal range, drops linearly outside
    if ideal_temp_min <= avg_temp <= ideal_temp_max:
        temp_score = 1.0
    else:
        distance = min(abs(avg_temp - ideal_temp_min), abs(avg_temp - ideal_temp_max))
        temp_score = max(0.0, 1.0 - distance / 20.0)

    # Rain: 0 rain days = 1.0, 7 rain days = 0.0
    rain_score = max(0.0, 1.0 - rain_days / 7.0)

    # Humidity: under 60% is ideal, penalize above
    if avg_humidity <= 60:
        humidity_score = 1.0
    elif avg_humidity <= 80:
        humidity_score = 1.0 - (avg_humidity - 60) / 40.0
    else:
        humidity_score = max(0.0, 0.5 - (avg_humidity - 80) / 40.0)

    # Weighted combination: temp 40% + rain 35% + humidity 25%
    return round(temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25, 3)
```

The scoring logic is transparent:

- **Temperature:** 20-28C is perfect (score 1.0). Each degree outside that range drops the score by 0.05. At 40C (12 degrees above 28), the temperature component is 0.4.
- **Rain:** Linear from 1.0 (no rain) to 0.0 (rain every day of the week).
- **Humidity:** Under 60% is comfortable (1.0). Between 60-80% it declines. Above 80% it drops further.
- **Weights:** Temperature matters most (40%), then rain (35%), then humidity (25%).

Why deterministic scoring? Because the same weather data produces the same score every time. You can write unit tests for it. You can explain to a user exactly why Tokyo in July scored 0.61 — "temperature was good at 27.3C, but 3 rain days and 72% humidity pulled the score down." Try getting that kind of consistency from an LLM.

---

## Flights Agent in Detail

The flights agent in `agents/flights.py` calls SerpApi's Google Flights engine to find flight prices.

### Step 1: No IATA resolution needed

SerpApi accepts both IATA codes and city names for `departure_id`/`arrival_id`, so the flights agent passes origin and destination directly — no separate IATA lookup step.

### The `_get_client()` singleton pattern

```python
_client = None
def _get_client():
    global _client
    if _client is None:
        _client = SerpApiClient(api_key=settings.serpapi_api_key)
    return _client
```

Why a module-level singleton? It avoids re-creating the client for each request. This is a pragmatic choice, not an architectural ideal. In a production system, you might use dependency injection. For a learning project, a module singleton is clear and works.

### Step 2: Search flights and parse prices

For each candidate window, the agent calls SerpApi Google Flights and extracts prices from `best_flights` and `other_flights`:

```python
def parse_flight_prices(api_response):
    all_flights = api_response.get("best_flights", []) + api_response.get("other_flights", [])
    if not all_flights: return None
    prices = [f["price"] for f in all_flights if "price" in f]
    if not prices: return None
    return {"min_price": min(prices), "avg_price": round(sum(prices)/len(prices), 2)}
```

SerpApi returns flights grouped into `best_flights` (curated picks) and `other_flights` (remaining options). We merge both lists, extract prices, and compute min and average. The `parse_flight_prices` function is deliberately separated from the node function — this makes it independently testable:

```python
def test_parse_prices():
    resp = {"best_flights": [{"price": 267}, {"price": 324}], "other_flights": [{"price": 278}]}
    result = parse_flight_prices(resp)
    assert result["min_price"] == 267
    assert result["avg_price"] == 289.67
```

### Step 3: Save to SQLite and handle failures

The flight data loop shows the full fallback pattern:

```python
for window in state["candidate_windows"]:
    try:
        response = client.search_flights(origin_iata, dest_iata, window["start"],
                                         currency=currency,
                                         adults=state.get("num_travelers", 1))
        parsed = parse_flight_prices(response)
        if parsed:
            db.save_flight(origin_iata, dest_iata, window["start"],
                          parsed["min_price"], currency)
            results.append({"window": window, "min_price": parsed["min_price"],
                "avg_price": parsed["avg_price"], "currency": currency,
                "score": 0.0, "is_historical": False})
        else:
            # API returned no offers — try historical
            hist = db.get_flight(origin_iata, dest_iata, window["start"],
                                tolerance_days=7)
            if hist:
                results.append({"window": window, "min_price": hist["price"],
                    "avg_price": hist["price"], "currency": hist["currency"],
                    "score": 0.0, "is_historical": True,
                    "fetched_at": hist["fetched_at"]})
    except Exception as e:
        # API error (network, auth, quota) — try historical
        hist = db.get_flight(origin_iata, dest_iata, window["start"],
                            tolerance_days=7)
        if hist:
            results.append({"window": window, "min_price": hist["price"],
                "avg_price": hist["price"], "currency": hist["currency"],
                "score": 0.0, "is_historical": True,
                "fetched_at": hist["fetched_at"]})
        else:
            errors.append(f"Flights: failed for {window['start']} — {e}")
```

Three paths, one result format. The downstream scorer does not need to know whether a price came from a live API call or a database lookup — it receives the same dict shape either way.

---

## Hotels Agent

The hotels agent in `agents/hotels.py` follows the same pattern as flights. SerpApi's Google Hotels engine accepts a text query (`q` param like "Hotels in Tokyo") — no IATA lookup or hotel ID resolution needed. A single API call returns hotel properties with nightly rates.

The price parsing extracts `rate_per_night.extracted_lowest` from each property and averages across all hotels:

```python
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
```

Properties missing `rate_per_night` are silently skipped rather than crashing the parser. The SQLite fallback pattern is identical to flights — save on success, query on failure with 7-day date tolerance.

---

## The Fallback Pattern — Why It Matters

The SQLite fallback is not just error handling. It is a system that gets better over time. Here is how it works across multiple uses:

**First search ever (Tokyo, July):** All data comes from live APIs. Every successful response is saved to SQLite. The database now has flight and hotel prices for Bangalore-Tokyo in July.

**Second search (Tokyo, July):** If APIs are available, you get fresh data. The fresh data is also saved, so the database now has two data points per route — useful for seeing price trends.

**API quota exhausted (100 searches/month used up):** SerpApi returns errors. The agents catch the exception and query SQLite. The user still gets results, marked as "estimated from history." The weather agent is unaffected (Open-Meteo has no limits).

**Similar search (Tokyo, August):** Even if the exact dates are not in the database, the 7-day tolerance query finds nearby data. A flight price for July 29 can serve as an estimate for August 3.

The SQLite schema is minimal:

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

The fallback query prioritizes closest date match, then most recent fetch:

```python
def get_flight(self, origin, destination, departure_date, tolerance_days=0):
    if tolerance_days > 0:
        row = self._conn.execute(
            "SELECT price, currency, fetched_at FROM flight_prices "
            "WHERE origin=? AND destination=? "
            "AND ABS(julianday(departure_date)-julianday(?))<=? "
            "ORDER BY ABS(julianday(departure_date)-julianday(?)) ASC, "
            "fetched_at DESC LIMIT 1",
            (origin, destination, departure_date,
             tolerance_days, departure_date)).fetchone()
```

This is the fallback hierarchy pattern from [Foundations Chapter 07](../foundations/07-reliability-and-production.md): try the best source first (live API), fall back to a less ideal source (historical data), and only report failure if everything is exhausted.

---

## "Tools" vs "Agents" — A Naming Clarification

In [Foundations Chapter 02](../foundations/02-tools-giving-llms-hands.md), we defined tools as functions that an LLM calls — `search_web`, `read_file`, `execute_code`. The LLM decides when and how to use them.

Our data agents are not called by an LLM. They are called by the graph. The supervisor does not "decide" to invoke the weather agent — the graph topology dictates it. There is no LLM reasoning involved in the orchestration.

So are they "tools" or "agents"? Honestly, the terminology is blurry and it does not matter much. What matters is the properties they exhibit:

- **Self-contained.** Each has a single responsibility: fetch one type of data.
- **Error-handling.** Each manages its own failures without crashing the system.
- **Structured I/O.** Each reads from specific state fields and writes to specific state fields.
- **Independently testable.** Each can be unit-tested with mocked dependencies.

In LangGraph terminology, they are **nodes** — Python functions that participate in a graph execution. In the broader agent literature, they are closer to "workers" in an orchestrator-workers pattern. Call them what makes sense to you. The architecture is the same regardless of the label.

The important insight is that not every node in a multi-agent system needs to contain an LLM. Most of our system is deterministic Python that calls APIs and processes data. The LLM is reserved for the one task where language generation genuinely adds value: explaining the results to the user.

---

## Key Takeaways

1. **All three data agents share the same pattern:** resolve identifiers, call API, save to SQLite, fall back on failure. Once you understand one, you understand all three.

2. **Deterministic scoring beats LLM scoring for structured data.** The weather scoring formula is consistent, fast, testable, and debuggable. An LLM would give different scores every run.

3. **The SQLite fallback creates a system that improves over time.** Each successful API call builds a historical database that future searches can draw from when APIs fail.

4. **Historical data is flagged, not hidden.** The `is_historical` field flows through the scorer to the UI, so users know when data is estimated.

5. **Each agent handles its own failures.** The graph does not have a global error handler. Each node catches exceptions, appends to the errors list, and returns whatever data it could gather.

6. **The distinction between "tools" and "agents" is less important than the properties of the code:** self-contained, error-handling, structured I/O, independently testable. These properties matter regardless of what you call the components.

Next chapters will cover the scorer (normalization and weighted ranking), the synthesizer (the one place we use an LLM), and the Streamlit UI that ties everything together.
