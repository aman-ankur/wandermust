# Chapter 05: Putting It Together — End-to-End Flow

Let's trace a real request through the entire system. The user enters "Tokyo" with dates June through September 2026. What happens at every step, in every agent, from click to result?

This chapter connects every concept from the foundations series to concrete code execution. By the end, you will understand exactly what happens at each stage, how long it takes, what can go wrong, and how the system recovers.

---

## Step 0: User Fills in the Form (Streamlit)

The user opens the Streamlit app and enters:

- **Destination:** "Tokyo, Japan"
- **Origin:** "Bangalore" (the default from `config.py`)
- **Date range:** June 1 to September 30, 2026
- **Duration:** 7 days
- **Priorities:** Weather 0.4, Flights 0.3, Hotels 0.3

When they click "Find Best Time", the Streamlit app in `app.py` normalizes the priority weights (ensuring they sum to 1.0) and builds the initial state dictionary:

```python
state = {
    "destination": "Tokyo, Japan",
    "origin": "Bangalore",
    "date_range": ("2026-06-01", "2026-09-30"),
    "duration_days": 7,
    "num_travelers": 1,
    "budget_max": None,
    "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
    "errors": [],
}
```

This state dictionary is a `TravelState` TypedDict — the shared data structure that flows through every node in the graph. As discussed in [Foundations Chapter 03](../foundations/03-state-and-memory.md), state is how agents communicate. No agent calls another agent directly. They all read from and write to this shared state.

The app then invokes the compiled LangGraph:

```python
result = build_graph().invoke(state)
```

From this point, LangGraph takes over and orchestrates every step.

---

## Step 1: Supervisor Generates Candidate Windows

The first node in the graph is the Supervisor (`agents/supervisor.py`). Its job is pure date math — no APIs, no LLMs.

The `generate_candidate_windows` function rolls through June 1 to September 30 in 7-day blocks, sliding forward by 7 days each step:

```python
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
```

For our June-September range, this produces approximately 17 windows:
- Jun 1-8, Jun 8-15, Jun 15-22, Jun 22-29, Jun 29-Jul 6, Jul 6-13, ... Sep 16-23, Sep 23-30

The Supervisor writes these into state and the graph continues:

```python
state after: {..., "candidate_windows": [{"start": "2026-06-01", "end": "2026-06-08"}, ...]}
```

**Time:** less than 1 millisecond. This is pure `datetime` arithmetic.

This is the "plan" step in the plan-then-execute pattern from [Foundations Chapter 04](../foundations/04-single-agent-architectures.md). The Supervisor decides *what work needs to be done* (which windows to evaluate) before any agent does the work.

---

## Step 2: Three Agents Run in Parallel

This is where the architecture pays off. LangGraph's fan-out edges mean the weather, flights, and hotels agents all start simultaneously:

```python
# From graph.py
graph.add_edge("supervisor", "weather")
graph.add_edge("supervisor", "flights")
graph.add_edge("supervisor", "hotels")
```

All three agents receive the same state (with `candidate_windows` populated). They run concurrently as described in [Foundations Chapter 05](../foundations/05-multi-agent-architectures.md) and [Chapter 06](../foundations/06-graphs-and-orchestration.md). Let's trace each one.

### Weather Agent (Open-Meteo) — ~3-5 seconds

The weather agent in `agents/weather.py` does two things: geocode the city, then fetch climate data for each window.

**Geocoding:** Calls Open-Meteo's geocoding API to convert "Tokyo, Japan" into coordinates: latitude 35.68, longitude 139.69. This is a single HTTP call via `services/geocoding.py`.

**Climate data:** For each of the ~17 windows, calls the Open-Meteo Climate API with those coordinates. Since all dates are in the future (June-September 2026), it uses the `CLIMATE_URL` endpoint with the `EC_Earth3P_HR` climate model. Each call returns daily temperature, precipitation, and humidity.

**Scoring:** For each window, the agent computes a weather score using `score_weather`:

```python
def score_weather(avg_temp, rain_days, avg_humidity,
                  ideal_temp_min=20.0, ideal_temp_max=28.0):
    # Temperature: 1.0 if in ideal range, degrades as distance increases
    # Rain: 0 days = 1.0, 7 days = 0.0
    # Humidity: <=60% = 1.0, degrades above
    return round(temp_score * 0.4 + rain_score * 0.35 + humidity_score * 0.25, 3)
```

A July window might return: avg_temp 28.5 degrees, 4 rain days (monsoon), humidity 75% — scoring around 0.55. A late September window: 24 degrees, 1 rain day, humidity 58% — scoring around 0.85.

**Output:** 17 entries in `weather_data`, each with temperature, rain days, humidity, and a 0-1 score.

**Time:** ~3-5 seconds for 17 API calls. Open-Meteo is free and has no rate limits, so this is just network latency.

### Flights Agent (Amadeus) — ~10-20 seconds

The flights agent in `agents/flights.py` is the most complex of the three because it deals with authenticated APIs, rate limits, and fallback logic.

**IATA resolution:** First, it resolves city names to airport codes using the Amadeus IATA lookup: "Bangalore" becomes BLR, "Tokyo" becomes NRT. This takes 2 API calls.

**Flight search:** For each of the ~17 windows, it searches for flights from BLR to NRT departing on the window's start date. Each call hits `AmadeusClient.search_flights`, which:
1. Checks if the OAuth token is still valid (auto-refreshes if within 1 minute of expiry)
2. Calls the Amadeus Flight Offers endpoint with `max=5` results
3. Returns up to 5 flight offers with prices

**Price parsing:** The `parse_flight_prices` function extracts the minimum and average prices:

```python
def parse_flight_prices(api_response):
    offers = api_response.get("data", [])
    if not offers: return None
    prices = [float(o["price"]["total"]) for o in offers]
    return {"min_price": min(prices), "avg_price": round(sum(prices)/len(prices), 2)}
```

**SQLite persistence:** Every successful price is saved to the `flight_prices` table in SQLite. This builds the historical database that enables fallback when the API is unavailable later.

**Fallback on failure:** If any individual window's API call fails (rate limit, timeout, server error), the agent queries SQLite for the same route and date within a 7-day tolerance. If found, it uses the historical price and flags `is_historical: True`.

**Output:** ~17 entries in `flight_data`, each with min_price, avg_price, currency, and an is_historical flag.

**Time:** ~10-20 seconds. Each Amadeus call takes 0.5-1 second, and there are ~17 flight searches plus 2 IATA lookups plus potential token refresh. This is the bottleneck of the entire pipeline.

### Hotels Agent (Amadeus) — ~10-20 seconds

The hotels agent in `agents/hotels.py` follows the same pattern as flights but with a two-stage Amadeus call.

**IATA resolution:** Resolves "Tokyo" to city code TYO.

**Hotel list:** Fetches up to 10 hotels in Tokyo using the Hotel List endpoint. This call is the same for every window and would benefit from caching (the first call populates the list; subsequent searches for the same city could reuse it).

**Hotel offers:** For each window, queries hotel offers across those 10 hotels with `bestRateOnly=true`, returning the cheapest rate per hotel. The `parse_hotel_prices` function averages across hotels:

```python
def parse_hotel_prices(api_response):
    hotels = api_response.get("data", [])
    if not hotels: return None
    prices = []
    for h in hotels:
        offers = h.get("offers", [])
        if offers: prices.append(float(offers[0]["price"]["total"]))
    if not prices: return None
    return {"avg_nightly": round(sum(prices)/len(prices), 2)}
```

**Fallback:** Same SQLite pattern as flights — saves every successful response, falls back to historical data on failure.

**Output:** ~17 entries in `hotel_data`.

**Time:** ~10-20 seconds, similar to flights. The hotel list call is one extra API call, but `bestRateOnly` keeps individual offer queries fast.

### Parallel Execution Total

Because all three agents run concurrently, the total wall-clock time is determined by the slowest agent — the Amadeus bottleneck at ~10-20 seconds. Without parallelism, this would be 25-45 seconds (weather + flights + hotels sequentially). The fan-out pattern from [Foundations Chapter 05](../foundations/05-multi-agent-architectures.md) cuts the wait roughly in half.

---

## Step 3: Scorer Ranks the Windows

Once all three agents complete, LangGraph's fan-in edges route their outputs to the Scorer:

```python
graph.add_edge("weather", "scorer")
graph.add_edge("flights", "scorer")
graph.add_edge("hotels", "scorer")
```

The Scorer in `agents/scorer.py` now has `weather_data` (17 entries), `flight_data` (17 entries), and `hotel_data` (17 entries) in state.

**Normalization:** Flight prices are normalized with `lower_is_better=True` (cheapest = 1.0). Hotel prices get the same treatment. Weather scores are already 0-1 from the weather agent, so they pass through unchanged.

**Weighting:** Applies the user's priorities: `0.4 * weather + 0.3 * flights + 0.3 * hotels`.

**Ranking:** Sorts all windows by total_score descending.

Walking through the top 3 results with realistic Tokyo numbers:

| Rank | Window | Weather | Flights | Hotels | Total |
|------|--------|---------|---------|--------|-------|
| #1 | Sep 15-22 | 0.82 | 0.95 | 0.88 | 0.878 |
| #2 | Sep 8-15 | 0.78 | 0.90 | 0.85 | 0.837 |
| #3 | Jun 1-8 | 0.70 | 0.85 | 0.80 | 0.775 |

September dominates because the monsoon is ending (decent weather scores), and both flights and hotels hit seasonal lows after the summer tourism peak. June scores reasonably before the monsoon ramps up, but July-August windows would rank lower due to heat, rain, and peak-season pricing.

**Time:** less than 1 millisecond. This is array iteration and arithmetic — no I/O whatsoever.

As covered in [Chapter 04 of this series](./04-scoring-and-synthesis.md), the Scorer also handles missing dimensions by reweighting priorities. If hotel_data were empty, it would redistribute the 0.3 hotel weight proportionally to weather and flights.

---

## Step 4: Synthesizer Explains

The Synthesizer in `agents/synthesizer.py` takes the top 5 ranked windows, formats them into a text summary, and sends a single prompt to an LLM via OpenRouter:

```python
prompt = (f"You are a travel advisor. Based on these ranked travel windows for a trip from "
    f"{state.get('origin', 'your city')} to {state.get('destination', 'the destination')}, "
    f"write a concise recommendation (3-5 sentences) about which dates are best and why. "
    f"Mention weather, flight cost, and hotel cost. If data is estimated from history, note that.\n\n{data_summary}")
```

The LLM receives pre-computed numbers — scores, prices, dates — and generates something like:

> *"Mid-September is your best window for Tokyo. The summer heat has eased to a comfortable 26 degrees with only 2 rain days expected as monsoon season winds down. Flights from Bangalore are at a seasonal low of around 18,000 INR, and hotels average 6,500 INR per night. If you have flexibility, the September 15-22 window offers the best combination across all three factors."*

**Time:** ~2-3 seconds for one LLM call. This is the only LLM call in the entire pipeline — approximately 500 tokens of input and 100-150 tokens of output, costing roughly 0.01 USD.

If the LLM call fails, the fallback kicks in and the user sees the formatted text version instead. Functional, just not polished.

---

## Step 5: Streamlit Renders Results

Back in `app.py`, the graph invocation returns the final state. Streamlit renders:

1. **Recommendation text** — the LLM's natural language summary (or fallback text)
2. **Top 3 cards** — each showing the window dates, total score, and per-dimension breakdown with costs
3. **Comparison table** — a DataFrame of all ranked windows with scores and costs
4. **Errors section** — a collapsed expander showing any warnings from agents that encountered issues

If any window used historical data (SQLite fallback), it gets an "(estimated)" badge so the user knows the prices may not be current.

---

## Error Scenarios — What If Things Go Wrong

Every stage in the pipeline is designed to degrade gracefully rather than crash. Here is what happens for each failure mode:

| Failure | What Happens | User Sees |
|---------|-------------|-----------|
| Amadeus rate limited | Flights/hotels agent catches the HTTP error, queries SQLite for historical prices within 7-day tolerance | Results with "(estimated)" badge, or missing dimension if no history exists |
| Open-Meteo down | Weather agent returns empty `weather_data`, Scorer reweights to flights + hotels only | Results without weather scores, warning in errors section |
| LLM down (OpenRouter) | Synthesizer catches the exception, returns `format_ranked_data_fallback` output | Plain text ranking instead of natural language — still contains all numbers |
| City not found in geocoding | `geocode_city` raises `ValueError`, weather agent catches it and appends to errors | Error message displayed, flights/hotels agents may still work if IATA resolves |
| IATA code not found | Flights or hotels agent returns empty data for that dimension | Scorer reweights remaining dimensions, warning in errors |
| All APIs down | All three agents return empty data, Scorer has nothing to rank | "No results found" message |
| Network timeout | Each API call has a 10-second timeout configured in `config.py`, with 3 retries via httpx | Delayed results or fallback to SQLite |

The key design principle: each agent catches its own exceptions and writes errors to the shared `errors` list rather than letting exceptions propagate. This means a failure in one agent never prevents the others from completing. The Scorer and Synthesizer always run, even if they receive partial data.

This is the reliability pattern from [Foundations Chapter 07](../foundations/07-reliability-and-production.md) applied at every layer.

---

## Performance Budget

Understanding the API economics is important for a project that depends on free-tier services.

**Amadeus (500 calls/month shared across all endpoints):**
- Per search: ~2 IATA lookups + ~17 flight searches + 1 hotel list + ~17 hotel offer searches = ~37 Amadeus calls
- Monthly budget: approximately 13 searches before hitting the free tier limit
- Mitigation: SQLite fallback means repeat searches for the same route use cached data, consuming zero API calls

**Open-Meteo (unlimited, free, no auth):**
- Per search: 1 geocoding call + ~17 climate calls = ~18 calls
- No limits. This is why weather was the easiest agent to build.

**OpenRouter (pay per token):**
- Per search: 1 LLM call, ~500 input tokens + ~150 output tokens
- Cost: approximately 0.01 USD per search with Claude Sonnet
- At 10 searches per month: 0.10 USD total

**Wall-clock time:**
- Supervisor: <1ms
- Data agents (parallel): 10-20 seconds
- Scorer: <1ms
- Synthesizer: 2-3 seconds
- **Total: ~12-23 seconds**

The bottleneck is Amadeus API latency. For a second search with the same origin-destination pair, cached IATA codes and SQLite fallback data can reduce this significantly.

---

## What You Have Learned

This end-to-end trace connects back to every chapter in the foundations series:

- **[Chapter 01 — What Are AI Agents](../foundations/01-what-are-ai-agents.md):** This is an agent system. The Supervisor makes autonomous decisions about which windows to evaluate. The data agents use tools (APIs) to gather information. The Synthesizer uses an LLM to generate output. All three characteristics of agency are present.

- **[Chapter 02 — Tools](../foundations/02-tools-giving-llms-hands.md):** The API clients in `services/` are the tools. `weather_client.py`, `amadeus_client.py`, and `geocoding.py` each wrap an external API with a clean Python interface. The agents call these tools — they do not make HTTP requests directly.

- **[Chapter 03 — State and Memory](../foundations/03-state-and-memory.md):** `TravelState` is the shared state flowing through the graph. SQLite provides long-term memory — historical prices persist across sessions and enable fallback.

- **[Chapter 04 — Single Agent Architectures](../foundations/04-single-agent-architectures.md):** The Supervisor uses plan-then-execute. It generates candidate windows (the plan) before any agent fetches data (the execution).

- **[Chapter 05 — Multi-Agent Architectures](../foundations/05-multi-agent-architectures.md):** Three data agents with distinct specializations, running in parallel. Fan-out from Supervisor, fan-in to Scorer.

- **[Chapter 06 — Graphs and Orchestration](../foundations/06-graphs-and-orchestration.md):** LangGraph's `StateGraph` with `add_edge` for both parallel and sequential edges. The compiled graph handles execution order automatically.

- **[Chapter 07 — Reliability and Production](../foundations/07-reliability-and-production.md):** Fallbacks everywhere. SQLite for API failures, formatted text for LLM failures, reweighting for missing dimensions, error collection instead of exception propagation.

---

## Key Takeaways

1. **Parallelism is the biggest performance win.** Running three agents concurrently cuts wall-clock time from 25-45 seconds to 10-20 seconds. The graph definition makes this trivial — just add edges from the same source to multiple targets.

2. **The bottleneck is always the external API.** All internal processing (window generation, scoring, formatting) takes under 1 millisecond combined. The 10-20 seconds of latency is entirely network I/O to Amadeus. Design around this reality.

3. **Free-tier API limits shape architecture.** The SQLite fallback, the `bestRateOnly` flag, the `max=5` flight results — these are all driven by the 500 calls/month Amadeus limit. Constraints breed design.

4. **Error isolation is non-negotiable.** Each agent catches its own exceptions. A failure in hotels never prevents weather from completing. The Scorer handles whatever data arrives. The Synthesizer works with or without an LLM.

5. **One LLM call is enough.** The entire system uses a single LLM invocation — at the very end, for natural language generation. Everything else is deterministic code. This keeps costs near zero and behavior predictable.

**Final note:** You have now seen the theory and the practice. The `BUILD_GUIDE.md` has all the code to implement this yourself — 14 tasks, test-first, one commit at a time. Happy building.
