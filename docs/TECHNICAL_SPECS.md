# Technical Specifications

## Data Flow

### Optimize When Mode
```
User Input (destination, dates, priorities)
    ↓
Supervisor → generates N candidate windows (7-day blocks)
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Weather   │   Flights   │   Hotels    │ (parallel)
│ Open-Meteo  │  SerpApi    │  SerpApi    │
└─────────────┴─────────────┴─────────────┘
    ↓
Scorer → normalize (0-1) + weighted rank
    ↓
Synthesizer → LLM generates recommendation
    ↓
Streamlit UI → cards, charts, tables
```

### Discover Where Mode
```
User → Onboarding (5 questions) → Profile extraction (LLM)
    ↓
User → Discovery (5 questions) → Trip intent extraction (LLM)
    ↓
LLM → Generate 3-5 destination suggestions
    ↓
User picks destination → Bridge to Optimizer
```

## API Integration Details

### SerpApi (Google Flights & Hotels)
**Endpoint**: `serpapi.com/search`
**Auth**: API key in query params
**Rate limit**: 100 searches/month (free tier)

**Flights** (`engine=google_flights`):
- Params: `departure_id`, `arrival_id`, `outbound_date`, `currency`, `adults`, `type=2` (one-way)
- Response: `{best_flights: [...], other_flights: [...]}`
- Each flight has `{price: number, ...}`

**Hotels** (`engine=google_hotels`):
- Params: `q="Hotels in {city}"`, `check_in_date`, `check_out_date`, `currency`, `adults`
- Response: `{properties: [{rate_per_night: {extracted_lowest: number}, ...}]}`

### Open-Meteo (Weather)
**Endpoints**:
- Geocoding: `geocoding-api.open-meteo.com/v1/search`
- Climate (future): `climate-api.open-meteo.com/v1/climate`
- Historical: `archive-api.open-meteo.com/v1/archive`

**Auth**: None (free, unlimited)
**Response**: Daily arrays of `temperature_2m_mean`, `precipitation_sum`, `relative_humidity_2m_mean`

### Tavily (Social Insights)
**Endpoint**: `api.tavily.com/search`
**Auth**: API key header
**Rate limit**: 1000 searches/month (free tier)
**Response**: `[{title, content, url, ...}]`

### Reddit (Social Insights)
**Library**: `praw` (Python Reddit API Wrapper)
**Auth**: `client_id`, `client_secret`, `user_agent`
**Rate limit**: 60 requests/minute (free, unlimited)
**Subreddits searched**: `travel`, `solotravel`, `TravelHacks`

## State Management

### TravelState (Optimizer)
```python
{
    "destination": str,
    "origin": str,
    "date_range": (start_iso, end_iso),
    "duration_days": int,
    "num_travelers": int,
    "budget_max": float | None,
    "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3, "social": 0.0},
    "candidate_windows": [{"start": iso, "end": iso}, ...],
    "weather_data": [{window, avg_temp, rain_days, humidity, score}, ...],
    "flight_data": [{window, min_price, avg_price, currency, is_historical}, ...],
    "hotel_data": [{window, avg_nightly, currency, is_historical}, ...],
    "social_data": [{window_start, window_end, social_score}, ...],
    "social_insights": [{destination, timing_score, crowd_level, events, tips, sentiment}, ...],
    "ranked_windows": [{window, total_score, weather_score, flight_score, ...}, ...],
    "recommendation": str,
    "errors": [str, ...]
}
```

### DiscoveryState
```python
{
    "user_profile": {
        "user_id": "default",
        "travel_history": [str, ...],
        "preferences": {"climate": str, "pace": str, "style": str},
        "budget_level": "budget|moderate|luxury",
        "passport_country": "IN"
    },
    "trip_intent": {
        "travel_month": str,
        "duration_days": int,
        "interests": [str, ...],
        "constraints": [str, ...],
        "travel_companions": "solo|couple|family|group",
        "region_preference": str,
        "budget_total": float
    },
    "suggestions": [{destination, country, reason, estimated_budget_per_day, match_score, tags}, ...],
    "chosen_destination": str | None,
    "errors": [str, ...]
}
```

## Scoring Algorithm

### Weather Score (0-1)
```python
temp_score = 1.0 if 20°C ≤ temp ≤ 28°C else max(0, 1 - distance/20)
rain_score = max(0, 1 - rain_days/7)
humidity_score = 1.0 if humidity ≤ 60% else declining function
final = 0.4*temp + 0.35*rain + 0.25*humidity
```

### Flight/Hotel Score (0-1)
```python
# Lower is better normalization
if all_prices:
    min_val, max_val = min(all_prices), max(all_prices)
    score = 1 - (price - min_val) / (max_val - min_val) if max_val > min_val else 1.0
```

### Total Score
```python
total = (weather_score * w_weather +
         flight_score * w_flights +
         hotel_score * w_hotels +
         social_score * w_social)
```

## SQLite Schema

```sql
-- User profiles
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    travel_history TEXT,  -- JSON array
    preferences TEXT,     -- JSON object
    budget_level TEXT,
    passport_country TEXT,
    created_at TEXT
);

-- Historical flight prices
CREATE TABLE flight_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT,
    destination TEXT,
    departure_date TEXT,
    price REAL,
    currency TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Historical hotel prices
CREATE TABLE hotel_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    checkin_date TEXT,
    avg_nightly REAL,
    currency TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Social insights cache
CREATE TABLE social_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    destination TEXT,
    month INTEGER,
    timing_score REAL,
    crowd_level TEXT,
    events TEXT,           -- JSON array
    itinerary_tips TEXT,   -- JSON array
    sentiment TEXT,
    source TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

-- Discovery sessions
CREATE TABLE discovery_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    trip_intent TEXT,      -- JSON object
    suggestions TEXT,      -- JSON array
    chosen_destination TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

## Error Handling Strategy

1. **Agent-level**: Each agent catches exceptions, appends to `errors` list, returns partial data
2. **API fallback**: On API failure, query SQLite for historical data within 7-day tolerance
3. **LLM fallback**: Synthesizer falls back to formatted text if LLM call fails
4. **Scorer reweighting**: If a dimension is missing, redistribute its weight to others
5. **Demo mode**: Bypass all APIs, use mock data generators

## Performance Characteristics

**Wall-clock time** (typical search with 17 windows):
- Supervisor: <1ms
- Weather: 3-5s (17 API calls, no auth)
- Flights: 5-15s (17 SerpApi calls)
- Hotels: 5-15s (17 SerpApi calls)
- Social: 5-10s (Tavily + Reddit + LLM extraction)
- Scorer: <1ms
- Synthesizer: 2-3s (1 LLM call)
- **Total: 15-40s** (bottleneck: SerpApi latency)

**API quota consumption** (per search):
- SerpApi: ~34 calls (17 flights + 17 hotels)
- Open-Meteo: ~18 calls (1 geocode + 17 climate)
- Tavily: 1 call
- Reddit: ~6 calls (3 subreddits × 2 queries)
- OpenAI: 1-3 calls (synthesis + optional discovery/onboarding)

## Testing Strategy

- **Unit tests**: Mock external APIs, test parsing logic
- **Integration tests**: Test agent nodes with mock state
- **Fixtures**: Use `@pytest.fixture` for common test data
- **Coverage**: 82 tests passing (as of 2026-03-23)

Run specific test suites:
```bash
pytest tests/test_serpapi_client.py -v
pytest tests/test_flights_agent.py -v
pytest tests/test_hotels_agent.py -v
pytest tests/test_scorer.py -v
```
