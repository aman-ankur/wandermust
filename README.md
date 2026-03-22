# Travel Optimizer

Find the best time to visit any destination — based on weather, flights, and hotels.

Built with LangGraph (multi-agent orchestration), Streamlit (UI), Amadeus API (flights + hotels), and Open-Meteo (weather + geocoding).

## Implementation Status

All 14 tasks complete. **41 tests passing.**

| # | Task | Status | Files |
|---|------|--------|-------|
| 1 | Project Scaffolding | Done | `config.py`, `models.py`, `.gitignore`, `requirements.txt`, `.env.example` |
| 2 | Geocoding Service | Done | `services/geocoding.py` |
| 3 | Cache + SQLite Store | Done | `cache.py`, `db.py` |
| 4 | Weather Client | Done | `services/weather_client.py` |
| 5 | Amadeus Client | Done | `services/amadeus_client.py` |
| 6 | Supervisor Agent | Done | `agents/supervisor.py` |
| 7 | Weather Agent | Done | `agents/weather.py` |
| 8 | Flights Agent | Done | `agents/flights.py` |
| 9 | Hotels Agent | Done | `agents/hotels.py` |
| 10 | Scorer Agent | Done | `agents/scorer.py` |
| 11 | Synthesizer Agent | Done | `agents/synthesizer.py` |
| 12 | LangGraph Wiring | Done | `graph.py` |
| 13 | Streamlit UI | Done | `app.py` |
| 14 | README | Done | `README.md` |

## Setup

```bash
cd travel-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

### Required API Keys

| Key | Source | Notes |
|-----|--------|-------|
| `AMADEUS_CLIENT_ID` | [developers.amadeus.com](https://developers.amadeus.com) | Free tier: 500 calls/month |
| `AMADEUS_CLIENT_SECRET` | Same as above | |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) | For LLM synthesis |

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
User → Supervisor → [Weather | Flights | Hotels] parallel → Scorer → Synthesizer → UI
```

### Agents

- **Supervisor** — Parses input, generates rolling candidate date windows
- **Weather Agent** — Fetches Open-Meteo climate/historical data, scores by temp/rain/humidity
- **Flights Agent** — Queries Amadeus Flight Offers, falls back to SQLite history
- **Hotels Agent** — Queries Amadeus Hotel Offers, falls back to SQLite history
- **Scorer** — Normalizes all data 0–1, applies user priority weights, ranks windows
- **Synthesizer** — LLM-generated recommendation via OpenRouter (falls back to raw data on failure)

### Services

- **Geocoding** — City name → lat/lon via Open-Meteo (free, no key)
- **Weather Client** — Climate API (future dates) / Historical API (past dates)
- **Amadeus Client** — OAuth2 token management, flight search, hotel search, IATA lookup

### Data Layer

- **TTL Cache** — In-memory cache (24h default) to avoid redundant API calls
- **SQLite History** — Stores flight/hotel prices for historical fallback when APIs fail

### Key Design Decisions

- LangGraph parallel fan-out for weather/flights/hotels (concurrent data fetching)
- `Annotated[List[str], operator.add]` reducer on `errors` field for safe concurrent writes
- `typing.Optional`/`Tuple`/`List`/`Dict` in `TravelState` for Python 3.9 compatibility with LangGraph's `get_type_hints()`
- Scorer reweights priorities automatically when a data dimension is missing
