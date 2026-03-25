# Wandermust Travel Optimizer - Project Context

## Quick Overview
AI-powered travel optimizer with two modes:
1. **Discover Where** - Conversational AI suggests destinations based on user profile
2. **Optimize When** - Multi-agent system finds best travel dates by analyzing weather, flights, hotels, social insights

## Tech Stack
- **Framework**: LangGraph (multi-agent orchestration)
- **UI**: Streamlit
- **APIs**: SerpApi (flights/hotels via Google), Open-Meteo (weather), Tavily (social), Reddit (optional)
- **LLM**: OpenAI (primary) or OpenRouter (fallback)
- **Storage**: SQLite (historical data fallback)
- **Language**: Python 3.9+

## Key Architecture Patterns
1. **Multi-agent pipeline**: Supervisor → 3 parallel data agents → Scorer → Synthesizer
2. **SQLite fallback**: Every API response cached for quota exhaustion scenarios
3. **Demo mode**: Full mock data flow, no API keys needed
4. **TDD approach**: Tests written first, implementation follows

## Current State (as of 2026-03-23)
- ✅ Amadeus API replaced with SerpApi (Google Flights & Hotels)
- ✅ Discovery flow with conversational onboarding
- ✅ Social insights via Tavily + Reddit (Reddit optional)
- ✅ Full test coverage (82 passing tests)
- ⚠️ Reddit client gracefully skips if credentials not configured

## API Keys Required
**Minimum viable**:
- `OPENAI_API_KEY` - LLM for synthesis, discovery, social extraction
- `SERPAPI_API_KEY` - Flights + hotels (100 free searches/month)

**Optional**:
- `TAVILY_API_KEY` - Social insights (1000 free/month)
- `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` - Social insights (free, unlimited)

## File Structure (Key Files Only)
```
agents/
  supervisor.py       - Generates candidate date windows
  weather.py          - Open-Meteo climate data + scoring
  flights.py          - SerpApi Google Flights search
  hotels.py           - SerpApi Google Hotels search
  social.py           - Tavily + Reddit → LLM extraction
  scorer.py           - Normalizes & ranks windows
  synthesizer.py      - LLM generates recommendation
  onboarding.py       - User profile collection
  discovery_chat.py   - Trip intent extraction
  suggestion_generator.py - LLM destination suggestions
  discovery_bridge.py - Converts discovery → optimizer state
  llm_helper.py       - OpenAI/OpenRouter factory

services/
  serpapi_client.py   - SerpApi wrapper (flights + hotels)
  geocoding.py        - Open-Meteo geocoding
  weather_client.py   - Open-Meteo weather API
  tavily_client.py    - Tavily search wrapper
  reddit_client.py    - Reddit search wrapper

app.py                - Streamlit UI (both modes)
graph.py              - LangGraph optimizer pipeline
discovery_graph.py    - LangGraph discovery pipeline
models.py             - Pydantic models (TravelState, DiscoveryState)
db.py                 - SQLite persistence
config.py             - Settings (loads from .env)
mock_data.py          - Demo mode data generators
```

## Common Tasks

### Run the app
```bash
streamlit run app.py
# or
python3 -m streamlit run app.py --server.port 8501
```

### Run tests
```bash
pytest tests/ -v
# Specific test file
pytest tests/test_serpapi_client.py -v
```

### Add new API integration
1. Create client in `services/`
2. Create tests in `tests/test_<name>.py`
3. Update agent to use new client
4. Update tests for agent
5. Run full test suite

### Debug LLM calls
Check terminal output - logging added to `_finish_discovery()` in app.py:
```
INFO:wandermust:Extracting trip intent...
INFO:wandermust:Generating destination suggestions...
```

## Important Notes
- `.env` is gitignored (safe for real keys)
- `.env.example` is tracked (use placeholders only)
- SerpApi uses ~34 calls per search (17 windows × 2 for flights+hotels)
- Weather agent always works (Open-Meteo is free, unlimited)
- Reddit client returns empty list if credentials not configured
- Demo mode bypasses all API calls
