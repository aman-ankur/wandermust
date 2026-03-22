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

See implementation plan in `docs/superpowers/plans/2026-03-22-travel-optimizer-plan.md` for complete task-by-task breakdown with tests and code samples.

---

## Quick Start

```bash
# Clone and setup
cd travel-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Amadeus and OpenRouter keys

# Run tests
pytest tests/ -v

# Launch UI
streamlit run app.py
```

---

## Next Steps (after MVP)

1. **Add Kiwi/Tequila as secondary flight source** — natural "add another agent" exercise
2. **Price trend charts** — query SQLite history to show how prices change over months
3. **Budget filtering** — filter out windows where total cost exceeds budget ceiling
4. **Multiple destinations** — compare "Tokyo vs Bangkok vs Lisbon" side by side
5. **Deployment** — Streamlit Cloud (free) for personal access from anywhere
6. **Async agents** — convert to async httpx calls for true parallelism (LangGraph supports async nodes)
