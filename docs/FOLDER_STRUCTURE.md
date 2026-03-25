# Complete Folder Structure

```
travel-optimizer/
├── .env                          # Real API keys (gitignored, safe)
├── .env.example                  # Template with placeholders (tracked)
├── .gitignore                    # Git ignore patterns
├── requirements.txt              # Python dependencies
├── config.py                     # Settings loader (reads .env)
├── app.py                        # Streamlit UI (both modes)
├── graph.py                      # LangGraph optimizer pipeline
├── discovery_graph.py            # LangGraph discovery pipeline
├── models.py                     # Pydantic models (TravelState, DiscoveryState)
├── db.py                         # SQLite persistence layer
├── cache.py                      # Cache utilities (unused currently)
├── mock_data.py                  # Demo mode data generators
├── README.md                     # User-facing documentation
├── BUILD_GUIDE.md                # Implementation guide
│
├── agents/                       # LangGraph agent nodes
│   ├── __init__.py
│   ├── supervisor.py             # Generates candidate date windows
│   ├── weather.py                # Weather data + scoring (Open-Meteo)
│   ├── flights.py                # Flight search (SerpApi Google Flights)
│   ├── hotels.py                 # Hotel search (SerpApi Google Hotels)
│   ├── social.py                 # Social insights (Tavily + Reddit → LLM)
│   ├── scorer.py                 # Normalizes & ranks windows
│   ├── synthesizer.py            # LLM recommendation generator
│   ├── onboarding.py             # User profile collection (LangGraph)
│   ├── discovery_chat.py         # Trip intent extraction (LangGraph)
│   ├── suggestion_generator.py   # LLM destination suggestions
│   ├── discovery_bridge.py       # Converts discovery → optimizer state
│   ├── llm_helper.py             # OpenAI/OpenRouter factory
│   └── mock_agents.py            # Demo mode agent implementations
│
├── services/                     # External API clients
│   ├── __init__.py
│   ├── serpapi_client.py         # SerpApi wrapper (flights + hotels)
│   ├── geocoding.py              # Open-Meteo geocoding
│   ├── weather_client.py         # Open-Meteo weather API
│   ├── tavily_client.py          # Tavily search wrapper
│   └── reddit_client.py          # Reddit search wrapper (praw)
│
├── tests/                        # Test suite (82 passing tests)
│   ├── __init__.py
│   ├── test_serpapi_client.py    # SerpApi client tests
│   ├── test_flights_agent.py     # Flight agent tests
│   ├── test_hotels_agent.py      # Hotel agent tests
│   ├── test_weather_agent.py     # Weather agent tests
│   ├── test_weather_client.py    # Weather client tests
│   ├── test_social_agent.py      # Social agent tests (needs langchain_openai)
│   ├── test_scorer.py            # Scorer tests
│   ├── test_synthesizer.py       # Synthesizer tests (needs langchain_openai)
│   ├── test_supervisor.py        # Supervisor tests
│   ├── test_db.py                # Database tests
│   ├── test_cache.py             # Cache tests
│   ├── test_models.py            # Pydantic model tests
│   ├── test_reddit_client.py     # Reddit client tests
│   ├── test_tavily_client.py     # Tavily client tests
│   ├── test_onboarding.py        # Onboarding tests (needs langchain_openai)
│   ├── test_discovery_chat.py    # Discovery chat tests (needs langchain_openai)
│   ├── test_suggestion_generator.py  # Suggestion tests (needs langchain_openai)
│   ├── test_graph.py             # Graph tests (needs langchain_openai)
│   ├── test_discovery_graph.py   # Discovery graph tests (needs langchain_openai)
│   └── test_mock_agents.py       # Mock agent tests (needs langchain_openai)
│
└── docs/                         # Documentation
    ├── PROJECT_CONTEXT.md        # Quick overview for AI assistants
    ├── TECHNICAL_SPECS.md        # Detailed technical specifications
    ├── FOLDER_STRUCTURE.md       # This file
    ├── BUILD_GUIDE.md            # Complete build guide
    │
    ├── learning/                 # Educational walkthrough
    │   ├── README.md
    │   ├── foundations/          # Conceptual foundations
    │   │   ├── 01-what-are-ai-agents.md
    │   │   ├── 02-tools-giving-llms-hands.md
    │   │   ├── 03-state-and-memory.md
    │   │   ├── 04-single-agent-architectures.md
    │   │   ├── 05-multi-agent-architectures.md
    │   │   ├── 06-graphs-and-orchestration.md
    │   │   └── 07-reliability-and-production.md
    │   │
    │   └── project-walkthrough/  # Implementation walkthrough
    │       ├── 01-project-overview.md
    │       ├── 02-setup-and-config.md
    │       ├── 03-data-agents.md
    │       ├── 04-scoring-and-synthesis.md
    │       └── 05-end-to-end-flow.md
    │
    └── superpowers/              # Advanced features & plans
        ├── plans/
        │   ├── 2026-03-22-travel-optimizer-plan.md
        │   └── 2026-03-22-replace-amadeus-with-serpapi.md
        │
        └── specs/
            ├── 2026-03-22-travel-optimizer-design.md
            └── 2026-03-22-agent-learning-guide-design.md
```

## Key Directories Explained

### `/agents/`
LangGraph nodes — each is a pure function that takes state dict, does work, returns state updates. No LLM in data agents (weather/flights/hotels), only in synthesis and discovery.

### `/services/`
External API wrappers. Each client is a class with methods that return structured data. All handle errors gracefully (return empty/None instead of raising).

### `/tests/`
Pytest test suite. Mocks external APIs using `unittest.mock.patch`. 82 tests pass, 8 fail due to missing `langchain_openai` (not critical for core functionality).

### `/docs/learning/`
Educational content explaining AI agent concepts and implementation details. Written for developers learning multi-agent systems.

### `/docs/superpowers/`
Implementation plans and design specs. Contains complete code for major features.

## Files You'll Edit Most

**Adding new API**:
- `services/<name>_client.py` - API wrapper
- `tests/test_<name>_client.py` - Client tests
- `agents/<name>.py` - Agent using the client
- `tests/test_<name>_agent.py` - Agent tests

**Modifying UI**:
- `app.py` - All Streamlit UI code (both modes)

**Changing data flow**:
- `graph.py` - Optimizer pipeline edges
- `discovery_graph.py` - Discovery pipeline edges
- `models.py` - State schemas

**Updating config**:
- `config.py` - Add new settings
- `.env.example` - Add placeholder
- `.env` - Add real value (gitignored)

## Generated/Runtime Files (gitignored)

```
.venv/                  # Python virtual environment
__pycache__/            # Python bytecode cache
*.pyc                   # Compiled Python files
.pytest_cache/          # Pytest cache
travel_history.db       # SQLite database (created at runtime)
.streamlit/             # Streamlit config (created at runtime)
.env                    # Real API keys (NEVER commit this)
```

## Important: .env vs .env.example

- **`.env`** - Real API keys, gitignored, safe to edit
- **`.env.example`** - Template with placeholders, tracked by git, NEVER put real keys here

Always edit `.env` for real keys, keep `.env.example` with placeholders only.
