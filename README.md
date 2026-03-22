# Wandermust 🌍✈️

**AI-powered travel optimizer** — Don't know where to go? **Discover** the perfect destination through a conversational AI. Already have a destination? **Optimize** the best travel dates by analyzing weather patterns, flight prices, hotel costs, and social insights.

Built with **LangGraph** (multi-agent orchestration), **Streamlit** (UI), **SerpApi** (flights + hotels via Google Flights & Google Hotels), **Open-Meteo** (weather), and **OpenRouter LLMs** (discovery + recommendations).

---

## ✨ Features

### 🔍 Discover Where (NEW)
- 🗺️ **Destination discovery** — Conversational AI that suggests where to travel
- 👋 **Smart onboarding** — Learns your travel history, style, and preferences (first-time only)
- 💬 **Interactive brainstorming** — Clickable option pills + free-text input for each question
- 🎯 **LLM-powered suggestions** — Reasons about visa, budget, seasonality, and your interests
- 🔗 **Bridge to optimizer** — One click to optimize dates for your chosen destination

### 📅 Optimize When
- 🤖 **Multi-agent AI system** — 7 specialized agents working in parallel
- 🌤️ **Weather analysis** — Climate data and historical patterns
- ✈️ **Flight price tracking** — Real-time pricing from SerpApi Google Flights
- 🏨 **Hotel cost analysis** — Accommodation pricing trends
- 🔍 **Social insights** — Crowd levels, events, and traveler tips from Reddit & Tavily
- 📊 **Smart scoring** — Weighted ranking based on your priorities
- 💾 **Historical fallback** — SQLite cache when APIs are unavailable

### General
- � **Demo mode** — Full end-to-end experience with mock data, no API keys needed
- �🎨 **Beautiful UI** — Interactive Streamlit dashboard with charts and chat interface

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [SerpApi API key](https://serpapi.com) (100 free searches/month)
- [OpenRouter API key](https://openrouter.ai) (for AI recommendations)

### Installation

```bash
# Clone the repository
git clone https://github.com/aman-ankur/wandermust.git
cd wandermust

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Run

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## 🧪 Testing

```bash
pytest tests/ -v
```

**131 tests passing** across all agents, services, and discovery components.

---

## 🏗️ Architecture

The app has **two modes**, each with its own LangGraph:

### Discover Where (discovery_graph.py)
```
Onboarding Agent → collects travel profile (first-time only)
    ↓
Discovery Chat Agent → asks adaptive trip questions
    ↓
Suggestion Generator → LLM reasons about destinations
    ↓
Bridge → converts choice to optimizer input
```

### Optimize When (graph.py)
```
Supervisor Agent (generates candidate date windows)
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│ Weather  │ Flights  │  Hotels  │  Social  │  ← Parallel execution
│  Agent   │  Agent   │  Agent   │  Agent   │
└──────────┴──────────┴──────────┴──────────┘
    ↓
Scorer Agent (normalizes & ranks)
    ↓
Synthesizer Agent (AI recommendation)
    ↓
Results Display
```

### Agent Roles

| Agent | Purpose | Data Source |
|-------|---------|-------------|
| **Onboarding** | Collects travel history & preferences | LLM + user input |
| **Discovery Chat** | Asks adaptive trip intent questions | LLM + user input |
| **Suggestion Generator** | Suggests 3–5 destinations with reasoning | OpenRouter LLM |
| **Bridge** | Converts discovery output to optimizer state | Logic-based |
| **Supervisor** | Generates rolling date windows from user input | Logic-based |
| **Weather** | Scores weather conditions (temp, rain, humidity) | Open-Meteo (free) |
| **Flights** | Finds cheapest flight prices | SerpApi Google Flights |
| **Hotels** | Calculates average nightly rates | SerpApi Google Hotels |
| **Social** | Crawls crowd levels, events, tips | Tavily + Reddit |
| **Scorer** | Normalizes data & applies priority weights | Internal |
| **Synthesizer** | Generates natural language recommendations | OpenRouter LLM |

---

## 🛠️ Tech Stack

- **LangGraph** — Multi-agent orchestration with parallel execution
- **Streamlit** — Interactive web UI
- **SerpApi** — Flight + hotel search via Google Flights & Google Hotels
- **Open-Meteo** — Weather and geocoding (free, no key required)
- **OpenRouter** — LLM for recommendations
- **SQLite** — Historical data fallback
- **Pydantic** — Data validation
- **pytest** — Testing

---

## 📖 Documentation

- **[Build Guide](docs/BUILD_GUIDE.md)** — Complete implementation guide
- **[Optimizer Plan](docs/superpowers/plans/2026-03-22-travel-optimizer-plan.md)** — Optimizer task-by-task breakdown
- **[Discovery Design Spec](docs/superpowers/specs/2026-03-22-destination-discovery-design.md)** — Discovery feature architecture
- **[Discovery Plan](docs/superpowers/plans/2026-03-22-destination-discovery-plan.md)** — Discovery implementation plan
- **[Learning Guide](docs/learning/)** — AI agents fundamentals

---

## 🎯 Example Usage

### Discover Where
1. Select **"🔍 Discover Where"** in the sidebar
2. Answer onboarding questions by clicking options or typing (first-time only)
3. Answer trip intent questions — when, how long, what interests you
4. Review destination suggestion cards with match scores
5. Click **"Let's go! ✈️"** on your favorite
6. Switch to **"📅 Optimize When"** — destination is pre-filled!

### Optimize When
1. Enter destination (or use one from discovery)
2. Set date range: June 1 - September 30, 2026
3. Choose trip duration: 7 days
4. Adjust priorities: Weather 40%, Flights 30%, Hotels 25%, Social 15%
5. Click "Find Best Time"
6. Get AI-powered recommendations with ranked date windows

---

## 🔑 Configuration

Edit `.env` with your API credentials:

```env
# SerpApi (https://serpapi.com) — flights + hotels via Google
SERPAPI_API_KEY=your_serpapi_key

# LLM via OpenRouter (https://openrouter.ai)
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514

# Defaults
DEFAULT_ORIGIN=Bangalore
DEFAULT_CURRENCY=INR
```

---

## 🚧 Roadmap

- [x] ~~Destination discovery with conversational AI~~
- [x] ~~Social media insights (Reddit, Tavily)~~
- [x] ~~Demo mode with full mock data~~
- [ ] Price trend visualization over time
- [ ] Multi-destination comparison
- [ ] Async API calls for faster execution
- [ ] User accounts & saved trips
- [ ] Deploy to Streamlit Cloud

---

## 📝 License

MIT

---

## 🤝 Contributing

Contributions welcome! This project is a great learning resource for:
- Multi-agent AI systems
- LangGraph orchestration
- API integration patterns
- Production-ready error handling

See `docs/learning/` for AI agent fundamentals.
