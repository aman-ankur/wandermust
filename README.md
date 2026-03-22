# Wandermust 🌍✈️

**AI-powered travel timing optimizer** — Find the perfect time to visit any destination by analyzing weather patterns, flight prices, and hotel costs across multiple date windows.

Built with **LangGraph** (multi-agent orchestration), **Streamlit** (UI), **Amadeus API** (flights + hotels), and **Open-Meteo** (weather).

---

## ✨ Features

- 🤖 **Multi-agent AI system** — 6 specialized agents working in parallel
- 🌤️ **Weather analysis** — Climate data and historical patterns
- ✈️ **Flight price tracking** — Real-time pricing from Amadeus
- 🏨 **Hotel cost analysis** — Accommodation pricing trends
- 📊 **Smart scoring** — Weighted ranking based on your priorities
- 💾 **Historical fallback** — SQLite cache when APIs are unavailable
- 🎨 **Beautiful UI** — Interactive Streamlit dashboard with charts

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Amadeus API credentials](https://developers.amadeus.com) (free tier: 500 calls/month)
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

**41 tests passing** across all agents and services.

---

## 🏗️ Architecture

```
User Input (Streamlit)
    ↓
Supervisor Agent (generates candidate date windows)
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Weather   │   Flights   │   Hotels    │  ← Parallel execution
│    Agent    │    Agent    │    Agent    │
└─────────────┴─────────────┴─────────────┘
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
| **Supervisor** | Generates rolling date windows from user input | Logic-based |
| **Weather** | Scores weather conditions (temp, rain, humidity) | Open-Meteo (free) |
| **Flights** | Finds cheapest flight prices | Amadeus API |
| **Hotels** | Calculates average nightly rates | Amadeus API |
| **Scorer** | Normalizes data & applies priority weights | Internal |
| **Synthesizer** | Generates natural language recommendations | OpenRouter LLM |

---

## 🛠️ Tech Stack

- **LangGraph** — Multi-agent orchestration with parallel execution
- **Streamlit** — Interactive web UI
- **Amadeus API** — Flight and hotel data
- **Open-Meteo** — Weather and geocoding (free, no key required)
- **OpenRouter** — LLM for recommendations
- **SQLite** — Historical data fallback
- **Pydantic** — Data validation
- **pytest** — Testing

---

## 📖 Documentation

- **[Build Guide](docs/BUILD_GUIDE.md)** — Complete implementation guide
- **[Implementation Plan](docs/superpowers/plans/2026-03-22-travel-optimizer-plan.md)** — Task-by-task breakdown
- **[Learning Guide](docs/learning/)** — AI agents fundamentals

---

## 🎯 Example Usage

1. Enter destination: "Tokyo, Japan"
2. Set date range: June 1 - September 30, 2026
3. Choose trip duration: 7 days
4. Adjust priorities: Weather 40%, Flights 30%, Hotels 30%
5. Click "Find Best Time"
6. Get AI-powered recommendations with ranked date windows

---

## 🔑 Configuration

Edit `.env` with your API credentials:

```env
# Amadeus API (https://developers.amadeus.com)
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret

# LLM via OpenRouter (https://openrouter.ai)
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514

# Defaults
DEFAULT_ORIGIN=Bangalore
DEFAULT_CURRENCY=INR
```

---

## 🚧 Roadmap

- [ ] Add multiple flight data sources (Kiwi, Skyscanner)
- [ ] Price trend visualization over time
- [ ] Budget ceiling filtering
- [ ] Multi-destination comparison
- [ ] Async API calls for faster execution
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
