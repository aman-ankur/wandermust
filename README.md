# Travel Optimizer

Find the best time to visit any destination — based on weather, flights, and hotels.

Built with LangGraph, Streamlit, Amadeus API, and Open-Meteo.

## Setup

```bash
cd travel-optimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

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
