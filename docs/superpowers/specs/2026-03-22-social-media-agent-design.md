# Social Media Intelligence Agent — Design Spec

## Overview

Add a social media intelligence agent to Wandermust that crawls Twitter/X and Reddit for destination-specific travel timing insights and itinerary suggestions. The agent integrates fully into the existing LangGraph pipeline as a 4th parallel data agent alongside weather, flights, and hotels.

## Goals

1. **Timing intelligence** — extract "best time to visit" wisdom: seasonal tips, crowd levels, local events, weather anecdotes from real travelers
2. **Itinerary suggestions** — collect must-see places, recommended day plans, hidden gems from locals
3. **Full pipeline integration** — social score influences ranking; insights enrich the synthesizer recommendation

## Data Sources

### Tavily Search API
- **Purpose:** Twitter/X content and general travel web content
- **Cost:** Free tier = 1,000 searches/month; paid = $20/mo for 10,000
- **Queries per destination:** 2-3 (site-filtered for Twitter + general travel)
- **SDK:** `tavily-python`

### Reddit via PRAW
- **Purpose:** Deep thread access to r/travel, r/solotravel, r/TravelHacks
- **Cost:** Free for non-commercial use (100 requests/min)
- **Queries per destination:** 2-3 (timing + itinerary searches)
- **SDK:** `praw`

### LLM Extraction (Haiku 3.5)
- **Purpose:** Extract structured insights from raw social media content
- **Model:** `anthropic/claude-haiku-3.5` via OpenRouter
- **Cost:** ~$0.005 per trip search (~4,500 tokens)
- **Approach:** One batched call per destination (not per window)

## Architecture

### Pipeline Position

```
Supervisor
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│ Weather  │ Flights  │ Hotels   │ Social   │  ← PARALLEL
└──────────┴──────────┴──────────┴──────────┘
    ↓
Scorer (4 dimensions: weather, flights, hotels, social)
    ↓
Synthesizer (enriched with social insights)
```

### New State Fields (TravelState)

```python
social_data: List[dict]      # Per-window social scores
social_insights: List[dict]  # Structured tips (destination-level)
```

### Social Data Schema (per window)

```python
{
    "window_start": "2025-04-01",
    "window_end": "2025-04-07",
    "social_score": 0.85,        # 0-1, higher = more recommended
}
```

### Social Insights Schema (per destination)

```python
{
    "destination": "Tokyo, Japan",
    "timing_score": 0.85,         # Overall social sentiment about timing
    "crowd_level": "moderate",    # low | moderate | high | extreme
    "events": [                   # Relevant events near travel dates
        {"name": "Cherry Blossom Festival", "period": "late March - mid April"}
    ],
    "itinerary_tips": [           # Top suggestions from social media
        {"tip": "Visit Fushimi Inari at sunrise to avoid crowds", "source": "reddit"},
        {"tip": "Tsukiji outer market is better than Toyosu for tourists", "source": "twitter"}
    ],
    "sentiment": "highly recommended",  # Overall traveler sentiment
    "sources": [                  # Links to original posts
        {"url": "https://...", "platform": "reddit", "title": "..."}
    ]
}
```

## Components

### New Files

#### `services/tavily_client.py`

```python
def search_destination(destination: str, month: str) -> List[dict]:
    """Search Tavily for travel insights about a destination.

    Queries:
    1. "best time to visit {destination} {month}" (site:twitter.com OR site:x.com)
    2. "best time to visit {destination} travel tips"

    Returns list of {title, content, url, source} dicts.
    """
```

#### `services/reddit_client.py`

```python
def search_subreddits(
    destination: str,
    subreddits: List[str] = ["travel", "solotravel", "TravelHacks"],
    time_filter: str = "year",  # last 2 years
    limit: int = 10
) -> List[dict]:
    """Search Reddit for travel posts about a destination.

    Returns list of {title, body, top_comments, url, subreddit, score} dicts.
    Filters by relevance and recency.
    """
```

#### `agents/social.py`

Follows the existing 5-step agent pattern:

```python
def social_node(state: TravelState) -> dict:
    """Social media intelligence agent.

    1. Fetch from Tavily (Twitter/web content)
    2. Fetch from PRAW (Reddit threads)
    3. Batch all content into one Haiku 3.5 extraction call
    4. Parse structured output (timing_score, crowd_level, events, itinerary_tips)
    5. Save to SQLite for historical fallback

    Returns social_data (per-window scores) and social_insights (destination-level tips).
    Error accumulation follows existing pattern — never raises.
    """
```

**LLM Extraction Prompt:**
- Input: Raw social media content (posts, comments, search results)
- Instruction: Extract timing recommendations, crowd levels, events, itinerary tips
- Output: JSON matching the social insights schema
- Model: Haiku 3.5 via OpenRouter

**Social Scoring Logic:**
- `timing_score` from LLM extraction (0-1 based on sentiment)
- Per-window score adjusted by: does the window overlap with recommended timing from social data?
- Windows matching popular recommended periods get higher scores

### Modified Files

#### `models.py`
- Add `social_data` and `social_insights` fields to `TravelState`

#### `graph.py`
- Add `social_node` to the parallel fan-out after supervisor
- Wire social_node → scorer (alongside weather, flights, hotels)

#### `agents/scorer.py`
- Add `social` as 4th scoring dimension
- Existing weight redistribution logic handles missing social data automatically
- Normalization: social_score already 0-1, no transformation needed

#### `agents/synthesizer.py`
- Update prompt to include social_insights (events, crowd levels, itinerary tips)
- LLM weaves social context into natural language recommendation

#### `agents/mock_agents.py`
- Add `mock_social_node()` with destination-aware preset insights
- Follows existing mock_data patterns

#### `mock_data.py`
- Add `get_mock_social_insights(destination, start_date)` function
- Destination-aware presets for popular cities (crowd levels, events, tips)

#### `config.py`
- Add: `tavily_api_key`, `reddit_client_id`, `reddit_client_secret`, `reddit_user_agent`, `social_extraction_model`

#### `db.py`
- Add `social_insights` table and `save_social()` / `get_social()` methods

#### `app.py` (Streamlit UI)
- Sidebar: New "Social insights" priority slider
- Pipeline visualization: Add "🔍 Social" node in parallel row
- Results: New "Social Insights" tab with crowd level, events, itinerary tips, source links
- Bar chart: Add social score dimension

#### `.env.example`
- Add Tavily and Reddit credential placeholders

#### `requirements.txt`
- Add `tavily-python`, `praw`

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS social_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    destination TEXT NOT NULL,
    month INTEGER NOT NULL,
    timing_score REAL,
    crowd_level TEXT,
    events TEXT,          -- JSON array
    itinerary_tips TEXT,  -- JSON array
    sentiment TEXT,
    source TEXT,          -- 'tavily' | 'reddit' | 'both'
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Fallback query: match by destination + month (±1 month tolerance).

## Error Handling

Follows existing patterns exactly:
- Errors accumulated in `state["errors"]`, never raised
- If Tavily fails: try Reddit only
- If Reddit fails: try Tavily only
- If both fail: try SQLite historical fallback
- If no data at all: log error, return empty social_data (scorer redistributes weight)
- LLM extraction failure: return raw "no insights available" with score 0.5 (neutral)

## Testing

- Unit tests for `tavily_client.py` and `reddit_client.py` (mock HTTP responses)
- Unit test for `social_node` (mock both services + LLM call)
- Unit test for `mock_social_node`
- Integration test: scorer with 4 dimensions
- Integration test: synthesizer with social insights in prompt

## Dependencies

```
tavily-python>=0.3.0
praw>=7.7.0
```
