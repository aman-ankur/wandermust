# Social Media Intelligence Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a social media agent that crawls Twitter/Reddit via Tavily + PRAW to extract travel timing insights and itinerary suggestions, fully integrated into the scoring pipeline.

**Architecture:** New `social_node` agent runs in parallel with weather/flights/hotels agents. It fetches social content from Tavily (Twitter/web) and Reddit (PRAW), extracts structured insights via Haiku 3.5 LLM call, and returns a per-window `social_score` plus destination-level `social_insights`. The scorer uses social as a 4th dimension; the synthesizer weaves insights into recommendations.

**Tech Stack:** Python 3.9+, LangGraph, Tavily (`tavily-python`), Reddit (`praw`), OpenRouter (Haiku 3.5), SQLite, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-social-media-agent-design.md`

---

### File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `services/tavily_client.py` | Tavily search API wrapper |
| Create | `services/reddit_client.py` | PRAW Reddit search wrapper |
| Create | `agents/social.py` | Social media intelligence agent node |
| Create | `tests/test_tavily_client.py` | Tests for Tavily client |
| Create | `tests/test_reddit_client.py` | Tests for Reddit client |
| Create | `tests/test_social_agent.py` | Tests for social agent node |
| Modify | `models.py` | Add `social_data`, `social_insights` to TravelState; `social_score` to RankedWindow |
| Modify | `config.py` | Add Tavily, Reddit, social model config |
| Modify | `db.py` | Add `social_insights` table + save/get methods |
| Modify | `agents/scorer.py` | Add social as 4th scoring dimension |
| Modify | `agents/synthesizer.py` | Include social insights in prompt |
| Modify | `agents/mock_agents.py` | Add `mock_social_node` |
| Modify | `mock_data.py` | Add `get_mock_social_insights` |
| Modify | `graph.py` | Wire social node into parallel fan-out |
| Modify | `app.py` | Add Social to pipeline viz, slider, results tab |
| Modify | `requirements.txt` | Add `tavily-python`, `praw` |
| Modify | `.env.example` | Add Tavily/Reddit credential placeholders |
| Modify | `tests/test_mock_agents.py` | Add mock social node tests |
| Modify | `tests/test_scorer.py` | Add 4-dimension scoring tests |
| Modify | `tests/test_graph.py` | Wire social node in e2e test mocks |

---

### Task 1: Config & Dependencies

**Files:**
- Modify: `config.py:3-18`
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: Add config fields**

In `config.py`, add these fields to the `Settings` class after line 13 (`api_max_retries`):

```python
    tavily_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "wandermust-travel-optimizer/1.0"
    social_extraction_model: str = "anthropic/claude-haiku-3.5"
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Append to `requirements.txt`:

```
tavily-python>=0.3.0
praw>=7.7.0
```

- [ ] **Step 3: Update .env.example**

Append to `.env.example`:

```
TAVILY_API_KEY=your_tavily_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

- [ ] **Step 4: Install dependencies**

Run: `pip install tavily-python>=0.3.0 praw>=7.7.0`

- [ ] **Step 5: Verify existing tests still pass**

Run: `pytest tests/ -v --tb=short`
Expected: All 41 existing tests pass (no regressions from config changes).

- [ ] **Step 6: Commit**

```bash
git add config.py requirements.txt .env.example
git commit -m "feat: add Tavily, Reddit, and social model config"
```

---

### Task 2: Models — State & RankedWindow Updates

**Files:**
- Modify: `models.py:46-60`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing test for new state fields**

Append to `tests/test_models.py`:

```python
from models import TravelState, RankedWindow

def test_travel_state_accepts_social_fields():
    state: TravelState = {
        "destination": "Tokyo",
        "social_data": [{"window_start": "2026-07-01", "social_score": 0.8}],
        "social_insights": [{"destination": "Tokyo", "timing_score": 0.8, "crowd_level": "moderate"}],
        "errors": [],
    }
    assert state["social_data"][0]["social_score"] == 0.8
    assert state["social_insights"][0]["crowd_level"] == "moderate"

def test_ranked_window_has_social_score():
    rw = RankedWindow(
        window={"start": "2026-07-01", "end": "2026-07-07"},
        weather_score=0.8, flight_score=0.7, hotel_score=0.6,
        social_score=0.85, total_score=0.75,
    )
    assert rw.social_score == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_ranked_window_has_social_score -v`
Expected: FAIL — `RankedWindow` does not have `social_score` field.

- [ ] **Step 3: Add fields to models.py**

In `models.py`, add to `TravelState` (after `hotel_data` line 57):

```python
    social_data: List[dict]
    social_insights: List[dict]
```

In `RankedWindow` (after `hotel_score` line 40):

```python
    social_score: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add social_data, social_insights, and social_score to models"
```

---

### Task 3: Database — Social Insights Table

**Files:**
- Modify: `db.py:9-29` (add table), `db.py:70` (add methods)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_db.py`:

```python
def test_save_and_get_social(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    db.save_social(
        destination="Tokyo", month=7, timing_score=0.85,
        crowd_level="moderate", events='[{"name":"Tanabata","period":"July 7"}]',
        itinerary_tips='[{"tip":"Visit Meiji Shrine early","source":"reddit"}]',
        sentiment="highly recommended", source="both",
    )
    result = db.get_social("Tokyo", 7)
    assert result is not None
    assert result["timing_score"] == 0.85
    assert result["crowd_level"] == "moderate"

def test_get_social_tolerance(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    db.save_social("Tokyo", 7, 0.85, "moderate", "[]", "[]", "good", "both")
    result = db.get_social("Tokyo", 8, tolerance_months=1)
    assert result is not None

def test_get_social_missing(tmp_path):
    db = HistoryDB(str(tmp_path / "test.db"))
    result = db.get_social("Tokyo", 7)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py::test_save_and_get_social -v`
Expected: FAIL — `HistoryDB` has no `save_social` method.

- [ ] **Step 3: Add social_insights table and methods to db.py**

In `db.py` `_create_tables`, add to the `executescript` string (after the hotel_prices table):

```sql
            CREATE TABLE IF NOT EXISTS social_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination TEXT NOT NULL,
                month INTEGER NOT NULL,
                timing_score REAL,
                crowd_level TEXT,
                events TEXT,
                itinerary_tips TEXT,
                sentiment TEXT,
                source TEXT,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
```

Add methods after `get_hotel`:

```python
    def save_social(self, destination, month, timing_score, crowd_level,
                    events, itinerary_tips, sentiment, source):
        self._conn.execute(
            "INSERT INTO social_insights (destination, month, timing_score, crowd_level, "
            "events, itinerary_tips, sentiment, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (destination, month, timing_score, crowd_level, events, itinerary_tips, sentiment, source))
        self._conn.commit()

    def get_social(self, destination, month, tolerance_months=0):
        if tolerance_months > 0:
            row = self._conn.execute(
                "SELECT timing_score, crowd_level, events, itinerary_tips, sentiment, source, fetched_at "
                "FROM social_insights WHERE destination=? AND ABS(month - ?) <= ? "
                "ORDER BY ABS(month - ?) ASC, fetched_at DESC LIMIT 1",
                (destination, month, tolerance_months, month)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT timing_score, crowd_level, events, itinerary_tips, sentiment, source, fetched_at "
                "FROM social_insights WHERE destination=? AND month=? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (destination, month)).fetchone()
        return dict(row) if row else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: All pass (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add social_insights table with save/get methods"
```

---

### Task 4: Tavily Client Service

**Files:**
- Create: `services/tavily_client.py`
- Create: `tests/test_tavily_client.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_tavily_client.py`:

```python
from unittest.mock import patch, MagicMock
from services.tavily_client import search_destination

def test_search_destination_returns_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "Best time to visit Tokyo", "content": "Cherry blossom season is March-April.",
             "url": "https://example.com/tokyo", "score": 0.9},
            {"title": "Tokyo travel tips", "content": "Avoid Golden Week in May.",
             "url": "https://example.com/tokyo2", "score": 0.8},
        ]
    }
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Tokyo", "July")
    assert len(results) == 2
    assert results[0]["title"] == "Best time to visit Tokyo"
    assert "content" in results[0]
    assert "url" in results[0]

def test_search_destination_empty():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Xyzzyville", "January")
    assert results == []

def test_search_destination_api_error():
    mock_client = MagicMock()
    mock_client.search.side_effect = Exception("API error")
    with patch("services.tavily_client._get_client", return_value=mock_client):
        results = search_destination("Tokyo", "July")
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tavily_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement tavily client**

Create `services/tavily_client.py`:

```python
from tavily import TavilyClient
from config import settings

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client

def search_destination(destination: str, month: str) -> list[dict]:
    """Search Tavily for travel insights about a destination.

    Returns list of {title, content, url, score} dicts.
    """
    client = _get_client()
    queries = [
        f"best time to visit {destination} {month} site:twitter.com OR site:x.com",
        f"best time to visit {destination} travel tips itinerary",
    ]
    all_results = []
    for query in queries:
        try:
            response = client.search(query, max_results=5)
            for r in response.get("results", []):
                all_results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0.0),
                })
        except Exception:
            continue
    return all_results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tavily_client.py -v`
Expected: All 3 pass.

- [ ] **Step 5: Commit**

```bash
git add services/tavily_client.py tests/test_tavily_client.py
git commit -m "feat: add Tavily search client for social media content"
```

---

### Task 5: Reddit Client Service

**Files:**
- Create: `services/reddit_client.py`
- Create: `tests/test_reddit_client.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_reddit_client.py`:

```python
from unittest.mock import patch, MagicMock
from services.reddit_client import search_subreddits

def _mock_submission(title, selftext, score, url, comments_list):
    sub = MagicMock()
    sub.title = title
    sub.selftext = selftext
    sub.score = score
    sub.url = url
    sub.subreddit.display_name = "travel"
    comment_mocks = []
    for body in comments_list:
        c = MagicMock()
        c.body = body
        c.score = 10
        comment_mocks.append(c)
    sub.comments.list.return_value = comment_mocks
    return sub

def test_search_subreddits_returns_posts():
    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [
        _mock_submission("Tokyo in July", "Great trip!", 42, "https://reddit.com/r/travel/1", ["Go to Shibuya"]),
    ]
    mock_reddit.subreddit.return_value = mock_subreddit
    with patch("services.reddit_client._get_reddit", return_value=mock_reddit):
        results = search_subreddits("Tokyo")
    assert len(results) >= 1
    assert results[0]["title"] == "Tokyo in July"
    assert "top_comments" in results[0]
    assert len(results[0]["top_comments"]) > 0

def test_search_subreddits_api_error():
    mock_reddit = MagicMock()
    mock_reddit.subreddit.side_effect = Exception("Auth failed")
    with patch("services.reddit_client._get_reddit", return_value=mock_reddit):
        results = search_subreddits("Tokyo")
    assert results == []

def test_search_subreddits_empty():
    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = []
    mock_reddit.subreddit.return_value = mock_subreddit
    with patch("services.reddit_client._get_reddit", return_value=mock_reddit):
        results = search_subreddits("Xyzzyville")
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reddit_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement Reddit client**

Create `services/reddit_client.py`:

```python
import praw
from config import settings

_reddit = None

def _get_reddit():
    global _reddit
    if _reddit is None:
        _reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
    return _reddit

def search_subreddits(
    destination: str,
    subreddits: list[str] | None = None,
    time_filter: str = "year",
    limit: int = 10,
) -> list[dict]:
    """Search Reddit for travel posts about a destination.

    Returns list of {title, body, top_comments, url, subreddit, score} dicts.
    """
    if subreddits is None:
        subreddits = ["travel", "solotravel", "TravelHacks"]
    reddit = _get_reddit()
    all_posts = []
    queries = [f"{destination} best time", f"{destination} itinerary"]
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for query in queries:
                for submission in subreddit.search(query, time_filter=time_filter, limit=limit):
                    top_comments = []
                    try:
                        for comment in submission.comments.list()[:5]:
                            if hasattr(comment, "body") and len(comment.body) > 20:
                                top_comments.append(comment.body)
                    except Exception:
                        pass
                    all_posts.append({
                        "title": submission.title,
                        "body": submission.selftext[:1000],
                        "top_comments": top_comments[:3],
                        "url": submission.url,
                        "subreddit": submission.subreddit.display_name,
                        "score": submission.score,
                    })
        except Exception:
            continue
    return all_posts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reddit_client.py -v`
Expected: All 3 pass.

- [ ] **Step 5: Commit**

```bash
git add services/reddit_client.py tests/test_reddit_client.py
git commit -m "feat: add Reddit PRAW client for travel subreddit search"
```

---

### Task 6: Social Agent Node

**Files:**
- Create: `agents/social.py`
- Create: `tests/test_social_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_social_agent.py`:

```python
import json
from unittest.mock import patch, MagicMock
from agents.social import social_node

def _make_state():
    return {
        "destination": "Tokyo, Japan",
        "origin": "Bangalore",
        "candidate_windows": [
            {"start": "2026-07-01", "end": "2026-07-07"},
            {"start": "2026-07-08", "end": "2026-07-14"},
        ],
        "errors": [],
    }

def _mock_llm_response():
    return json.dumps({
        "timing_score": 0.7,
        "crowd_level": "high",
        "events": [{"name": "Tanabata", "period": "July 7"}],
        "itinerary_tips": [
            {"tip": "Visit Senso-ji temple early morning", "source": "reddit"},
            {"tip": "Try street food in Ameyoko market", "source": "twitter"},
        ],
        "sentiment": "recommended",
        "best_months": [3, 4, 10, 11],
    })

def test_social_node_success():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=_mock_llm_response())
    with patch("agents.social.search_destination", return_value=[
            {"title": "Tokyo tips", "content": "Cherry blossoms in April", "url": "https://x.com/1", "score": 0.9}]), \
         patch("agents.social.search_subreddits", return_value=[
            {"title": "Tokyo July", "body": "Hot and humid", "top_comments": ["Go to Hakone"], "url": "https://reddit.com/1", "subreddit": "travel", "score": 50}]), \
         patch("agents.social._get_llm", return_value=mock_llm), \
         patch("agents.social.HistoryDB"):
        result = social_node(_make_state())
    assert "social_data" in result
    assert "social_insights" in result
    assert len(result["social_data"]) == 2  # one per window
    assert 0 <= result["social_data"][0]["social_score"] <= 1.0
    assert result["social_insights"][0]["crowd_level"] == "high"

def test_social_node_both_sources_fail():
    with patch("agents.social.search_destination", return_value=[]), \
         patch("agents.social.search_subreddits", return_value=[]), \
         patch("agents.social.HistoryDB") as mock_db:
        mock_db.return_value.get_social.return_value = None
        result = social_node(_make_state())
    assert result["social_data"] == []
    assert result["social_insights"] == []
    assert len(result["errors"]) > 0

def test_social_node_tavily_fails_reddit_ok():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=_mock_llm_response())
    with patch("agents.social.search_destination", return_value=[]), \
         patch("agents.social.search_subreddits", return_value=[
            {"title": "Tokyo", "body": "Nice in spring", "top_comments": [], "url": "https://reddit.com/1", "subreddit": "travel", "score": 30}]), \
         patch("agents.social._get_llm", return_value=mock_llm), \
         patch("agents.social.HistoryDB"):
        result = social_node(_make_state())
    assert len(result["social_data"]) == 2
    assert len(result["social_insights"]) > 0

def test_social_node_llm_fails_returns_neutral():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM unavailable")
    with patch("agents.social.search_destination", return_value=[
            {"title": "Tokyo", "content": "Visit in spring", "url": "https://x.com/1", "score": 0.8}]), \
         patch("agents.social.search_subreddits", return_value=[]), \
         patch("agents.social._get_llm", return_value=mock_llm), \
         patch("agents.social.HistoryDB"):
        result = social_node(_make_state())
    # Should still return data with neutral score
    assert "social_data" in result
    for d in result["social_data"]:
        assert d["social_score"] == 0.5

def test_social_node_db_fallback():
    with patch("agents.social.search_destination", return_value=[]), \
         patch("agents.social.search_subreddits", return_value=[]), \
         patch("agents.social.HistoryDB") as mock_db_cls:
        mock_db = mock_db_cls.return_value
        mock_db.get_social.return_value = {
            "timing_score": 0.8, "crowd_level": "low",
            "events": "[]", "itinerary_tips": "[]",
            "sentiment": "recommended", "source": "both",
        }
        result = social_node(_make_state())
    assert len(result["social_data"]) == 2
    assert result["social_data"][0]["social_score"] == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_social_agent.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement social agent**

Create `agents/social.py`:

```python
import json
from datetime import date
from langchain_openai import ChatOpenAI
from models import TravelState
from config import settings
from db import HistoryDB
from services.tavily_client import search_destination
from services.reddit_client import search_subreddits

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.social_extraction_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _llm

EXTRACTION_PROMPT = """You are a travel intelligence extractor. Analyze the following social media posts and comments about traveling to {destination}. Extract structured insights.

Return ONLY valid JSON with this exact schema:
{{
    "timing_score": <float 0-1, how recommended this destination is based on social sentiment>,
    "crowd_level": "<low|moderate|high|extreme>",
    "events": [{{ "name": "<event name>", "period": "<when>" }}],
    "itinerary_tips": [{{ "tip": "<actionable tip>", "source": "<reddit|twitter|web>" }}],
    "sentiment": "<overall traveler sentiment summary in 2-3 words>",
    "best_months": [<list of best month numbers 1-12>]
}}

Social media content:
{content}
"""

def social_node(state: TravelState) -> dict:
    """Social media intelligence agent.

    Fetches from Tavily + Reddit, extracts insights via LLM,
    returns per-window social_score and destination-level social_insights.
    """
    errors = list(state.get("errors", []))
    destination = state["destination"]
    windows = state["candidate_windows"]
    sample_month = date.fromisoformat(windows[0]["start"]).strftime("%B") if windows else "summer"

    # Step 1: Fetch from both sources
    tavily_results = search_destination(destination, sample_month)
    reddit_results = search_subreddits(destination)

    # Step 2: If no content from either source, try DB fallback
    if not tavily_results and not reddit_results:
        db = HistoryDB(settings.db_path)
        month_num = date.fromisoformat(windows[0]["start"]).month if windows else 7
        cached = db.get_social(destination, month_num, tolerance_months=1)
        if cached:
            score = cached["timing_score"]
            social_data = [{"window_start": w["start"], "window_end": w["end"],
                            "social_score": score} for w in windows]
            insights = [{
                "destination": destination,
                "timing_score": cached["timing_score"],
                "crowd_level": cached["crowd_level"],
                "events": json.loads(cached["events"]) if isinstance(cached["events"], str) else cached["events"],
                "itinerary_tips": json.loads(cached["itinerary_tips"]) if isinstance(cached["itinerary_tips"], str) else cached["itinerary_tips"],
                "sentiment": cached["sentiment"],
                "sources": [],
                "is_historical": True,
            }]
            return {"social_data": social_data, "social_insights": insights, "errors": errors}
        errors.append(f"Social: no data from Tavily, Reddit, or cache for {destination}")
        return {"social_data": [], "social_insights": [], "errors": errors}

    # Step 3: Combine content for LLM extraction
    content_parts = []
    sources = []
    for r in tavily_results:
        content_parts.append(f"[Twitter/Web] {r['title']}: {r['content']}")
        sources.append({"url": r["url"], "platform": "twitter", "title": r["title"]})
    for r in reddit_results:
        body = f"[Reddit r/{r['subreddit']}] {r['title']}: {r['body']}"
        if r["top_comments"]:
            body += "\nTop comments: " + " | ".join(r["top_comments"][:3])
        content_parts.append(body)
        sources.append({"url": r["url"], "platform": "reddit", "title": r["title"]})

    combined_content = "\n\n".join(content_parts[:15])  # Cap at 15 items

    # Step 4: LLM extraction
    try:
        llm = _get_llm()
        prompt = EXTRACTION_PROMPT.format(destination=destination, content=combined_content)
        response = llm.invoke(prompt)
        extracted = json.loads(response.content)
    except Exception as e:
        errors.append(f"Social: LLM extraction failed — {e}")
        # Neutral fallback
        social_data = [{"window_start": w["start"], "window_end": w["end"],
                        "social_score": 0.5} for w in windows]
        return {"social_data": social_data, "social_insights": [], "errors": errors}

    # Step 5: Compute per-window scores
    base_score = extracted.get("timing_score", 0.5)
    best_months = extracted.get("best_months", [])
    social_data = []
    for w in windows:
        window_month = date.fromisoformat(w["start"]).month
        if best_months and window_month in best_months:
            score = min(1.0, base_score + 0.15)
        elif best_months and window_month not in best_months:
            score = max(0.0, base_score - 0.1)
        else:
            score = base_score
        social_data.append({
            "window_start": w["start"],
            "window_end": w["end"],
            "social_score": round(score, 3),
        })

    insights = [{
        "destination": destination,
        "timing_score": base_score,
        "crowd_level": extracted.get("crowd_level", "unknown"),
        "events": extracted.get("events", []),
        "itinerary_tips": extracted.get("itinerary_tips", []),
        "sentiment": extracted.get("sentiment", "unknown"),
        "sources": sources[:10],
    }]

    # Step 6: Save to DB
    try:
        db = HistoryDB(settings.db_path)
        month_num = date.fromisoformat(windows[0]["start"]).month
        source_type = "both" if tavily_results and reddit_results else ("tavily" if tavily_results else "reddit")
        db.save_social(
            destination=destination, month=month_num,
            timing_score=base_score,
            crowd_level=extracted.get("crowd_level", "unknown"),
            events=json.dumps(extracted.get("events", [])),
            itinerary_tips=json.dumps(extracted.get("itinerary_tips", [])),
            sentiment=extracted.get("sentiment", "unknown"),
            source=source_type,
        )
    except Exception as e:
        errors.append(f"Social: DB save failed — {e}")

    return {"social_data": social_data, "social_insights": insights, "errors": errors}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_social_agent.py -v`
Expected: All 5 pass.

- [ ] **Step 5: Commit**

```bash
git add agents/social.py tests/test_social_agent.py
git commit -m "feat: add social media intelligence agent with Tavily + Reddit + LLM extraction"
```

---

### Task 7: Scorer — Add Social as 4th Dimension

**Files:**
- Modify: `agents/scorer.py:11-61`
- Modify: `tests/test_scorer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_scorer.py`:

```python
def test_scorer_with_social_dimension():
    state = {
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "weather_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.9},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "score": 0.5}],
        "flight_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "min_price": 25000}],
        "hotel_data": [
            {"window": {"start": "2026-07-01", "end": "2026-07-07"}, "avg_nightly": 8000},
            {"window": {"start": "2026-07-08", "end": "2026-07-14"}, "avg_nightly": 6000}],
        "social_data": [
            {"window_start": "2026-07-01", "window_end": "2026-07-07", "social_score": 0.85},
            {"window_start": "2026-07-08", "window_end": "2026-07-14", "social_score": 0.6}],
        "errors": []}
    result = scorer_node(state)
    assert result["ranked_windows"][0]["total_score"] >= result["ranked_windows"][1]["total_score"]
    # social_score should be present in ranked output
    assert "social_score" in result["ranked_windows"][0]
    assert result["ranked_windows"][0]["social_score"] >= 0

def test_scorer_social_missing_reweights():
    """Without social_data, scorer should redistribute weights."""
    state = {
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "weather_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "score": 0.8}],
        "flight_data": [{"window": {"start": "2026-07-01", "end": "2026-07-07"}, "min_price": 15000}],
        "hotel_data": [],
        "social_data": [],
        "errors": []}
    result = scorer_node(state)
    assert len(result["ranked_windows"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scorer.py::test_scorer_with_social_dimension -v`
Expected: FAIL — no `social_score` in output.

- [ ] **Step 3: Update scorer to include social dimension**

Replace the entire `scorer_node` function in `agents/scorer.py`:

```python
def scorer_node(state: TravelState) -> dict:
    priorities = state.get("priorities", {"weather": 0.4, "flights": 0.3, "hotels": 0.3})
    weather_data = state.get("weather_data", [])
    flight_data = state.get("flight_data", [])
    hotel_data = state.get("hotel_data", [])
    social_data = state.get("social_data", [])

    weather_by_start = {d["window"]["start"]: d for d in weather_data}
    flight_by_start = {d["window"]["start"]: d for d in flight_data}
    hotel_by_start = {d["window"]["start"]: d for d in hotel_data}
    social_by_start = {d["window_start"]: d for d in social_data}

    all_starts = sorted(set(
        [d["window"]["start"] for d in weather_data] +
        [d["window"]["start"] for d in flight_data] +
        [d["window"]["start"] for d in hotel_data] +
        [d["window_start"] for d in social_data]))

    # Normalize prices (lower = better)
    flight_starts = [s for s in all_starts if s in flight_by_start]
    hotel_starts = [s for s in all_starts if s in hotel_by_start]
    flight_norm = dict(zip(flight_starts,
        normalize_scores([flight_by_start[s]["min_price"] for s in flight_starts], lower_is_better=True))) if flight_starts else {}
    hotel_norm = dict(zip(hotel_starts,
        normalize_scores([hotel_by_start[s]["avg_nightly"] for s in hotel_starts], lower_is_better=True))) if hotel_starts else {}

    # Reweight if dimension missing
    active = {}
    if weather_data: active["weather"] = priorities.get("weather", 0.35)
    if flight_data: active["flights"] = priorities.get("flights", 0.25)
    if hotel_data: active["hotels"] = priorities.get("hotels", 0.25)
    if social_data: active["social"] = priorities.get("social", 0.15)
    total_w = sum(active.values()) or 1.0
    norm_w = {k: v/total_w for k, v in active.items()}

    ranked = []
    for start in all_starts:
        ws = weather_by_start.get(start, {}).get("score", 0.0)
        fs = flight_norm.get(start, 0.0)
        hs = hotel_norm.get(start, 0.0)
        ss = social_by_start.get(start, {}).get("social_score", 0.0)
        total = (ws * norm_w.get("weather", 0) + fs * norm_w.get("flights", 0) +
                 hs * norm_w.get("hotels", 0) + ss * norm_w.get("social", 0))
        window = (weather_by_start.get(start) or flight_by_start.get(start) or hotel_by_start.get(start, {}))
        ranked.append({
            "window": window.get("window", {"start": start, "end": ""}),
            "weather_score": round(ws, 3), "flight_score": round(fs, 3),
            "hotel_score": round(hs, 3), "social_score": round(ss, 3),
            "total_score": round(total, 3),
            "estimated_flight_cost": flight_by_start.get(start, {}).get("min_price", 0.0),
            "estimated_hotel_cost": hotel_by_start.get(start, {}).get("avg_nightly", 0.0),
            "has_historical_data": any([
                weather_by_start.get(start, {}).get("is_historical", False),
                flight_by_start.get(start, {}).get("is_historical", False),
                hotel_by_start.get(start, {}).get("is_historical", False)]),
        })
    ranked.sort(key=lambda x: x["total_score"], reverse=True)
    return {"ranked_windows": ranked, "errors": state.get("errors", [])}
```

- [ ] **Step 4: Run all scorer tests**

Run: `pytest tests/test_scorer.py -v`
Expected: All pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add agents/scorer.py tests/test_scorer.py
git commit -m "feat: add social as 4th scoring dimension in scorer"
```

---

### Task 8: Synthesizer — Enrich with Social Insights

**Files:**
- Modify: `agents/synthesizer.py:23-38`
- Test: `tests/test_synthesizer.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_synthesizer.py` (read the file first to see existing test patterns):

```python
def test_synthesizer_includes_social_insights():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Tokyo is best in spring. Crowds are moderate. Visit Senso-ji early.")
    state = {
        "destination": "Tokyo", "origin": "Bangalore",
        "ranked_windows": [{
            "window": {"start": "2026-04-01", "end": "2026-04-07"},
            "total_score": 0.9, "weather_score": 0.85, "flight_score": 0.8,
            "hotel_score": 0.7, "social_score": 0.9,
            "estimated_flight_cost": 30000, "estimated_hotel_cost": 8000,
        }],
        "social_insights": [{
            "destination": "Tokyo",
            "crowd_level": "moderate",
            "events": [{"name": "Cherry Blossom Festival", "period": "late March - mid April"}],
            "itinerary_tips": [{"tip": "Visit Senso-ji early morning", "source": "reddit"}],
            "sentiment": "highly recommended",
        }],
        "errors": [],
    }
    with patch("agents.synthesizer._get_llm", return_value=mock_llm):
        result = synthesizer_node(state)
    # Verify the prompt sent to LLM includes social insights
    call_args = mock_llm.invoke.call_args[0][0]
    assert "crowd" in call_args.lower() or "social" in call_args.lower()
    assert "Cherry Blossom" in call_args or "itinerary" in call_args.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesizer.py::test_synthesizer_includes_social_insights -v`
Expected: FAIL — synthesizer prompt does not include social insights.

- [ ] **Step 3: Update synthesizer to include social insights**

In `agents/synthesizer.py`, update the `synthesizer_node` function. Replace the prompt construction (the `prompt = (f"You are...` block) with:

```python
    prompt = (f"You are a travel advisor. Based on these ranked travel windows for a trip from "
        f"{state.get('origin', 'your city')} to {state.get('destination', 'the destination')}, "
        f"write a concise recommendation (3-5 sentences) about which dates are best and why. "
        f"Mention weather, flight cost, and hotel cost. If data is estimated from history, note that.\n\n{data_summary}")

    # Enrich with social insights if available
    social_insights = state.get("social_insights", [])
    if social_insights:
        si = social_insights[0]
        social_section = "\n\nSocial media insights from travelers:"
        if si.get("crowd_level"):
            social_section += f"\n- Crowd level: {si['crowd_level']}"
        if si.get("sentiment"):
            social_section += f"\n- Traveler sentiment: {si['sentiment']}"
        for event in si.get("events", [])[:3]:
            social_section += f"\n- Event: {event.get('name', '')} ({event.get('period', '')})"
        for tip in si.get("itinerary_tips", [])[:5]:
            social_section += f"\n- Tip: {tip.get('tip', '')} (via {tip.get('source', 'social media')})"
        social_section += "\n\nIncorporate these social insights into your recommendation — mention crowd levels, events, and include 2-3 itinerary tips."
        prompt += social_section
```

- [ ] **Step 4: Run all synthesizer tests**

Run: `pytest tests/test_synthesizer.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add agents/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: enrich synthesizer prompt with social media insights"
```

---

### Task 9: Mock Agent & Mock Data for Demo Mode

**Files:**
- Modify: `mock_data.py`
- Modify: `agents/mock_agents.py`
- Modify: `tests/test_mock_agents.py`

- [ ] **Step 1: Write failing test for mock social data**

Append to `tests/test_mock_agents.py`:

```python
from mock_data import get_mock_social_insights
from agents.mock_agents import mock_social_node

def test_mock_social_insights_known():
    data = get_mock_social_insights("Tokyo", "2026-07-01")
    assert "timing_score" in data
    assert 0.0 <= data["timing_score"] <= 1.0
    assert data["crowd_level"] in ("low", "moderate", "high", "extreme")
    assert isinstance(data["events"], list)
    assert isinstance(data["itinerary_tips"], list)

def test_mock_social_insights_unknown():
    data = get_mock_social_insights("Xyzzyville", "2026-07-01")
    assert 0.0 <= data["timing_score"] <= 1.0

def test_mock_social_node_shape():
    state = {
        "destination": "Tokyo",
        "candidate_windows": [
            {"start": "2026-07-01", "end": "2026-07-07"},
            {"start": "2026-07-08", "end": "2026-07-14"},
        ],
        "errors": [],
    }
    result = mock_social_node(state)
    assert "social_data" in result
    assert "social_insights" in result
    assert len(result["social_data"]) == 2
    for d in result["social_data"]:
        assert 0.0 <= d["social_score"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mock_agents.py::test_mock_social_insights_known -v`
Expected: FAIL — `get_mock_social_insights` not found.

- [ ] **Step 3: Add mock social data to mock_data.py**

Append to `mock_data.py` (after the `PRESETS` dict, add a new `SOCIAL_PRESETS` dict and the `get_mock_social_insights` function):

```python
SOCIAL_PRESETS = {
    "tokyo": {
        "timing_score": 0.85,
        "crowd_level": "high",
        "events": [
            {"name": "Cherry Blossom Season", "period": "late March - mid April"},
            {"name": "Tanabata Festival", "period": "July 7"},
            {"name": "Autumn Leaves", "period": "mid November - early December"},
        ],
        "itinerary_tips": [
            {"tip": "Visit Fushimi Inari at sunrise to avoid crowds", "source": "reddit"},
            {"tip": "Get a 7-day JR Pass for day trips to Kyoto and Osaka", "source": "reddit"},
            {"tip": "Tsukiji outer market is better than Toyosu for tourists", "source": "twitter"},
        ],
        "sentiment": "highly recommended",
        "best_months": [3, 4, 10, 11],
    },
    "paris": {
        "timing_score": 0.80,
        "crowd_level": "high",
        "events": [
            {"name": "Bastille Day", "period": "July 14"},
            {"name": "Paris Fashion Week", "period": "late September"},
        ],
        "itinerary_tips": [
            {"tip": "Skip the Eiffel Tower queue — book Montparnasse Tower instead", "source": "reddit"},
            {"tip": "Walk along Canal Saint-Martin for local vibe", "source": "twitter"},
        ],
        "sentiment": "highly recommended",
        "best_months": [4, 5, 6, 9, 10],
    },
    "bangkok": {
        "timing_score": 0.75,
        "crowd_level": "moderate",
        "events": [
            {"name": "Songkran Water Festival", "period": "April 13-15"},
            {"name": "Loy Krathong", "period": "November full moon"},
        ],
        "itinerary_tips": [
            {"tip": "Take the Chao Phraya Express boat instead of taxis", "source": "reddit"},
            {"tip": "Visit temples before 9am to beat the heat", "source": "reddit"},
        ],
        "sentiment": "recommended",
        "best_months": [11, 12, 1, 2],
    },
    "bali": {
        "timing_score": 0.80,
        "crowd_level": "moderate",
        "events": [
            {"name": "Nyepi (Day of Silence)", "period": "March"},
            {"name": "Galungan Festival", "period": "varies"},
        ],
        "itinerary_tips": [
            {"tip": "Rent a scooter — it's the best way to explore", "source": "reddit"},
            {"tip": "Uluwatu temple sunset is unmissable", "source": "twitter"},
        ],
        "sentiment": "highly recommended",
        "best_months": [5, 6, 7, 8, 9],
    },
    "dubai": {
        "timing_score": 0.70,
        "crowd_level": "moderate",
        "events": [
            {"name": "Dubai Shopping Festival", "period": "January - February"},
            {"name": "Dubai Food Festival", "period": "February - March"},
        ],
        "itinerary_tips": [
            {"tip": "Visit the desert safari at sunset, not midday", "source": "reddit"},
            {"tip": "Friday brunch is a Dubai institution — book ahead", "source": "twitter"},
        ],
        "sentiment": "recommended",
        "best_months": [11, 12, 1, 2, 3],
    },
}


def get_mock_social_insights(destination: str, start_date: str) -> dict:
    """Generate mock social media insights for a destination."""
    month = date.fromisoformat(start_date).month
    preset = None
    dest_lower = destination.lower()
    for key, data in SOCIAL_PRESETS.items():
        if key in dest_lower:
            preset = data
            break

    if preset:
        timing_score = preset["timing_score"]
        best_months = preset.get("best_months", [])
        if best_months and month in best_months:
            timing_score = min(1.0, timing_score + 0.1)
        elif best_months:
            timing_score = max(0.0, timing_score - 0.1)
        return {
            "timing_score": round(timing_score, 2),
            "crowd_level": preset["crowd_level"],
            "events": preset["events"],
            "itinerary_tips": preset["itinerary_tips"],
            "sentiment": preset["sentiment"],
            "best_months": best_months,
        }
    else:
        seed = _hash_seed(destination)
        rng = random.Random(seed)
        return {
            "timing_score": round(rng.uniform(0.4, 0.8), 2),
            "crowd_level": rng.choice(["low", "moderate", "high"]),
            "events": [],
            "itinerary_tips": [
                {"tip": f"Explore local markets in {destination}", "source": "reddit"},
            ],
            "sentiment": "recommended",
            "best_months": [],
        }
```

- [ ] **Step 4: Add mock social agent node**

Append to `agents/mock_agents.py`:

First add the import at the top (with existing imports from mock_data):

```python
from mock_data import (
    get_mock_weather,
    get_mock_flight_price,
    get_mock_hotel_price,
    get_mock_recommendation,
    get_mock_social_insights,
)
```

Then add the function at the bottom:

```python
def mock_social_node(state: TravelState) -> dict:
    """Mock social agent — returns destination-aware simulated insights."""
    destination = state["destination"]
    windows = state["candidate_windows"]
    insights_data = get_mock_social_insights(destination, windows[0]["start"])
    social_data = []
    for w in windows:
        month = __import__("datetime").date.fromisoformat(w["start"]).month
        score = insights_data["timing_score"]
        best = insights_data.get("best_months", [])
        if best and month in best:
            score = min(1.0, score + 0.1)
        elif best:
            score = max(0.0, score - 0.1)
        social_data.append({
            "window_start": w["start"],
            "window_end": w["end"],
            "social_score": round(score, 3),
        })
    social_insights = [{
        "destination": destination,
        "timing_score": insights_data["timing_score"],
        "crowd_level": insights_data["crowd_level"],
        "events": insights_data["events"],
        "itinerary_tips": insights_data["itinerary_tips"],
        "sentiment": insights_data["sentiment"],
        "sources": [],
    }]
    return {"social_data": social_data, "social_insights": social_insights, "errors": []}
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_mock_agents.py -v`
Expected: All pass (existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add mock_data.py agents/mock_agents.py tests/test_mock_agents.py
git commit -m "feat: add mock social data and mock social agent for demo mode"
```

---

### Task 10: Graph — Wire Social Agent into Pipeline

**Files:**
- Modify: `graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Write failing test**

Update `tests/test_graph.py`. Add this test:

```python
def test_demo_graph_includes_social():
    """Demo graph should include social node and produce social data."""
    g = build_graph(demo=True)
    result = g.invoke({
        "destination": "Tokyo",
        "origin": "Bangalore",
        "date_range": ("2026-07-01", "2026-07-28"),
        "duration_days": 7,
        "num_travelers": 1,
        "budget_max": None,
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "errors": [],
    })
    assert "social_data" in result
    assert len(result["social_data"]) > 0
    assert "social_insights" in result
    assert len(result["social_insights"]) > 0
    # Ranked windows should have social_score
    assert "social_score" in result["ranked_windows"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py::test_demo_graph_includes_social -v`
Expected: FAIL — no social_data in result.

- [ ] **Step 3: Update graph.py to wire social node**

Replace `graph.py` entirely:

```python
from langgraph.graph import StateGraph, END
from models import TravelState
from agents.supervisor import supervisor_node
from agents.weather import weather_node
from agents.flights import flights_node
from agents.hotels import hotels_node
from agents.social import social_node
from agents.scorer import scorer_node
from agents.synthesizer import synthesizer_node

def build_graph(demo: bool = False):
    if demo:
        from agents.mock_agents import (
            mock_weather_node, mock_flights_node,
            mock_hotels_node, mock_synthesizer_node,
            mock_social_node,
        )
        weather_fn = mock_weather_node
        flights_fn = mock_flights_node
        hotels_fn = mock_hotels_node
        social_fn = mock_social_node
        synth_fn = mock_synthesizer_node
    else:
        weather_fn = weather_node
        flights_fn = flights_node
        hotels_fn = hotels_node
        social_fn = social_node
        synth_fn = synthesizer_node

    graph = StateGraph(TravelState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather", weather_fn)
    graph.add_node("flights", flights_fn)
    graph.add_node("hotels", hotels_fn)
    graph.add_node("social", social_fn)
    graph.add_node("scorer", scorer_node)
    graph.add_node("synthesizer", synth_fn)

    graph.set_entry_point("supervisor")
    # Fan-out: supervisor → 4 data agents in parallel
    graph.add_edge("supervisor", "weather")
    graph.add_edge("supervisor", "flights")
    graph.add_edge("supervisor", "hotels")
    graph.add_edge("supervisor", "social")
    # Fan-in: all 4 → scorer
    graph.add_edge("weather", "scorer")
    graph.add_edge("flights", "scorer")
    graph.add_edge("hotels", "scorer")
    graph.add_edge("social", "scorer")
    # Sequential: scorer → synthesizer → end
    graph.add_edge("scorer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
```

- [ ] **Step 4: Update the mocked e2e test**

In `tests/test_graph.py`, update `test_graph_e2e_mocked` to also mock the social agent. Add these patches inside the `with` block:

```python
         patch("agents.social.search_destination", return_value=[]), \
         patch("agents.social.search_subreddits", return_value=[]), \
         patch("agents.social.HistoryDB") as mock_social_db, \
```

And before the `result = build_graph().invoke(...)` line, add:

```python
        mock_social_db.return_value.get_social.return_value = None
```

Also update the state's priorities to include social:

```python
            "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
```

- [ ] **Step 5: Run all graph tests**

Run: `pytest tests/test_graph.py -v`
Expected: All pass.

- [ ] **Step 6: Run ALL tests to check for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass. Some existing tests may need minor updates if they check exact field counts in ranked_windows — fix any that fail due to the new `social_score` field.

- [ ] **Step 7: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: wire social agent into LangGraph pipeline as 4th parallel agent"
```

---

### Task 11: Streamlit UI — Pipeline Viz, Slider, Results Tab

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add Social to pipeline constants**

At the top of `app.py`, update the pipeline steps (line 12-14):

```python
PIPELINE_STEPS = ["Supervisor", "Weather", "Flights", "Hotels", "Social", "Scorer", "Synthesizer"]
STEP_ICONS = {"Supervisor": "📋", "Weather": "🌤️", "Flights": "✈️",
              "Hotels": "🏨", "Social": "🔍", "Scorer": "📊", "Synthesizer": "✍️"}
```

- [ ] **Step 2: Add social priority slider in sidebar**

After the hotels slider (line 50), add:

```python
    w_social = st.slider("Social insights", 0.0, 1.0, 0.15, 0.05)
```

- [ ] **Step 3: Update state construction**

Update the `total_w` and `priorities` dict (around line 58-64) to include social:

```python
        total_w = w_weather + w_flights + w_hotels + w_social or 1.0
        state = {
            ...
            "priorities": {"weather": w_weather/total_w, "flights": w_flights/total_w,
                          "hotels": w_hotels/total_w, "social": w_social/total_w},
            ...
        }
```

- [ ] **Step 4: Add social agent import and execution**

After the hotels import block (around line 80-90), add `mock_social_node`/`social_node` import:

```python
        if demo_mode:
            from agents.mock_agents import (
                mock_weather_node as weather_fn,
                mock_flights_node as flights_fn,
                mock_hotels_node as hotels_fn,
                mock_social_node as social_fn,
                mock_synthesizer_node as synth_fn,
            )
        else:
            from agents.weather import weather_node as weather_fn
            from agents.flights import flights_node as flights_fn
            from agents.hotels import hotels_node as hotels_fn
            from agents.social import social_node as social_fn
            from agents.synthesizer import synthesizer_node as synth_fn
```

After the Hotels execution block (around line 134-145), add the Social execution block:

```python
        with pipeline_area.container():
            render_pipeline(active="Social", completed=completed)
        log("**Social** → Crawling Twitter & Reddit for insights...")
        t0 = time.time()
        soc_result = social_fn(state)
        state.update(soc_result)
        n_social = len(state.get("social_data", []))
        n_tips = len(state.get("social_insights", [{}])[0].get("itinerary_tips", [])) if state.get("social_insights") else 0
        log(f"**Social** → Got data for **{n_social}/{n_windows} windows**, **{n_tips} tips** ({time.time()-t0:.1f}s)")
        completed.add("Social")
        with log_area.container():
            with st.expander("🔍 Agent Execution Log", expanded=True):
                st.markdown("\n\n".join(log_lines))
```

- [ ] **Step 5: Add social score to results display**

Update the comparison tab (around line 199-206). Add social to the DataFrame and bar chart:

```python
            df = pd.DataFrame([{"Window": r["window"]["start"], "Weather": r["weather_score"],
                "Flights": r["flight_score"], "Hotels": r["hotel_score"],
                "Social": r.get("social_score", 0.0), "Total": r["total_score"],
                "Flight Cost": r["estimated_flight_cost"], "Hotel/Night": r["estimated_hotel_cost"]} for r in ranked]).set_index("Window")
            t1, t2, t3, t4 = st.tabs(["Scores", "Flight Prices", "Hotel Prices", "Social Insights"])
            with t1: st.bar_chart(df[["Weather", "Flights", "Hotels", "Social", "Total"]])
            with t2: st.line_chart(df["Flight Cost"])
            with t3: st.line_chart(df["Hotel/Night"])
            with t4:
                insights = state.get("social_insights", [])
                if insights:
                    si = insights[0]
                    st.subheader(f"Crowd Level: {si.get('crowd_level', 'unknown').title()}")
                    st.write(f"**Traveler sentiment:** {si.get('sentiment', 'N/A')}")
                    if si.get("events"):
                        st.subheader("Upcoming Events")
                        for event in si["events"]:
                            st.write(f"- **{event.get('name', '')}** — {event.get('period', '')}")
                    if si.get("itinerary_tips"):
                        st.subheader("Itinerary Tips from Travelers")
                        for tip in si["itinerary_tips"]:
                            st.write(f"- {tip.get('tip', '')} *(via {tip.get('source', 'social media')})*")
                    if si.get("sources"):
                        with st.expander("Sources"):
                            for src in si["sources"][:5]:
                                st.write(f"- [{src.get('title', 'Link')}]({src.get('url', '#')}) ({src.get('platform', '')})")
                else:
                    st.info("No social insights available for this search.")
```

- [ ] **Step 6: Test manually**

Run: `streamlit run app.py`
1. Toggle Demo Mode ON
2. Enter "Tokyo" as destination
3. Set Social insights slider to 0.15
4. Click "Find Best Time"
5. Verify: Social agent appears in pipeline viz, Social Insights tab shows events + tips

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add social agent to Streamlit UI with pipeline viz, slider, and insights tab"
```

---

### Task 12: Final Integration Test & Cleanup

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass. Fix any regressions.

- [ ] **Step 2: Run demo mode end-to-end**

Run: `streamlit run app.py`
Test with Demo Mode ON for: Tokyo, Paris, Bangkok, and an unknown destination ("Zurich").
Verify all 4 produce results with social data.

- [ ] **Step 3: Update existing demo graph test**

In `tests/test_mock_agents.py`, update `test_demo_graph_full_run` to include social priority and verify social output:

```python
def test_demo_graph_full_run():
    """Run the full graph in demo mode — no API calls needed."""
    g = build_graph(demo=True)
    result = g.invoke({
        "destination": "Tokyo",
        "origin": "Bangalore",
        "date_range": ("2026-07-01", "2026-07-28"),
        "duration_days": 7,
        "num_travelers": 1,
        "budget_max": None,
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "errors": [],
    })
    assert "ranked_windows" in result
    assert len(result["ranked_windows"]) > 0
    assert "recommendation" in result
    assert len(result["recommendation"]) > 0
    assert "social_data" in result
    assert len(result["social_data"]) > 0
    assert "social_insights" in result
    assert len(result.get("errors", [])) == 0
```

- [ ] **Step 4: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: update integration tests for social agent pipeline"
```
