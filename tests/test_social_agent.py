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
