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
        _mock_submission("Tokyo in July", "Great trip!", 42, "https://reddit.com/r/travel/1", ["Go to Shibuya — it is amazing"]),
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
