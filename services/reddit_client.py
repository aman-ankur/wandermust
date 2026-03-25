from __future__ import annotations
import logging
import praw
from config import settings

logger = logging.getLogger("wandermust.reddit")
_reddit = None

def _get_reddit():
    global _reddit
    if _reddit is None:
        logger.info("Initializing Reddit (PRAW) client")
        _reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
    return _reddit

def search_subreddits(
    destination: str,
    subreddits: list | None = None,
    time_filter: str = "year",
    limit: int = 10,
) -> list:
    """Search Reddit for travel posts about a destination.

    Returns list of {title, body, top_comments, url, subreddit, score} dicts.
    Returns empty list if Reddit credentials are not configured.
    """
    if not settings.reddit_client_id or settings.reddit_client_id == "your_reddit_client_id":
        logger.warning("Reddit credentials not configured — skipping Reddit search")
        return []
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
