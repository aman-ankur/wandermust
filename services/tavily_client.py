import logging
from tavily import TavilyClient
from config import settings

logger = logging.getLogger("wandermust.tavily")
_client = None

def _get_client():
    global _client
    if _client is None:
        logger.info("Initializing Tavily client")
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client

def search_destination(destination: str, month: str) -> list:
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
            logger.info(f"Tavily search: {query[:80]}...")
            response = client.search(query, max_results=5)
            n = len(response.get("results", []))
            logger.info(f"Tavily returned {n} results")
            for r in response.get("results", []):
                all_results.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0.0),
                })
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            continue
    logger.info(f"Tavily total: {len(all_results)} results for {destination}")
    return all_results
