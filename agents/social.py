import json
import logging
from datetime import date
from models import TravelState
from config import settings
from db import HistoryDB
from services.tavily_client import search_destination
from services.reddit_client import search_subreddits
from agents.llm_helper import get_llm, parse_json_response

logger = logging.getLogger("wandermust.social")

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(settings.social_extraction_model)
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
    windows = state.get("candidate_windows", [])
    sample_month = date.fromisoformat(windows[0]["start"]).strftime("%B") if windows else "summer"

    # Step 1: Fetch from both sources
    logger.info(f"Social agent: fetching data for {destination}, month={sample_month}")
    tavily_results = search_destination(destination, sample_month)
    logger.info(f"Social: Tavily returned {len(tavily_results)} results")
    reddit_results = search_subreddits(destination)
    logger.info(f"Social: Reddit returned {len(reddit_results)} results")

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
        logger.info(f"Social: calling LLM for extraction ({len(combined_content)} chars of content)")
        response = llm.invoke(prompt)
        logger.info(f"Social: LLM extraction complete, parsing JSON")
        extracted = parse_json_response(response.content)
        logger.info(f"Social: extracted timing_score={extracted.get('timing_score')}, crowd={extracted.get('crowd_level')}, {len(extracted.get('itinerary_tips', []))} tips")
    except Exception as e:
        logger.error(f"Social: LLM extraction failed — {e}")
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
