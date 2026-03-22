from langchain_openai import ChatOpenAI
from models import TravelState
from config import settings

_llm = None
def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=settings.openrouter_model,
            api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    return _llm

def format_ranked_data_fallback(ranked, top_n=3):
    lines = []
    for i, r in enumerate(ranked[:top_n], 1):
        w = r["window"]
        lines.append(f"#{i}: {w['start']} to {w['end']} (score: {r['total_score']:.2f}) — "
            f"Weather: {r['weather_score']:.2f}, Flights: ~{r['estimated_flight_cost']:.0f}, "
            f"Hotels: ~{r['estimated_hotel_cost']:.0f}/night"
            + (" [estimated from history]" if r.get("has_historical_data") else ""))
    return "\n".join(lines)

def synthesizer_node(state: TravelState) -> dict:
    ranked = state.get("ranked_windows", [])
    errors = list(state.get("errors", []))
    if not ranked:
        return {"recommendation": "No data available.", "errors": errors}
    data_summary = format_ranked_data_fallback(ranked[:5], top_n=5)
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

    try:
        response = _get_llm().invoke(prompt)
        return {"recommendation": response.content, "errors": errors}
    except Exception as e:
        errors.append(f"Synthesizer: LLM failed — {e}")
        return {"recommendation": data_summary, "errors": errors}
