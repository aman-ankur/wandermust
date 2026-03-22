"""Bridge function — converts discovery output to optimizer TravelState.

Takes chosen destination + trip_intent and populates a TravelState dict
ready for the existing optimizer pipeline.
"""
from datetime import date, timedelta
from models import DiscoveryState
from config import settings

MONTH_TO_DATE_RANGE = {
    "january": ("01-01", "01-31"), "february": ("02-01", "02-28"),
    "march": ("03-01", "03-31"), "april": ("04-01", "04-30"),
    "may": ("05-01", "05-31"), "june": ("06-01", "06-30"),
    "july": ("07-01", "07-31"), "august": ("08-01", "08-31"),
    "september": ("09-01", "09-30"), "october": ("10-01", "10-31"),
    "november": ("11-01", "11-30"), "december": ("12-01", "12-31"),
    "winter": ("12-01", "02-28"), "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"), "fall": ("09-01", "11-30"),
    "autumn": ("09-01", "11-30"),
}


def _resolve_date_range(travel_month: str, duration_days: int) -> tuple:
    """Convert a month/season name into a concrete date range for the next occurrence."""
    month_key = travel_month.lower().strip()
    today = date.today()
    year = today.year

    if month_key in MONTH_TO_DATE_RANGE:
        start_suffix, end_suffix = MONTH_TO_DATE_RANGE[month_key]
        start_str = f"{year}-{start_suffix}"
        end_str = f"{year}-{end_suffix}"

        start = date.fromisoformat(start_str)
        # If the month is in the past, use next year
        if start < today:
            year += 1
            start_str = f"{year}-{start_suffix}"
            end_str = f"{year}-{end_suffix}"

        return (start_str, end_str)

    # Fallback: 3 months from now
    start = today + timedelta(days=90)
    end = start + timedelta(days=90)
    return (start.isoformat(), end.isoformat())


def build_optimizer_state(
    destination: str,
    trip_intent: dict,
    origin: str = "",
) -> dict:
    """Convert discovery output into a TravelState dict for the optimizer."""
    origin = origin or settings.default_origin
    travel_month = trip_intent.get("travel_month", "")
    duration_days = trip_intent.get("duration_days", 7)
    budget_total = trip_intent.get("budget_total", 0)
    companions = trip_intent.get("travel_companions", "solo")

    # Map companions to num_travelers
    companion_map = {"solo": 1, "couple": 2, "family": 4, "group": 4}
    num_travelers = companion_map.get(companions, 1)

    date_range = _resolve_date_range(travel_month, duration_days)

    return {
        "destination": destination,
        "origin": origin,
        "date_range": date_range,
        "duration_days": duration_days,
        "num_travelers": num_travelers,
        "budget_max": float(budget_total) if budget_total else None,
        "priorities": {"weather": 0.35, "flights": 0.25, "hotels": 0.25, "social": 0.15},
        "errors": [],
    }


def bridge_node(state: DiscoveryState) -> dict:
    """LangGraph node: build optimizer state from discovery results."""
    errors = list(state.get("errors", []))
    chosen = state.get("chosen_destination")
    trip_intent = state.get("trip_intent", {})

    if not chosen:
        errors.append("Bridge: no destination chosen")
        return {"optimizer_state": None, "errors": errors}

    optimizer_state = build_optimizer_state(chosen, trip_intent)
    return {
        "optimizer_state": optimizer_state,
        "errors": errors,
    }
