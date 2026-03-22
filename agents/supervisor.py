from datetime import date, timedelta
from models import TravelState

def generate_candidate_windows(start_date, end_date, duration_days=7):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows = []
    current = start
    while current + timedelta(days=duration_days) <= end:
        windows.append({
            "start": current.isoformat(),
            "end": (current + timedelta(days=duration_days)).isoformat(),
        })
        current += timedelta(days=7)
    return windows

def supervisor_node(state: TravelState) -> dict:
    windows = generate_candidate_windows(
        state["date_range"][0], state["date_range"][1],
        state.get("duration_days", 7))
    return {"candidate_windows": windows, "errors": state.get("errors", [])}
