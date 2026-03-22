from agents.supervisor import generate_candidate_windows
from datetime import date

def test_generates_correct_number():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    assert 3 <= len(windows) <= 5

def test_window_duration():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    for w in windows:
        d = (date.fromisoformat(w["end"]) - date.fromisoformat(w["start"])).days
        assert d == 7

def test_windows_within_range():
    windows = generate_candidate_windows("2026-06-01", "2026-06-30", 7)
    for w in windows:
        assert w["start"] >= "2026-06-01"
        assert w["end"] <= "2026-06-30"

def test_wide_range():
    windows = generate_candidate_windows("2026-06-01", "2026-09-30", 7)
    assert len(windows) >= 10
