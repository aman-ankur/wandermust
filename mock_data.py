"""Destination-aware mock data generator for demo mode.

Provides realistic fake weather, flight, and hotel data for ~10 popular
destinations with month-aware variation. Unknown destinations get
hash-seeded plausible data.
"""
import hashlib
import random
from datetime import date
from typing import Optional

# --- Destination presets ---
# Format: {city_keyword: {month(1-12): {avg_temp, rain_days, avg_humidity}, flight_range, hotel_range}}

PRESETS = {
    "tokyo": {
        "weather": {
            1: (5.0, 2, 45), 2: (6.0, 3, 48), 3: (10.0, 4, 52),
            4: (15.0, 5, 58), 5: (20.0, 5, 62), 6: (23.0, 7, 72),
            7: (27.0, 5, 75), 8: (28.0, 4, 70), 9: (24.0, 6, 68),
            10: (18.0, 4, 60), 11: (13.0, 3, 55), 12: (7.0, 2, 47),
        },
        "flight_range": (25000, 45000),
        "hotel_range": (6000, 12000),
    },
    "paris": {
        "weather": {
            1: (4.0, 5, 80), 2: (5.0, 4, 77), 3: (9.0, 4, 72),
            4: (12.0, 4, 65), 5: (16.0, 5, 62), 6: (20.0, 4, 58),
            7: (22.0, 3, 55), 8: (22.0, 3, 56), 9: (18.0, 3, 60),
            10: (13.0, 5, 72), 11: (8.0, 5, 80), 12: (5.0, 5, 82),
        },
        "flight_range": (35000, 55000),
        "hotel_range": (8000, 15000),
    },
    "london": {
        "weather": {
            1: (5.0, 6, 82), 2: (5.0, 5, 78), 3: (7.0, 5, 74),
            4: (10.0, 5, 68), 5: (14.0, 5, 65), 6: (17.0, 4, 62),
            7: (19.0, 4, 60), 8: (19.0, 4, 62), 9: (16.0, 4, 66),
            10: (12.0, 5, 74), 11: (8.0, 6, 80), 12: (5.0, 6, 83),
        },
        "flight_range": (30000, 50000),
        "hotel_range": (9000, 16000),
    },
    "new york": {
        "weather": {
            1: (-1.0, 4, 60), 2: (0.0, 4, 58), 3: (5.0, 5, 55),
            4: (12.0, 5, 52), 5: (18.0, 5, 55), 6: (23.0, 5, 60),
            7: (26.0, 4, 62), 8: (25.0, 4, 63), 9: (21.0, 4, 60),
            10: (15.0, 4, 58), 11: (9.0, 4, 62), 12: (3.0, 4, 62),
        },
        "flight_range": (40000, 65000),
        "hotel_range": (10000, 20000),
    },
    "bangkok": {
        "weather": {
            1: (27.0, 1, 60), 2: (28.0, 1, 58), 3: (30.0, 2, 62),
            4: (31.0, 4, 68), 5: (30.0, 8, 75), 6: (29.0, 8, 76),
            7: (29.0, 8, 76), 8: (29.0, 9, 78), 9: (28.0, 10, 80),
            10: (28.0, 8, 78), 11: (27.0, 4, 68), 12: (26.0, 1, 60),
        },
        "flight_range": (12000, 22000),
        "hotel_range": (3000, 7000),
    },
    "dubai": {
        "weather": {
            1: (19.0, 0, 60), 2: (20.0, 1, 58), 3: (23.0, 1, 52),
            4: (27.0, 0, 45), 5: (32.0, 0, 40), 6: (35.0, 0, 42),
            7: (37.0, 0, 48), 8: (37.0, 0, 50), 9: (34.0, 0, 52),
            10: (30.0, 0, 50), 11: (25.0, 0, 55), 12: (21.0, 0, 58),
        },
        "flight_range": (15000, 28000),
        "hotel_range": (5000, 14000),
    },
    "sydney": {
        "weather": {
            1: (24.0, 4, 65), 2: (24.0, 5, 68), 3: (23.0, 5, 66),
            4: (20.0, 4, 62), 5: (17.0, 4, 60), 6: (14.0, 5, 62),
            7: (13.0, 4, 58), 8: (14.0, 3, 55), 9: (17.0, 3, 52),
            10: (19.0, 4, 58), 11: (21.0, 4, 60), 12: (23.0, 4, 63),
        },
        "flight_range": (35000, 60000),
        "hotel_range": (7000, 14000),
    },
    "rome": {
        "weather": {
            1: (8.0, 4, 72), 2: (9.0, 4, 68), 3: (12.0, 4, 62),
            4: (15.0, 4, 58), 5: (20.0, 3, 52), 6: (25.0, 2, 48),
            7: (28.0, 1, 42), 8: (28.0, 1, 44), 9: (24.0, 3, 52),
            10: (19.0, 4, 62), 11: (13.0, 5, 70), 12: (9.0, 5, 74),
        },
        "flight_range": (32000, 52000),
        "hotel_range": (6000, 13000),
    },
    "bali": {
        "weather": {
            1: (27.0, 10, 82), 2: (27.0, 9, 80), 3: (27.0, 8, 78),
            4: (28.0, 5, 72), 5: (28.0, 3, 68), 6: (27.0, 2, 65),
            7: (26.0, 1, 62), 8: (27.0, 1, 60), 9: (27.0, 2, 64),
            10: (28.0, 4, 70), 11: (28.0, 7, 76), 12: (27.0, 9, 80),
        },
        "flight_range": (18000, 32000),
        "hotel_range": (3000, 8000),
    },
    "cape town": {
        "weather": {
            1: (22.0, 1, 50), 2: (22.0, 1, 48), 3: (20.0, 1, 52),
            4: (17.0, 3, 60), 5: (14.0, 5, 68), 6: (12.0, 6, 72),
            7: (11.0, 7, 75), 8: (12.0, 6, 72), 9: (14.0, 4, 65),
            10: (16.0, 3, 58), 11: (19.0, 2, 52), 12: (21.0, 1, 48),
        },
        "flight_range": (28000, 48000),
        "hotel_range": (4000, 10000),
    },
}


def _match_preset(destination: str) -> Optional[dict]:
    """Find a preset that matches the destination (case-insensitive substring)."""
    dest_lower = destination.lower()
    for key, preset in PRESETS.items():
        if key in dest_lower:
            return preset
    return None


def _hash_seed(destination: str) -> int:
    """Generate a deterministic seed from destination name."""
    return int(hashlib.md5(destination.lower().encode()).hexdigest()[:8], 16)


def _jitter(value: float, pct: float = 0.10, rng: Optional[random.Random] = None) -> float:
    """Add ±pct random jitter to a value."""
    r = rng or random
    factor = 1.0 + r.uniform(-pct, pct)
    return round(value * factor, 2)


def get_mock_weather(destination: str, start_date: str, end_date: str) -> dict:
    """Generate mock weather data for a destination and date window."""
    month = date.fromisoformat(start_date).month
    preset = _match_preset(destination)
    seed = _hash_seed(destination + start_date)
    rng = random.Random(seed)

    if preset:
        base_temp, base_rain, base_humidity = preset["weather"].get(month, (24.0, 3, 60))
    else:
        # Fallback: generate plausible data seeded by destination
        base_rng = random.Random(_hash_seed(destination))
        base_temp = base_rng.uniform(10.0, 32.0)
        base_rain = base_rng.randint(0, 8)
        base_humidity = base_rng.uniform(40.0, 80.0)

    return {
        "avg_temp": round(_jitter(base_temp, 0.08, rng), 1),
        "rain_days": max(0, int(_jitter(base_rain, 0.15, rng))),
        "avg_humidity": round(_jitter(base_humidity, 0.08, rng), 1),
    }


def get_mock_flight_price(destination: str, start_date: str) -> dict:
    """Generate mock flight price for a destination and date."""
    preset = _match_preset(destination)
    seed = _hash_seed(destination + start_date)
    rng = random.Random(seed)

    if preset:
        low, high = preset["flight_range"]
    else:
        base_rng = random.Random(_hash_seed(destination))
        mid = base_rng.uniform(20000, 50000)
        low, high = mid * 0.7, mid * 1.3

    min_price = round(rng.uniform(low, high), 2)
    avg_price = round(min_price * rng.uniform(1.05, 1.25), 2)
    return {"min_price": min_price, "avg_price": avg_price}


def get_mock_hotel_price(destination: str, start_date: str) -> dict:
    """Generate mock hotel nightly rate for a destination and date."""
    preset = _match_preset(destination)
    seed = _hash_seed(destination + start_date)
    rng = random.Random(seed)

    if preset:
        low, high = preset["hotel_range"]
    else:
        base_rng = random.Random(_hash_seed(destination))
        mid = base_rng.uniform(4000, 15000)
        low, high = mid * 0.7, mid * 1.3

    avg_nightly = round(rng.uniform(low, high), 2)
    return {"avg_nightly": avg_nightly}


def get_mock_recommendation(destination: str, origin: str, ranked: list[dict]) -> str:
    """Generate a static recommendation string without LLM."""
    if not ranked:
        return "No data available for a recommendation."
    top = ranked[0]
    w = top["window"]
    lines = [
        f"**Best window: {w['start']} to {w['end']}** (score: {top['total_score']:.2f})",
        "",
        f"For your trip from {origin} to {destination}, this window offers the best "
        f"combination of weather, flight prices, and hotel rates.",
        "",
        f"- Weather score: {top['weather_score']:.2f}",
        f"- Estimated flight cost: ~₹{top['estimated_flight_cost']:,.0f}",
        f"- Estimated hotel cost: ~₹{top['estimated_hotel_cost']:,.0f}/night",
    ]
    if len(ranked) > 1:
        r2 = ranked[1]
        w2 = r2["window"]
        lines.append(f"\nRunner-up: {w2['start']} to {w2['end']} (score: {r2['total_score']:.2f})")
    lines.append("\n*This is simulated demo data — not from live APIs.*")
    return "\n".join(lines)
