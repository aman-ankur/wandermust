from __future__ import annotations
from pydantic import BaseModel
from typing import TypedDict

class CandidateWindow(BaseModel):
    start: str
    end: str

class WeatherResult(BaseModel):
    window: CandidateWindow
    avg_temp: float
    rain_days: int
    avg_humidity: float
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None

class FlightResult(BaseModel):
    window: CandidateWindow
    min_price: float
    avg_price: float
    currency: str = "INR"
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None

class HotelResult(BaseModel):
    window: CandidateWindow
    avg_nightly: float
    currency: str = "INR"
    score: float = 0.0
    is_historical: bool = False
    fetched_at: str | None = None

class RankedWindow(BaseModel):
    window: CandidateWindow
    weather_score: float
    flight_score: float
    hotel_score: float
    total_score: float
    estimated_flight_cost: float = 0.0
    estimated_hotel_cost: float = 0.0
    has_historical_data: bool = False

class TravelState(TypedDict, total=False):
    destination: str
    origin: str
    date_range: tuple[str, str]
    duration_days: int
    num_travelers: int
    budget_max: float | None
    priorities: dict[str, float]
    candidate_windows: list[dict]
    weather_data: list[dict]
    flight_data: list[dict]
    hotel_data: list[dict]
    ranked_windows: list[dict]
    recommendation: str
    errors: list[str]
