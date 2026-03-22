from __future__ import annotations
import operator
from pydantic import BaseModel
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

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
    social_score: float = 0.0
    total_score: float
    estimated_flight_cost: float = 0.0
    estimated_hotel_cost: float = 0.0
    has_historical_data: bool = False

class TravelState(TypedDict, total=False):
    destination: str
    origin: str
    date_range: Tuple[str, str]
    duration_days: int
    num_travelers: int
    budget_max: Optional[float]
    priorities: Dict[str, float]
    candidate_windows: List[dict]
    weather_data: List[dict]
    flight_data: List[dict]
    hotel_data: List[dict]
    social_data: List[dict]
    social_insights: List[dict]
    ranked_windows: List[dict]
    recommendation: str
    discovery_context: Optional[dict]
    errors: Annotated[List[str], operator.add]


# --- Discovery Models ---

class UserProfile(BaseModel):
    """User travel profile, persisted in SQLite. Created during onboarding."""
    user_id: str = "default"
    travel_history: List[str] = []
    preferences: Dict = {}
    budget_level: str = "moderate"
    passport_country: str = "IN"
    created_at: str = ""


class DestinationSuggestion(BaseModel):
    """A single destination suggestion from the suggestion generator."""
    destination: str
    country: str
    reason: str
    estimated_budget_per_day: float = 0.0
    best_months: List[int] = []
    match_score: float = 0.0
    tags: List[str] = []


class DiscoveryState(TypedDict, total=False):
    # User profile (loaded or built during onboarding)
    user_profile: dict

    # Onboarding
    onboarding_complete: bool
    onboarding_messages: Annotated[List[dict], operator.add]

    # Discovery conversation
    discovery_messages: Annotated[List[dict], operator.add]
    discovery_complete: bool
    trip_intent: dict

    # Suggestions
    suggestions: List[dict]
    chosen_destination: Optional[str]

    # Bridge to optimizer
    optimizer_state: Optional[dict]

    # Errors
    errors: Annotated[List[str], operator.add]
