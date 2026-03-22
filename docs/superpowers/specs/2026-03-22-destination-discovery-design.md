# Destination Discovery — Design Spec

## Overview

A conversational destination discovery system that answers "where should I travel?" — the complement to the existing optimizer ("when should I travel?"). Uses LangGraph's `interrupt()` and `Command` for human-in-the-loop chat, with an LLM-powered suggestion generator that reasons about visa requirements, budget, flights, seasonality, and user preferences.

**Purpose:** Extend the Travel Optimizer with a "Discover Where" mode that feeds chosen destinations into the existing "Optimize When" pipeline.

**Stack:** Same as existing — Python, LangGraph, Streamlit, OpenRouter/Claude, plus new conversational patterns.

---

## Architecture

### New Agents (4 nodes in the discovery graph)

| Agent | Type | Role |
|-------|------|------|
| Onboarding Agent | Conversational (LLM) | Collects travel history + preferences (first-time only) |
| Discovery Chat Agent | Conversational (LLM) | Asks 3-5 adaptive questions about trip intent |
| Suggestion Generator | LLM-powered | Reasons about visa, budget, flights, seasonality to suggest destinations |
| Bridge | Deterministic | Feeds chosen destination into existing optimizer pipeline |

### Flow

```
[User enters "Discover Where" mode]
     |
[Check user profile exists?]
     |--- NO ---> [Onboarding Agent] (interrupt/resume loop, 3-5 questions)
     |--- YES --> skip
     |
[Discovery Chat Agent] (interrupt/resume loop, 3-5 adaptive questions)
     |
[Suggestion Generator] → 3-5 destination suggestions with reasoning
     |
[User picks one]
     |
[Bridge] → populates TravelState and hands off to existing optimizer graph
```

**LLM usage:** Onboarding, Discovery Chat, and Suggestion Generator all use LLM. Bridge is deterministic.

**Human-in-the-loop:** Uses LangGraph's `interrupt()` to pause for user input and `Command(resume=...)` to continue.

---

## State & Data Model

```python
class UserProfile(BaseModel):
    """Persisted in SQLite. Created during onboarding."""
    user_id: str = "default"
    travel_history: list[str] = []       # ["Japan", "Thailand", "Italy"]
    preferences: dict = {}                # {"climate": "warm", "pace": "relaxed", ...}
    budget_level: str = "moderate"        # "budget" | "moderate" | "luxury"
    passport_country: str = "IN"          # ISO country code
    created_at: str = ""

class DiscoveryState(TypedDict, total=False):
    # User profile (loaded or built during onboarding)
    user_profile: dict

    # Onboarding
    onboarding_complete: bool
    onboarding_messages: list[dict]       # [{role, content}]

    # Discovery conversation
    discovery_messages: list[dict]        # [{role, content}]
    discovery_complete: bool
    trip_intent: dict                     # extracted from conversation

    # Suggestions
    suggestions: list[dict]              # [{destination, country, reason, ...}]
    chosen_destination: str | None

    # Bridge to optimizer
    optimizer_state: dict | None          # populated TravelState for handoff

    # Errors
    errors: list[str]
```

---

## Onboarding Agent

**Trigger:** First time user (no profile in DB).

**Questions (3-5, adaptive):**
1. "What countries have you visited before?"
2. "What kind of climate do you prefer?" (warm/cold/moderate/tropical)
3. "What's your typical travel style?" (adventure/relaxation/culture/foodie)
4. "What's your comfort budget level?" (budget/moderate/luxury)
5. "What passport do you hold?" (for visa reasoning)

**Persistence:** Saves `UserProfile` to SQLite after completion.

**Skip logic:** If profile exists in DB, skip entirely.

---

## Discovery Chat Agent

**Always runs.** Asks 3-5 questions about THIS trip:

1. "When are you thinking of traveling?" (month/season)
2. "How many days do you have?"
3. "Any specific interests for this trip?" (beaches, mountains, history, food, nightlife)
4. "Any constraints?" (visa-free only, direct flights only, under X budget)
5. "Traveling solo or with others?"

**Adaptive:** If user profile shows they always travel budget, skip budget question. If they mentioned a region interest, drill deeper.

**Output:** Structured `trip_intent` dict extracted by LLM.

---

## Suggestion Generator

**Input:** `user_profile` + `trip_intent`

**Reasoning dimensions:**
- **Visa:** Based on passport country, filter visa-free or e-visa destinations
- **Budget:** Match budget level to destination cost tier
- **Flights:** Consider origin city for flight connectivity
- **Seasonality:** Match travel month to destination's best season
- **Preferences:** Weight toward user's preferred climate/style
- **History:** Avoid recently visited, suggest similar-but-new

**Output:** 3-5 suggestions, each with:
```python
{
    "destination": "Tbilisi, Georgia",
    "country": "Georgia",
    "reason": "Visa-free for Indian passports, ...",
    "estimated_budget_per_day": 4000,
    "best_months": [5, 6, 9, 10],
    "match_score": 0.87,
    "tags": ["culture", "food", "budget-friendly"]
}
```

---

## Bridge Function

Converts chosen destination + trip intent into a `TravelState` dict:
- Sets `destination`, `origin`, `date_range`, `duration_days` from trip intent
- Sets default priorities
- Returns state ready for `build_graph().invoke(state)`

---

## Streamlit UI Changes

### Mode Toggle
- Radio button at top: "🔍 Discover Where" | "📅 Optimize When"
- Default: Optimize When (existing behavior unchanged)

### Discover Where Mode
- Chat interface using `st.chat_message` and `st.chat_input`
- Shows onboarding messages if first-time
- Shows discovery questions
- Shows suggestion cards
- "Use this destination" button → switches to Optimize When with destination pre-filled

### Optimize When Mode
- Existing sidebar + results (unchanged)

---

## Testing Strategy

- Unit tests for each agent with mocked LLM
- DB tests for user_profile CRUD
- Integration test for discovery graph with mocked LLM
- Mock agents for demo mode
- All existing tests must continue to pass
