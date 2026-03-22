# Destination Discovery Agent — Design Spec

## Overview

Add a destination discovery system to Wandermust that answers "where should I travel?" before the existing optimizer answers "when should I go?" The discovery agent is a conversational AI travel advisor that reasons about practical constraints (visa, budget reality, flight connectivity, seasonality) and personal fit (travel history, preferences, trip intent) to generate genuinely useful destination suggestions — not generic listicles.

**Purpose:** Turn Wandermust from a single-purpose optimizer into a full travel planning product: discover → optimize → book.

**Core principle:** Ask few questions, give rich answers. Like a great travel advisor who listens for 2 minutes then gives you 5 perfect options with reasoning.

---

## Architecture

### Two-Graph Design

```
┌─────────────────────────────────────────────┐
│           DISCOVERY GRAPH (new)             │
│                                             │
│  [Check Profile] → exists? → [Load Profile] │
│       │ no                        ↓         │
│       ↓                    [Discovery Agent] │
│  [Onboarding Agent]        (chat loop 3-5Q)  │
│  (travel history +              ↓            │
│   preferences)           [Suggestion Gen]    │
│       ↓                   (rich LLM call)    │
│  [Profile Builder]              ↓            │
│       ↓                  5 suggestions       │
│  [Discovery Agent] ←────────────┘            │
└──────────────┬──────────────────────────────┘
               │ user picks one
               ↓
        [Bridge Function]
        maps DiscoveryState → TravelState
               ↓
┌──────────────────────────────────────────────┐
│        OPTIMIZER GRAPH (existing)            │
│  Supervisor → Weather|Flights|Hotels|Social  │
│            → Scorer → Synthesizer            │
└──────────────────────────────────────────────┘
```

Each graph has its own state type. The bridge function maps between them. The existing optimizer graph is untouched except for accepting optional discovery context in the synthesizer prompt.

---

## The Discovery Problem

Finding where to travel is 4 layered problems:

1. **Feasibility** — Can you actually go there? Visa rules for your passport, flight connectivity from your origin, budget reality (not US-centric pricing)
2. **Fit** — Does it match what you enjoy? Based on travel history patterns + stated preferences
3. **Timing** — Is it good to visit in your travel window? Monsoons, hurricane season, extreme heat/cold, festival alignment
4. **Novelty** — Have you been before? Would something similar-but-new be better? Avoid suggesting what they've already done

---

## Onboarding System

### When It Runs

First time the user enters Discovery mode. Skipped on subsequent visits (profile loaded from SQLite). User can trigger re-onboarding via "Update preferences" button.

### What It Collects

**A) Travel History** — conversational, not a form:
- Agent asks: "Where have you travelled in the last few years? Just list the places — doesn't have to be exact."
- User responds in free text: "Bali last year, Thailand and Sri Lanka in 2023, Japan in 2024"
- LLM parses into structured entries: destination, country, approximate timing, inferred trip type

**B) Profile** — inferred from history + 2-3 targeted questions:
- LLM analyzes history first: "You seem to enjoy Southeast Asia, beach + culture mix, moderate budgets"
- Then asks only what it can't infer: nationality/passport, home city, dealbreakers (e.g., "no 20+ hour flights", "avoid extreme heat")
- Budget range if not inferrable from history

### Single-User Design

This is a personal tool — no auth, no login, consistent with the existing optimizer. The `user_profile` table holds exactly one row (the user). Use `INSERT OR REPLACE` with `id=1` for upserts. `travel_history` has multiple rows (one per trip) but all belong to the single user.

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS user_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- single user, always id=1
    origin TEXT NOT NULL,              -- "Bangalore, India"
    nationality TEXT,                  -- "Indian" (for visa reasoning)
    preferred_climate TEXT,            -- JSON: ["tropical", "Mediterranean"]
    budget_tier TEXT,                  -- "budget" | "moderate" | "premium" | "luxury"
    trip_styles TEXT,                  -- JSON: ["beach", "culture", "food", "adventure"]
    interests TEXT,                    -- JSON: ["street_food", "history", "hiking", "nightlife"]
    dealbreakers TEXT,                 -- JSON: ["long_flights", "extreme_heat", "no_visa_on_arrival"]
    max_flight_hours INTEGER,          -- NULL = no limit
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS travel_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    destination TEXT NOT NULL,         -- "Bali, Indonesia"
    country TEXT NOT NULL,             -- "Indonesia"
    trip_type TEXT,                    -- "beach", "culture", "adventure", "food"
    month_visited INTEGER,             -- 1-12
    year_visited INTEGER,
    duration_days INTEGER,
    status TEXT DEFAULT 'visited',     -- "visited" | "searched" | "booked"
    notes TEXT,                        -- free-form
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

---

## Discovery Conversation Flow

### The Chat Loop

The discovery agent asks 3-5 questions maximum, adapting based on profile context. It does NOT ask what it already knows from the profile.

**Question 1 — Trip intent:**
> "What's pulling you to travel this time?"
> Options: Unwind/relax | Explore somewhere new | Food trip | Adventure/outdoors | Special occasion | No idea — surprise me

**Question 2 — Geography (adaptive based on history):**
> "You've done a lot of Southeast Asia and Japan. Want to..."
> Options: Go deeper in Asia (new countries) | Try a new region entirely | Revisit a favorite with a twist

**Question 3 — Timing:**
> "Any rough timing in mind?"
> Options: Next month | 2-3 months out | Flexible — find the best window | Specific dates: ___

**Question 4 — Budget (skip if known from profile):**
> "Budget per person for flights + hotels?"
> Options: Under ₹50K | ₹50K-1L | ₹1-2L | Flexible

**Question 5 — Duration (skip if obvious from context):**
> "How many days?"
> Options: Long weekend (3-4) | Week (5-7) | Extended (8-14) | Open-ended

The agent skips questions when the answer is already known or inferrable. A returning user with a clear profile might only get 2-3 questions.

### The "Surprise Me" Path

If user picks "No idea — surprise me", the agent skips preference questions and generates diverse suggestions across different trip types, explicitly varied by region and style. It leans heavily on history to avoid repetition.

---

## Suggestion Generation

### The Rich LLM Call

After the conversation, the discovery agent makes a single LLM call with a carefully constructed prompt:

```
CONTEXT:
- Origin: {profile.origin}
- Passport: {profile.nationality}
- Travel history: {formatted history list}
- Inferred preferences: {profile.trip_styles}, {profile.interests}
- Dealbreakers: {profile.dealbreakers}

THIS SESSION:
- Trip intent: {conversation.trip_intent}
- Region preference: {conversation.region_pref}
- Timing: {conversation.timing}
- Budget: {conversation.budget}
- Duration: {conversation.duration_days} days

CONSTRAINTS TO REASON ABOUT:
1. VISA: {nationality} passport — prioritize visa-on-arrival, e-visa,
   or visa-free. Flag any requiring embassy visits.
2. FLIGHTS: From {origin} — reasonable connectivity (direct or 1 stop).
   No 24hr+ journeys for trips under 10 days.
3. BUDGET REALITY: {budget} must cover flights + hotels for {duration} nights.
   Consider actual flight costs from {origin}, not US-centric pricing.
4. SEASON: {timing} — avoid monsoon/hurricane/extreme weather zones.
   Flag festivals or peak seasons (positive or negative).
5. NOVELTY: User has visited: {history}. Suggest NEW places.
   Note patterns (regions they like, styles they repeat).
6. FIT: Match demonstrated taste but in new geography where possible.

OUTPUT FORMAT (JSON):
Return exactly 5 destinations, each with:
- destination: "City, Country"
- country_code: "XX"
- why_you: personalized 2-3 sentence pitch referencing their history/preferences
- visa_situation: brief visa info for their passport
- flight_estimate: cost range + duration from their origin
- timing_verdict: why their chosen timing works (or caveats)
- budget_fit: estimated total trip cost vs their budget
- novelty_note: how it differs from places they've been
- tags: ["beach", "food", "culture", ...]
- confidence: 0.0-1.0 (how well it matches overall)
```

### Example Output

```json
{
  "suggestions": [
    {
      "destination": "Tbilisi, Georgia",
      "country_code": "GE",
      "why_you": "You loved food scenes in Japan and Thailand — Georgian cuisine is a revelation. Beach + mountains + old town culture in one trip, and wine country rivals anything in Europe at a fraction of the cost.",
      "visa_situation": "E-visa for Indian passport, approved in ~5 days, $20",
      "flight_estimate": "₹25-35K round trip via Dubai/Doha, 8-10 hours",
      "timing_verdict": "September is perfect — warm (24°C), dry, wine harvest season. Tbilisi Wine Festival happens mid-Sept.",
      "budget_fit": "₹60-80K total easily. Hotels ₹2-3K/night, meals ₹500-800.",
      "novelty_note": "Completely different from Asia — Caucasus mountains, Soviet-era architecture, European feel at Asian prices. A region most Indian travelers haven't explored.",
      "tags": ["food", "culture", "wine", "mountains"],
      "confidence": 0.92
    }
  ]
}
```

---

## Discovery → Optimizer Bridge

### State Mapping

```python
def bridge_discovery_to_optimizer(
    discovery: DiscoveryState,
    chosen: DestinationSuggestion
) -> TravelState:
    return TravelState(
        destination=chosen.destination,
        origin=discovery.profile.origin,
        date_range=discovery.date_range,
        duration_days=discovery.duration_days,
        budget_max=discovery.budget_max,
        priorities=infer_priorities(discovery.trip_intent),
        discovery_context={
            "trip_intent": discovery.trip_intent,
            "why_chosen": chosen.why_you,
            "tags": chosen.tags,
        }
    )
```

### Priority Inference from Trip Intent

| Trip intent | Weather | Flights | Hotels | Social |
|------------|---------|---------|--------|--------|
| Unwind/relax | 0.40 | 0.20 | 0.25 | 0.15 |
| Explore new | 0.25 | 0.25 | 0.20 | 0.30 |
| Food trip | 0.20 | 0.25 | 0.20 | 0.35 |
| Adventure | 0.45 | 0.20 | 0.15 | 0.20 |
| Special occasion | 0.30 | 0.15 | 0.35 | 0.20 |
| Surprise me | 0.25 | 0.25 | 0.25 | 0.25 |
| Budget-focused | 0.15 | 0.40 | 0.35 | 0.10 |

User can override on the optimizer page. These are smart defaults.

### Synthesizer Enrichment

The existing synthesizer receives `discovery_context` and weaves it into the recommendation:

> "Georgia in mid-September is your best window. Flights from Bangalore via Doha are cheapest Sept 8-15 (₹28K). This coincides with the Tbilisi Wine Festival — perfect for the food exploration you're after."

vs the current generic output.

### History Logging

After optimizer completes, the search is auto-logged to `travel_history` with `status='searched'`. A "Mark as booked" button changes status to `booked`, improving future discovery.

---

## Data Models

### DiscoveryState

```python
class DiscoveryState(TypedDict, total=False):
    # Profile (loaded or built during onboarding)
    profile: dict               # UserProfile as dict
    history: List[dict]         # List of TravelHistoryEntry as dicts
    profile_exists: bool

    # Conversation state
    conversation_messages: List[dict]  # chat history
    trip_intent: str            # "explore", "relax", "food", "adventure", "surprise", "occasion"
    region_preference: str      # "same_region", "new_region", "revisit"
    date_range: Optional[Tuple[str, str]]  # None when user picks "Flexible"
    duration_days: int
    budget_max: float           # upper bound of selected range (e.g., "50K-1L" → 100000.0)
    questions_asked: int        # track to enforce 3-5 limit

    # Output
    suggestions: List[dict]     # DestinationSuggestion dicts
    chosen_destination: str     # user's pick

    errors: Annotated[List[str], operator.add]
```

**Budget mapping:** Categorical ranges map to `budget_max` as the upper bound: "Under ₹50K" → 50000, "₹50K-1L" → 100000, "₹1-2L" → 200000, "Flexible" → None.

**Flexible timing:** When user picks "Flexible — find the best window", `date_range` is set to None. The bridge function defaults to the next 6 months from today when passing to the optimizer.

### Pydantic Models

```python
class UserProfile(BaseModel):
    origin: str
    nationality: str
    preferred_climate: List[str] = []
    budget_tier: str = "moderate"
    trip_styles: List[str] = []
    interests: List[str] = []
    dealbreakers: List[str] = []
    max_flight_hours: int | None = None

class TravelHistoryEntry(BaseModel):
    destination: str
    country: str
    trip_type: str | None = None
    month_visited: int | None = None
    year_visited: int | None = None
    duration_days: int | None = None
    status: str = "visited"  # visited | searched | booked

class DestinationSuggestion(BaseModel):
    destination: str
    country_code: str
    why_you: str
    visa_situation: str
    flight_estimate: str
    timing_verdict: str
    budget_fit: str
    novelty_note: str
    tags: List[str]
    confidence: float
```

---

## Discovery Graph Definition

```python
# discovery_graph.py

def build_discovery_graph():
    graph = StateGraph(DiscoveryState)

    graph.add_node("check_profile", check_profile_node)
    graph.add_node("onboarding", onboarding_node)     # chat loop
    graph.add_node("build_profile", build_profile_node)
    graph.add_node("load_profile", load_profile_node)
    graph.add_node("discovery_chat", discovery_chat_node)  # Q&A loop
    graph.add_node("generate_suggestions", generate_suggestions_node)

    graph.set_entry_point("check_profile")

    # Conditional: profile exists?
    graph.add_conditional_edges("check_profile", route_profile, {
        "exists": "load_profile",
        "missing": "onboarding",
    })

    graph.add_edge("onboarding", "build_profile")
    graph.add_edge("build_profile", "discovery_chat")
    graph.add_edge("load_profile", "discovery_chat")

    # discovery_chat is a human-in-the-loop node
    # It interrupts for user input after each question
    graph.add_conditional_edges("discovery_chat", route_discovery, {
        "need_more": "discovery_chat",  # ask another question
        "ready": "generate_suggestions",
    })

    graph.add_edge("generate_suggestions", END)

    return graph.compile(interrupt_before=["discovery_chat", "onboarding"])
```

### Human-in-the-Loop

Both `onboarding` and `discovery_chat` use LangGraph's interrupt mechanism. The pattern is **re-entrant nodes**: each node is invoked multiple times, once per question-answer cycle.

**How it works:**
1. Node runs → examines state → generates next question → calls `interrupt()` with the question payload
2. Graph execution pauses, Streamlit displays the question to the user
3. User responds → Streamlit calls `graph.invoke(Command(resume=user_answer))`
4. Same node re-enters → reads the answer from resumed value → updates state → decides: call `interrupt()` again (another question) or return (done, proceed to next node)

**State tracking:** Each node uses `questions_asked` counter and `conversation_messages` list in state to track where it is in the conversation. The node is stateless between invocations — all context is in `DiscoveryState`.

**Onboarding re-entry flow:**
- Invocation 1: "Where have you travelled?" → interrupt
- Invocation 2: receives history text → "What's your home city?" → interrupt
- Invocation 3: receives origin → "Any dealbreakers?" → interrupt
- Invocation 4: receives dealbreakers → return (done, proceed to build_profile)

**Discovery chat re-entry flow:**
- Invocation 1: loads profile, generates first question based on what's missing → interrupt
- Invocation 2-4: adapts next question based on answers so far → interrupt or return
- Final invocation: has enough info → return (proceed to generate_suggestions)

```python
# Conceptual pattern for a re-entrant node
def discovery_chat_node(state: DiscoveryState) -> Command[Literal["discovery_chat", "generate_suggestions"]]:
    # Check if we have enough info
    if has_enough_info(state):
        return Command(goto="generate_suggestions", update=state)

    # Generate next question based on what we know
    question = generate_next_question(state)

    # Interrupt and wait for user
    user_answer = interrupt(question)

    # Process answer, update state
    updated = process_answer(state, user_answer)
    return Command(goto="discovery_chat", update=updated)
```

---

## Streamlit UI Changes

### Mode Toggle

Top of the app, two tabs:

```
[🔍 Discover Where]    [📅 Optimize When]
```

### Discover Mode — Chat Interface

- Chat-style UI using `st.chat_message` / `st.chat_input`
- Agent messages show as cards with options (rendered as buttons)
- After suggestions are generated: 5 destination cards with key info
- Each card has "Explore this →" button that triggers the bridge + optimizer

### Optimizer Mode

Current UI, unchanged. New optional field: "Coming from discovery" badge if destination was auto-filled.

### Profile Management

- Sidebar: "👤 My Travel Profile" expander
- Shows current profile summary + travel history
- "Update preferences" button → re-runs onboarding
- "Add past trip" → quick form to add to history

---

## New & Modified Files

### New Files

| File | Purpose |
|------|---------|
| `discovery_graph.py` | LangGraph definition for the discovery flow |
| `discovery_models.py` | DiscoveryState, UserProfile, TravelHistoryEntry, DestinationSuggestion |
| `agents/onboarding.py` | Chat agent for first-time profile + history collection |
| `agents/discovery.py` | Conversational Q&A agent + suggestion generation |
| `services/profile_manager.py` | High-level interface over db.py: load/save UserProfile Pydantic models, infer patterns from travel_history (e.g., "prefers tropical, moderate budget"), format profile for LLM prompts |

### Modified Files

| File | Change |
|------|--------|
| `db.py` | Add `user_profile` and `travel_history` tables + raw SQL CRUD methods (profile_manager.py wraps these with Pydantic) |
| `models.py` | Add `discovery_context: Optional[Dict[str, Any]]` to `TravelState` (keys: `trip_intent`, `why_chosen`, `tags`) |
| `agents/synthesizer.py` | Accept `discovery_context` in prompt for personalized recommendations |
| `app.py` | Add Discover mode tab, chat UI, mode toggle, profile sidebar, bridge logic |
| `config.py` | No changes needed — reuses existing `OPENROUTER_API_KEY` and model |

### NOT Modified

- `graph.py` — existing optimizer graph untouched
- `agents/supervisor.py`, `weather.py`, `flights.py`, `hotels.py`, `social.py`, `scorer.py` — untouched
- `services/amadeus_client.py`, `weather_client.py`, `geocoding.py` — untouched

---

## Error Handling

Follows existing patterns:
- Errors accumulated in `state["errors"]`, never raised
- If LLM fails during onboarding: retry with backoff, fall back to form-based input
- If LLM fails during discovery chat: retry, fall back to simplified suggestions
- If suggestion generation fails: return generic "popular destinations for your profile" fallback
- If bridge fails: show error, let user manually enter destination in optimizer mode
- Profile DB errors: log and continue without persistence (session-only fallback)

---

## Testing

- Unit tests for `agents/onboarding.py` — mock LLM, verify profile extraction from free text
- Unit tests for `agents/discovery.py` — mock LLM, verify question flow adapts to profile
- Unit tests for suggestion generation — mock LLM, verify constraint reasoning appears in output
- Unit tests for `services/profile_manager.py` — CRUD operations, pattern inference
- Unit tests for bridge function — verify DiscoveryState → TravelState mapping, priority inference
- Unit tests for `discovery_models.py` — Pydantic validation
- Integration test: full discovery graph with mocked LLM responses
- Integration test: discovery → bridge → optimizer pipeline end-to-end with mocks
- Existing tests remain untouched (optimizer graph unchanged)

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Destination knowledge | Pure LLM generation | LLMs have broad, up-to-date travel knowledge. No need to maintain a curated DB. |
| Visa/flight reasoning | LLM prompt, not APIs | Good enough for suggestion phase. Real validation happens in optimizer (Amadeus). |
| Conversation limit | 3-5 questions max | Ask few, give rich. Users hate 20-question surveys. |
| Profile storage | SQLite (existing DB) | Consistent with existing pattern, no new dependencies. |
| Two-graph design | Separate DiscoveryGraph | Clean separation of concerns, existing optimizer untouched, independently testable. |
| Human-in-the-loop | LangGraph interrupt | Native pattern for conversational agents, well-supported. |
| History tracking | Auto-log searches + manual "booked" | Builds history passively, improves future discovery without friction. |
